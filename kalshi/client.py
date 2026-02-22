"""
Kalshi API client with signed authentication, rate limiting, and endpoint routing.
"""
from __future__ import annotations

import base64
import logging
import os
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable, Dict, Iterable, List, Optional
from urllib.parse import urlparse

import requests

from .router import KalshiDataRouter


def _normalize_env(value: str) -> str:
    """Internal helper for normalize env."""
    v = str(value).lower().strip()
    if v in {"prod", "production", "live"}:
        return "prod"
    return "demo"


@dataclass
class RetryPolicy:
    """HTTP retry settings for Kalshi API requests."""
    max_retries: int = 4
    backoff_seconds: float = 0.5
    timeout_seconds: float = 20.0


@dataclass
class RateLimitPolicy:
    """Token-bucket rate-limit settings for Kalshi API access."""
    requests_per_second: float = 6.0
    burst: int = 2


class RequestLimiter:
    """
    Lightweight token-bucket limiter with runtime limit updates.
    """

    def __init__(self, policy: Optional[RateLimitPolicy] = None):
        """Initialize RequestLimiter."""
        cfg = policy or RateLimitPolicy()
        self.requests_per_second = float(max(cfg.requests_per_second, 0.1))
        self.burst = float(max(cfg.burst, 1))
        self._tokens = self.burst
        self._last_refill = time.monotonic()
        self._lock = threading.Lock()

    def _refill(self) -> None:
        """Internal helper for refill."""
        now = time.monotonic()
        elapsed = max(0.0, now - self._last_refill)
        self._last_refill = now
        self._tokens = min(self.burst, self._tokens + elapsed * self.requests_per_second)

    def acquire(self) -> None:
        """acquire."""
        while True:
            with self._lock:
                self._refill()
                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    return
                deficit = 1.0 - self._tokens
                wait_for = deficit / max(self.requests_per_second, 1e-6)
            time.sleep(max(wait_for, 0.01))

    def update_rate(self, requests_per_second: float, burst: Optional[int] = None) -> None:
        """Update rate in response to input changes."""
        with self._lock:
            self.requests_per_second = float(max(requests_per_second, 0.1))
            if burst is not None:
                self.burst = float(max(int(burst), 1))
                self._tokens = min(self._tokens, self.burst)

    def update_from_account_limits(self, payload: Dict[str, object]) -> None:
        """
        Attempt to derive limiter settings from API account-limit payloads.
        """
        candidates = [payload]
        nested = payload.get("limits")
        if isinstance(nested, dict):
            candidates.append(nested)

        rps = None
        burst = None
        for block in candidates:
            if not isinstance(block, dict):
                continue
            if "max_requests_per_second" in block:
                try:
                    rps = float(block["max_requests_per_second"])
                except (TypeError, ValueError):
                    pass
            if "requests_per_second" in block and rps is None:
                try:
                    rps = float(block["requests_per_second"])
                except (TypeError, ValueError):
                    pass
            if "max_requests_per_minute" in block and rps is None:
                try:
                    rps = float(block["max_requests_per_minute"]) / 60.0
                except (TypeError, ValueError, ZeroDivisionError):
                    pass
            if "burst" in block:
                try:
                    burst = int(block["burst"])
                except (TypeError, ValueError):
                    pass

        if rps is not None:
            self.update_rate(rps, burst=burst)


class KalshiSigner:
    """
    Signs Kalshi requests using RSA-PSS SHA256.

    Uses the ``cryptography`` library for in-process signing (fast, no temp
    files).  Falls back to OpenSSL subprocess only when ``cryptography`` is
    not installed.

    Message format (exact):
        <timestamp_ms><HTTP_METHOD><path>
    """

    def __init__(
        self,
        access_key: str,
        private_key_path: Optional[str] = None,
        private_key_pem: Optional[str] = None,
        passphrase: Optional[str] = None,
        sign_func: Optional[Callable[[bytes], bytes | str]] = None,
    ):
        """Initialize KalshiSigner."""
        self.access_key = str(access_key).strip()
        self.private_key_path = str(private_key_path).strip() if private_key_path else ""
        self.private_key_pem = private_key_pem
        self.passphrase = passphrase
        self.sign_func = sign_func
        # Cache the loaded private key object for reuse across requests.
        self._cached_key = None
        self._crypto_available = False
        try:
            from cryptography.hazmat.primitives import hashes, serialization
            from cryptography.hazmat.primitives.asymmetric import padding as asym_padding
            self._crypto_available = True
        except ImportError:
            pass

    def available(self) -> bool:
        """Return whether the resource is available in the current runtime."""
        return bool(self.access_key and (self.sign_func or self.private_key_path or self.private_key_pem))

    @staticmethod
    def _canonical_path(path: str) -> str:
        """Internal helper for canonical path."""
        parsed = urlparse(str(path))
        out = parsed.path or "/"
        return "/" + out.lstrip("/")

    def _load_private_key(self):
        """Load and cache the private key using the cryptography library."""
        if self._cached_key is not None:
            return self._cached_key

        from cryptography.hazmat.primitives.serialization import load_pem_private_key

        pw = self.passphrase.encode("utf-8") if self.passphrase else None

        if self.private_key_path and os.path.exists(self.private_key_path):
            with open(self.private_key_path, "rb") as f:
                pem_data = f.read()
        elif self.private_key_pem:
            pem_data = self.private_key_pem.encode("utf-8") if isinstance(self.private_key_pem, str) else self.private_key_pem
        else:
            raise RuntimeError("Kalshi signer missing private key material.")

        self._cached_key = load_pem_private_key(pem_data, password=pw)
        return self._cached_key

    def _sign_with_cryptography(self, message: bytes) -> bytes:
        """Sign using the in-process cryptography library (preferred)."""
        from cryptography.hazmat.primitives import hashes
        from cryptography.hazmat.primitives.asymmetric import padding as asym_padding, utils as asym_utils

        key = self._load_private_key()
        signature = key.sign(
            message,
            asym_padding.PSS(
                mgf=asym_padding.MGF1(hashes.SHA256()),
                salt_length=asym_padding.PSS.MAX_LENGTH,
            ),
            hashes.SHA256(),
        )
        return signature

    def _sign_with_openssl(self, message: bytes, key_path: str) -> bytes:
        """Fallback: sign using OpenSSL subprocess."""
        cmd = [
            "openssl",
            "dgst",
            "-sha256",
            "-sign",
            key_path,
            "-sigopt",
            "rsa_padding_mode:pss",
            "-sigopt",
            "rsa_pss_saltlen:-1",
        ]
        if self.passphrase:
            cmd.extend(["-passin", f"pass:{self.passphrase}"])
        proc = subprocess.run(cmd, input=message, capture_output=True)
        if proc.returncode == 0:
            return proc.stdout

        cmd2 = [
            "openssl",
            "pkeyutl",
            "-sign",
            "-inkey",
            key_path,
            "-pkeyopt",
            "rsa_padding_mode:pss",
            "-pkeyopt",
            "rsa_pss_saltlen:-1",
            "-pkeyopt",
            "digest:sha256",
        ]
        if self.passphrase:
            cmd2.extend(["-passin", f"pass:{self.passphrase}"])
        proc2 = subprocess.run(cmd2, input=message, capture_output=True)
        if proc2.returncode == 0:
            return proc2.stdout

        stderr = (proc.stderr or b"") + b"\n" + (proc2.stderr or b"")
        raise RuntimeError(f"OpenSSL signing failed: {stderr.decode(errors='ignore').strip()}")

    def sign(self, timestamp_ms: str, method: str, path: str) -> str:
        """sign."""
        canonical_path = self._canonical_path(path)
        payload = f"{timestamp_ms}{method.upper()}{canonical_path}".encode("utf-8")

        if self.sign_func is not None:
            out = self.sign_func(payload)
            if isinstance(out, str):
                return out
            return base64.b64encode(bytes(out)).decode("ascii")

        if not self.private_key_path and not self.private_key_pem:
            raise RuntimeError("Kalshi signer missing private key material.")

        # Prefer in-process cryptography library (no temp files, no subprocess).
        if self._crypto_available:
            try:
                raw_sig = self._sign_with_cryptography(payload)
                return base64.b64encode(raw_sig).decode("ascii")
            except (ValueError, TypeError, RuntimeError):
                pass  # Fall through to OpenSSL subprocess

        # OpenSSL subprocess fallback
        key_path = self.private_key_path
        if key_path:
            raw_sig = self._sign_with_openssl(payload, key_path)
        else:
            with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False) as tf:
                tf.write(str(self.private_key_pem))
                tf.flush()
                tmp_path = tf.name
            try:
                raw_sig = self._sign_with_openssl(payload, tmp_path)
            finally:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

        return base64.b64encode(raw_sig).decode("ascii")


class KalshiClient:
    """
    Kalshi HTTP wrapper with:
      - RSA-PSS signed auth headers
      - demo/prod environment separation
      - request throttling + 429 backoff
      - historical/live route selection
    """

    _DEFAULT_BASE_URLS = {
        "demo": "https://demo-api.kalshi.co/trade-api/v2",
        "prod": "https://api.elections.kalshi.com/trade-api/v2",
    }

    def __init__(
        self,
        base_url: Optional[str] = None,
        environment: Optional[str] = None,
        historical_base_url: Optional[str] = None,
        historical_cutoff_ts: Optional[str] = None,
        access_key: Optional[str] = None,
        private_key_path: Optional[str] = None,
        private_key_pem: Optional[str] = None,
        private_key_passphrase: Optional[str] = None,
        signer: Optional[KalshiSigner] = None,
        retry_policy: Optional[RetryPolicy] = None,
        rate_limit_policy: Optional[RateLimitPolicy] = None,
        limiter: Optional[RequestLimiter] = None,
        router: Optional[KalshiDataRouter] = None,
        session: Optional[requests.Session] = None,
        logger: Optional[logging.Logger] = None,
        # Legacy aliases kept for compatibility.
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
    ):
        """Initialize KalshiClient."""
        env = _normalize_env(environment or os.environ.get("KALSHI_ENV", "demo"))
        base = (
            base_url
            or os.environ.get(f"KALSHI_{env.upper()}_API_BASE_URL")
            or os.environ.get("KALSHI_API_BASE_URL")
            or self._DEFAULT_BASE_URLS[env]
        )
        self.environment = env
        self.base_url = str(base).rstrip("/")

        resolved_access_key = (
            access_key
            or api_key
            or os.environ.get(f"KALSHI_{env.upper()}_ACCESS_KEY")
            or os.environ.get("KALSHI_ACCESS_KEY", "")
        )
        resolved_key_path = (
            private_key_path
            or os.environ.get(f"KALSHI_{env.upper()}_PRIVATE_KEY_PATH")
            or os.environ.get("KALSHI_PRIVATE_KEY_PATH", "")
        )
        resolved_key_pem = (
            private_key_pem
            or api_secret
            or os.environ.get(f"KALSHI_{env.upper()}_PRIVATE_KEY")
            or os.environ.get("KALSHI_PRIVATE_KEY")
        )

        self.signer = signer or KalshiSigner(
            access_key=str(resolved_access_key),
            private_key_path=str(resolved_key_path),
            private_key_pem=resolved_key_pem,
            passphrase=(
                private_key_passphrase
                or os.environ.get(f"KALSHI_{env.upper()}_PRIVATE_KEY_PASSPHRASE")
                or os.environ.get("KALSHI_PRIVATE_KEY_PASSPHRASE")
            ),
        )

        historical_base = (
            historical_base_url
            or os.environ.get(f"KALSHI_{env.upper()}_HISTORICAL_API_BASE_URL")
            or os.environ.get("KALSHI_HISTORICAL_API_BASE_URL")
            or self.base_url
        )

        self.router = router or KalshiDataRouter(
            live_base_url=self.base_url,
            historical_base_url=historical_base,
            historical_cutoff_ts=(
                historical_cutoff_ts
                or os.environ.get("KALSHI_HISTORICAL_CUTOFF_TS")
            ),
        )

        self.retry_policy = retry_policy or RetryPolicy()
        self.limiter = limiter or RequestLimiter(rate_limit_policy or RateLimitPolicy())
        self.session = session or requests.Session()
        self.logger = logger or logging.getLogger(__name__)

    def available(self) -> bool:
        """Return whether the resource is available in the current runtime."""
        return bool(self.signer and self.signer.available())

    @staticmethod
    def _join_url(base_url: str, path: str) -> str:
        """Internal helper for join url."""
        p = str(path)
        if p.startswith("http://") or p.startswith("https://"):
            return p

        base = str(base_url).rstrip("/")
        clean = "/" + p.lstrip("/")

        # Prevent duplicated API prefix when caller already includes it.
        if base.endswith("/trade-api/v2") and clean.startswith("/trade-api/v2/"):
            clean = clean[len("/trade-api/v2"):]
        return f"{base}{clean}"

    def _auth_headers(self, method: str, signed_path: str) -> Dict[str, str]:
        """Internal helper for auth headers."""
        if not self.available():
            raise RuntimeError(
                "Kalshi signer unavailable. Set KALSHI_*_ACCESS_KEY and private key.",
            )
        ts_ms = str(int(time.time() * 1000))
        signature = self.signer.sign(ts_ms, method=method, path=signed_path)
        return {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "KALSHI-ACCESS-KEY": self.signer.access_key,
            "KALSHI-ACCESS-TIMESTAMP": ts_ms,
            "KALSHI-ACCESS-SIGNATURE": signature,
        }

    def _request_with_retries(
        self,
        method: str,
        url: str,
        signed_path: str,
        params: Optional[Dict[str, object]] = None,
        json_body: Optional[Dict[str, object]] = None,
    ) -> Dict[str, object]:
        """Internal helper for request with retries."""
        last_err: Optional[Exception] = None
        for attempt in range(self.retry_policy.max_retries + 1):
            try:
                self.limiter.acquire()
                resp = self.session.request(
                    method=method.upper(),
                    url=url,
                    headers=self._auth_headers(method=method, signed_path=signed_path),
                    params=params,
                    json=json_body,
                    timeout=self.retry_policy.timeout_seconds,
                )

                request_id = resp.headers.get("X-Request-Id") or resp.headers.get("x-request-id") or ""
                if request_id:
                    self.logger.debug("Kalshi request_id=%s method=%s path=%s", request_id, method.upper(), signed_path)

                if resp.status_code == 429:
                    retry_after = resp.headers.get("Retry-After")
                    if retry_after:
                        try:
                            time.sleep(max(float(retry_after), 0.25))
                        except (TypeError, ValueError):
                            pass
                    raise requests.HTTPError(
                        f"Transient status=429 path={signed_path}",
                        response=resp,
                    )
                if resp.status_code in (500, 502, 503, 504):
                    raise requests.HTTPError(
                        f"Transient status={resp.status_code} path={signed_path}",
                        response=resp,
                    )

                resp.raise_for_status()
                payload = resp.json()
                if not isinstance(payload, dict):
                    raise ValueError(f"Unexpected payload type: {type(payload).__name__}")
                return payload
            except (requests.RequestException, ValueError, RuntimeError, OSError) as exc:
                last_err = exc
                if attempt >= self.retry_policy.max_retries:
                    break
                sleep_for = self.retry_policy.backoff_seconds * (2 ** attempt)
                time.sleep(sleep_for)

        raise RuntimeError(f"Kalshi request failed after retries: {last_err}")

    def _request(
        self,
        method: str,
        path: str,
        params: Optional[Dict[str, object]] = None,
        json_body: Optional[Dict[str, object]] = None,
    ) -> Dict[str, object]:
        """Internal helper for request."""
        decision = self.router.resolve(path=path, params=params)
        primary_url = self._join_url(decision.base_url, decision.path)
        signed_path = urlparse(primary_url).path

        try:
            return self._request_with_retries(
                method=method,
                url=primary_url,
                signed_path=signed_path,
                params=params,
                json_body=json_body,
            )
        except (RuntimeError, requests.RequestException):
            # Fallback: if historical route fails, try live route once.
            if not decision.use_historical:
                raise
            fallback_url = self._join_url(self.base_url, path)
            fallback_signed_path = urlparse(fallback_url).path
            return self._request_with_retries(
                method=method,
                url=fallback_url,
                signed_path=fallback_signed_path,
                params=params,
                json_body=json_body,
            )

    def get(self, path: str, params: Optional[Dict[str, object]] = None) -> Dict[str, object]:
        """get."""
        return self._request("GET", path=path, params=params)

    def paginate(
        self,
        path: str,
        params: Optional[Dict[str, object]] = None,
        items_key: str = "markets",
        cursor_key: str = "cursor",
        cursor_param: str = "cursor",
    ) -> Iterable[Dict[str, object]]:
        """
        Iterate API pages until no next cursor is returned.
        """
        query = dict(params or {})
        cursor = None
        while True:
            if cursor:
                query[cursor_param] = cursor
            payload = self.get(path=path, params=query)
            items = payload.get(items_key, [])
            if isinstance(items, list):
                for row in items:
                    if isinstance(row, dict):
                        yield row
            cursor = payload.get(cursor_key, None)
            if not cursor:
                break

    def get_account_limits(self) -> Dict[str, object]:
        """Return account limits."""
        payload = self.get("/account/limits")
        self.limiter.update_from_account_limits(payload)
        return payload

    def fetch_historical_cutoff(self) -> Optional[str]:
        """fetch historical cutoff."""
        for ep in ("/historical/cutoff", "/partition/cutoff", "/cutoff"):
            try:
                payload = self.get(ep)
            except (RuntimeError, requests.RequestException):
                continue
            for key in ("cutoff_ts", "historical_cutoff_ts", "cutoff", "timestamp"):
                raw = payload.get(key)
                if raw:
                    self.router.update_cutoff(str(raw))
                    return str(raw)
        return None

    def server_time_utc(self) -> datetime:
        """
        Fetch server time when endpoint exists; otherwise return local UTC now.
        """
        endpoints = ["/time", "/server/time", "/exchange/time"]
        for ep in endpoints:
            try:
                payload = self.get(ep)
            except (RuntimeError, requests.RequestException):
                continue
            for key in ("server_time", "time", "timestamp"):
                raw = payload.get(key)
                if raw is None:
                    continue
                try:
                    return datetime.fromisoformat(str(raw).replace("Z", "+00:00")).astimezone(timezone.utc)
                except (ValueError, TypeError):
                    continue
        return datetime.now(timezone.utc)

    def clock_skew_seconds(self) -> float:
        """clock skew seconds."""
        server_ts = self.server_time_utc()
        local_ts = datetime.now(timezone.utc)
        return float((local_ts - server_ts).total_seconds())

    def list_markets(self, status: Optional[str] = None) -> List[Dict[str, object]]:
        """List markets."""
        params: Dict[str, object] = {}
        if status:
            params["status"] = status
        return list(
            self.paginate(
                path="/markets",
                params=params,
                items_key="markets",
                cursor_key="cursor",
            ),
        )

    def list_contracts(self, market_id: str) -> List[Dict[str, object]]:
        """List contracts."""
        return list(
            self.paginate(
                path=f"/markets/{market_id}/contracts",
                params=None,
                items_key="contracts",
                cursor_key="cursor",
            ),
        )

    def list_trades(
        self,
        contract_id: str,
        start_ts: Optional[str] = None,
        end_ts: Optional[str] = None,
    ) -> List[Dict[str, object]]:
        """List trades."""
        params: Dict[str, object] = {}
        if start_ts:
            params["start_ts"] = start_ts
        if end_ts:
            params["end_ts"] = end_ts
        return list(
            self.paginate(
                path=f"/contracts/{contract_id}/trades",
                params=params,
                items_key="trades",
                cursor_key="cursor",
            ),
        )

    def list_quotes(
        self,
        contract_id: str,
        start_ts: Optional[str] = None,
        end_ts: Optional[str] = None,
    ) -> List[Dict[str, object]]:
        """List quotes."""
        params: Dict[str, object] = {}
        if start_ts:
            params["start_ts"] = start_ts
        if end_ts:
            params["end_ts"] = end_ts
        return list(
            self.paginate(
                path=f"/contracts/{contract_id}/quotes",
                params=params,
                items_key="quotes",
                cursor_key="cursor",
            ),
        )
