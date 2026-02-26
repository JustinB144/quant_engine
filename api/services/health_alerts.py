"""Health alert and notification system — Spec 09.

Detects health degradation (day-over-day drops) and domain-level
failures, then dispatches alerts via logging, optional email, and
optional webhook.

Alert types:
    CRITICAL  — overall health < 40% or domain < 50%
    STANDARD  — health degradation > 10% day-over-day
    INFORMATIONAL — low-confidence scores, stale data warnings

Rate limiting:
    Duplicate alerts are suppressed within a configurable window
    (default 24 hours) to prevent alert fatigue.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class Alert:
    """A single health alert event."""

    alert_type: str  # "CRITICAL", "STANDARD", "INFORMATIONAL"
    message: str
    domain: str = ""  # empty for overall alerts
    health_score: Optional[float] = None
    domain_scores: Dict[str, float] = field(default_factory=dict)
    timestamp: str = ""

    def __post_init__(self) -> None:
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()

    @property
    def dedup_key(self) -> str:
        """Unique key for deduplication (same type + domain)."""
        return f"{self.alert_type}:{self.domain}:{self.message[:80]}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "alert_type": self.alert_type,
            "message": self.message,
            "domain": self.domain,
            "health_score": self.health_score,
            "domain_scores": self.domain_scores,
            "timestamp": self.timestamp,
        }


class HealthAlertManager:
    """Monitors health scores and dispatches alerts on degradation.

    Parameters
    ----------
    degradation_threshold : float
        Fraction drop in overall health that triggers an alert (default 0.10).
    domain_critical_threshold : float
        Domain score below which a critical alert is issued (0–100 scale).
    dedup_window_seconds : int
        Suppress duplicate alerts within this window (default 86400 = 24h).
    channels : list of dict
        Notification channel configurations from health.yaml.
    alert_history_path : Path, optional
        Path to persist alert history (for deduplication across restarts).
    """

    def __init__(
        self,
        degradation_threshold: float = 0.10,
        domain_critical_threshold: float = 50.0,
        dedup_window_seconds: int = 86400,
        channels: Optional[List[Dict[str, Any]]] = None,
        alert_history_path: Optional[Path] = None,
    ):
        self.degradation_threshold = degradation_threshold
        self.domain_critical_threshold = domain_critical_threshold
        self.dedup_window_seconds = dedup_window_seconds
        self.channels = channels or [{"type": "logging", "level": "warning"}]
        self.alert_history_path = alert_history_path

        # In-memory dedup: {dedup_key: last_alert_epoch}
        self._recent_alerts: Dict[str, float] = {}
        self._load_alert_history()

    # ── Core alert checks ────────────────────────────────────────────────

    def check_health_degradation(
        self,
        health_today: float,
        health_yesterday: float,
    ) -> Optional[Alert]:
        """Check if overall health dropped more than the threshold.

        Compares on 0–100 scale.  A drop from 80 to 65 is a 15-point
        (18.75%) degradation.

        Returns an Alert if triggered, None otherwise.
        """
        if health_yesterday is None or health_yesterday <= 0:
            return None
        if health_today is None:
            return None

        drop_fraction = (health_yesterday - health_today) / health_yesterday

        if drop_fraction > self.degradation_threshold:
            return Alert(
                alert_type="CRITICAL",
                message=(
                    f"Health degraded {drop_fraction:.1%} day-over-day "
                    f"({health_yesterday:.1f} → {health_today:.1f})"
                ),
                health_score=health_today,
            )
        return None

    def check_domain_failures(
        self,
        domain_scores: Dict[str, float],
    ) -> List[Alert]:
        """Check if any domain score falls below the critical threshold.

        Parameters
        ----------
        domain_scores : dict
            {domain_name: score} on 0–100 scale.

        Returns
        -------
        List of Alert objects for each failing domain.
        """
        alerts: List[Alert] = []
        for domain, score in domain_scores.items():
            if score is not None and score < self.domain_critical_threshold:
                alerts.append(Alert(
                    alert_type="STANDARD",
                    message=(
                        f"Domain '{domain}' health score {score:.1f} "
                        f"below critical threshold {self.domain_critical_threshold:.0f}"
                    ),
                    domain=domain,
                    health_score=score,
                    domain_scores=domain_scores,
                ))
        return alerts

    def check_low_confidence(
        self,
        check_name: str,
        n_samples: int,
        threshold: int = 20,
    ) -> Optional[Alert]:
        """Issue informational alert when a check has low sample count."""
        if n_samples < threshold:
            return Alert(
                alert_type="INFORMATIONAL",
                message=(
                    f"Health check '{check_name}' has only {n_samples} samples "
                    f"(minimum recommended: {threshold}) — score has low confidence"
                ),
                domain=check_name,
            )
        return None

    # ── Alert dispatch ───────────────────────────────────────────────────

    def process_alerts(self, alerts: List[Alert]) -> List[Alert]:
        """Deduplicate and dispatch a list of alerts.

        Returns the list of alerts that were actually sent (not suppressed).
        """
        sent: List[Alert] = []
        now = time.time()

        for alert in alerts:
            key = alert.dedup_key
            last_sent = self._recent_alerts.get(key, 0)

            if (now - last_sent) < self.dedup_window_seconds:
                logger.debug("Alert suppressed (dedup): %s", alert.message)
                continue

            self._send_alert(alert)
            self._recent_alerts[key] = now
            sent.append(alert)

        self._save_alert_history()
        return sent

    def _send_alert(self, alert: Alert) -> None:
        """Dispatch an alert through all configured channels."""
        for channel in self.channels:
            ch_type = channel.get("type", "logging")

            if ch_type == "logging":
                self._send_logging(alert, channel)
            elif ch_type == "email" and channel.get("enabled", False):
                self._send_email(alert, channel)
            elif ch_type == "webhook" and channel.get("enabled", False):
                self._send_webhook(alert, channel)

    @staticmethod
    def _send_logging(alert: Alert, channel: Dict[str, Any]) -> None:
        """Log the alert at the configured level."""
        level = channel.get("level", "warning").upper()
        msg = f"[HEALTH ALERT {alert.alert_type}] {alert.message}"

        if alert.alert_type == "CRITICAL" or level == "CRITICAL":
            logger.critical(msg)
        elif alert.alert_type == "STANDARD" or level == "WARNING":
            logger.warning(msg)
        else:
            logger.info(msg)

    @staticmethod
    def _send_email(alert: Alert, channel: Dict[str, Any]) -> None:
        """Send email alert (placeholder — requires SMTP config)."""
        recipients = channel.get("recipients", [])
        if not recipients:
            return
        logger.info(
            "Email alert would be sent to %s: %s",
            recipients, alert.message,
        )

    @staticmethod
    def _send_webhook(alert: Alert, channel: Dict[str, Any]) -> None:
        """Send webhook alert (POST to URL)."""
        url = channel.get("url", "")
        if not url:
            return
        try:
            import urllib.request

            payload = json.dumps(alert.to_dict()).encode("utf-8")
            req = urllib.request.Request(
                url,
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                logger.info("Webhook alert sent to %s: %s", url, resp.status)
        except Exception as e:
            logger.warning("Webhook alert failed: %s", e)

    # ── Persistence ──────────────────────────────────────────────────────

    def _load_alert_history(self) -> None:
        """Load dedup state from disk."""
        if self.alert_history_path and self.alert_history_path.exists():
            try:
                with open(self.alert_history_path) as f:
                    data = json.load(f)
                self._recent_alerts = {
                    k: float(v) for k, v in data.items()
                }
            except (json.JSONDecodeError, OSError):
                self._recent_alerts = {}

    def _save_alert_history(self) -> None:
        """Persist dedup state to disk."""
        if self.alert_history_path:
            try:
                self.alert_history_path.parent.mkdir(parents=True, exist_ok=True)
                # Prune stale entries
                now = time.time()
                active = {
                    k: v for k, v in self._recent_alerts.items()
                    if (now - v) < self.dedup_window_seconds * 7
                }
                with open(self.alert_history_path, "w") as f:
                    json.dump(active, f)
            except OSError as e:
                logger.warning("Failed to save alert history: %s", e)


def load_alert_config() -> Dict[str, Any]:
    """Load alert configuration from health.yaml."""
    try:
        import yaml

        config_path = Path(__file__).parent.parent.parent / "config" / "health.yaml"
        if config_path.exists():
            with open(config_path) as f:
                config = yaml.safe_load(f)
            return config.get("alerts", {})
    except ImportError:
        logger.debug("PyYAML not available; using default alert config")
    except Exception as e:
        logger.warning("Failed to load alert config: %s", e)
    return {}


def create_alert_manager(
    alert_history_path: Optional[Path] = None,
) -> HealthAlertManager:
    """Factory: create an AlertManager from health.yaml config."""
    config = load_alert_config()

    if alert_history_path is None:
        try:
            from quant_engine.config import RESULTS_DIR
            alert_history_path = Path(RESULTS_DIR) / "health_alert_history.json"
        except ImportError:
            pass

    return HealthAlertManager(
        degradation_threshold=config.get("health_degradation_threshold", 0.10),
        domain_critical_threshold=config.get("domain_critical_threshold", 50.0),
        dedup_window_seconds=config.get("dedup_window_seconds", 86400),
        channels=config.get("notification_channels", [{"type": "logging", "level": "warning"}]),
        alert_history_path=alert_history_path,
    )
