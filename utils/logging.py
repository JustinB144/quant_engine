"""
Structured logging for the quant engine.

Provides:
    - StructuredFormatter: JSON formatter for machine-parseable log output.
    - get_logger: Factory for structured loggers.
    - MetricsEmitter: Emit key metrics on every cycle and check alert thresholds.
    - AlertHistory: Persistent alert history with optional webhook notifications.
"""
from __future__ import annotations

import json
import logging
import sys
import urllib.request
import urllib.error
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


class StructuredFormatter(logging.Formatter):
    """JSON formatter for machine-parseable log output.

    Each log record is serialised as a single JSON line containing at minimum:
        timestamp, level, module, message.
    If the record carries a ``metrics`` attribute (set via ``extra={"metrics": {...}}``),
    those key-value pairs are included under the ``"metrics"`` key.
    """

    def format(self, record: logging.LogRecord) -> str:
        """format."""
        log_entry: Dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "module": record.module,
            "message": record.getMessage(),
        }
        if hasattr(record, "metrics"):
            log_entry["metrics"] = record.metrics
        if record.exc_info and record.exc_info[1] is not None:
            log_entry["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_entry, default=str)


def get_logger(name: str, level: str = "INFO") -> logging.Logger:
    """Get a structured logger for the quant engine.

    Parameters
    ----------
    name : str
        Logger name (typically ``__name__`` of the calling module).
    level : str
        Minimum log level.  One of DEBUG, INFO, WARNING, ERROR, CRITICAL.

    Returns
    -------
    logging.Logger
        Configured logger with a ``StructuredFormatter`` handler attached.
        If the logger already has handlers (e.g. from a previous call),
        no duplicate handler is added.
    """
    logger = logging.getLogger(name)
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(numeric_level)

    # Avoid adding duplicate handlers on repeated calls
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setLevel(numeric_level)
        handler.setFormatter(StructuredFormatter())
        logger.addHandler(handler)

    return logger


class AlertHistory:
    """Persistent alert history with optional webhook notifications.

    Stores alert events as JSON records with timestamps so operators can
    query when thresholds were breached over time.  Optionally sends
    notifications to a webhook URL (Slack, Discord, or any HTTP endpoint
    that accepts a JSON POST body with a ``"text"`` field).

    Parameters
    ----------
    history_file : str or Path, optional
        Path to the JSON file where alert history is persisted.  Defaults
        to the ``ALERT_HISTORY_FILE`` config value.
    webhook_url : str, optional
        HTTP(S) URL for webhook notifications.  Empty string or ``None``
        disables webhook delivery.  Defaults to ``ALERT_WEBHOOK_URL``
        from config.
    max_history : int, default 5000
        Maximum number of alert records to retain in the history file.
        Oldest records are pruned when this limit is exceeded.

    Usage::

        history = AlertHistory()
        history.record("IC below 2%: 0.0100", context={"ic": 0.01})
        recent = history.query(last_n=10)
        breaches = history.query(alert_type="IC below")
    """

    def __init__(
        self,
        history_file: Optional[str] = None,
        webhook_url: Optional[str] = None,
        max_history: int = 5000,
    ) -> None:
        # Lazy import to avoid circular dependency at module level
        """Initialize AlertHistory."""
        from ..config import ALERT_HISTORY_FILE, ALERT_WEBHOOK_URL

        if history_file is not None:
            self.history_file = Path(history_file)
        else:
            self.history_file = Path(ALERT_HISTORY_FILE)
        self.history_file.parent.mkdir(parents=True, exist_ok=True)

        self.webhook_url = webhook_url if webhook_url is not None else ALERT_WEBHOOK_URL
        self.max_history = max(100, int(max_history))
        self._logger = get_logger("quant_engine.alerts")

    def _load(self) -> List[Dict[str, Any]]:
        """Load alert history from disk."""
        if self.history_file.exists():
            try:
                with open(self.history_file, "r") as f:
                    data = json.load(f)
                if isinstance(data, list):
                    return data
            except (json.JSONDecodeError, OSError):
                pass
        return []

    def _save(self, records: List[Dict[str, Any]]) -> None:
        """Save alert history to disk, pruning to max_history."""
        records = records[-self.max_history:]
        try:
            with open(self.history_file, "w") as f:
                json.dump(records, f, indent=2, default=str)
        except OSError as e:
            self._logger.error(f"Failed to save alert history: {e}")

    def record(
        self,
        message: str,
        severity: str = "WARNING",
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Record an alert event and optionally send a webhook notification.

        Parameters
        ----------
        message : str
            Human-readable alert message.
        severity : str, default "WARNING"
            Alert severity level (INFO, WARNING, ERROR, CRITICAL).
        context : dict, optional
            Additional key-value context (metric values, thresholds, etc.).

        Returns
        -------
        dict
            The recorded alert event.
        """
        event: Dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "message": message,
            "severity": severity,
        }
        if context:
            event["context"] = context

        records = self._load()
        records.append(event)
        self._save(records)

        # Attempt webhook notification (fire-and-forget)
        self._notify_webhook(event)
        return event

    def record_batch(
        self,
        alerts: List[str],
        severity: str = "WARNING",
        context: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Record multiple alert events at once.

        Parameters
        ----------
        alerts : list of str
            Alert messages to record.
        severity : str, default "WARNING"
            Severity level applied to all alerts in this batch.
        context : dict, optional
            Shared context applied to all alerts.

        Returns
        -------
        list of dict
            The recorded alert events.
        """
        if not alerts:
            return []

        records = self._load()
        events: List[Dict[str, Any]] = []
        for msg in alerts:
            event: Dict[str, Any] = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "message": msg,
                "severity": severity,
            }
            if context:
                event["context"] = context
            records.append(event)
            events.append(event)
        self._save(records)

        # Send a single combined webhook for the batch
        if events:
            combined = {
                "timestamp": events[0]["timestamp"],
                "message": " | ".join(e["message"] for e in events),
                "severity": severity,
                "alert_count": len(events),
            }
            self._notify_webhook(combined)
        return events

    def query(
        self,
        last_n: Optional[int] = None,
        alert_type: Optional[str] = None,
        since: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Query alert history.

        Parameters
        ----------
        last_n : int, optional
            Return only the most recent *last_n* records.
        alert_type : str, optional
            Filter to records whose message contains this substring
            (case-insensitive).
        since : str, optional
            ISO-format timestamp.  Only return records at or after this
            time.

        Returns
        -------
        list of dict
            Matching alert records, ordered oldest to newest.
        """
        records = self._load()

        if since is not None:
            records = [
                r for r in records
                if r.get("timestamp", "") >= since
            ]

        if alert_type is not None:
            needle = alert_type.lower()
            records = [
                r for r in records
                if needle in r.get("message", "").lower()
            ]

        if last_n is not None and last_n > 0:
            records = records[-last_n:]

        return records

    def _notify_webhook(self, event: Dict[str, Any]) -> bool:
        """Send an alert event to the configured webhook URL.

        Uses urllib (no extra dependencies).  Failures are logged but
        never raised -- webhook delivery is best-effort.

        Parameters
        ----------
        event : dict
            Alert event payload.

        Returns
        -------
        bool
            ``True`` if the notification was sent successfully, ``False``
            otherwise (including when webhooks are disabled).
        """
        if not self.webhook_url:
            return False

        payload = json.dumps({
            "text": (
                f"[{event.get('severity', 'ALERT')}] "
                f"{event.get('message', 'Unknown alert')}"
            ),
            "timestamp": event.get("timestamp", ""),
            "details": event,
        }).encode("utf-8")

        req = urllib.request.Request(
            self.webhook_url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                return 200 <= resp.status < 300
        except (urllib.error.URLError, OSError, ValueError) as e:
            self._logger.warning(f"Webhook notification failed: {e}")
            return False


class MetricsEmitter:
    """Emit key metrics on every cycle and check alert thresholds.

    Usage::

        emitter = MetricsEmitter()
        emitter.emit_cycle_metrics(
            model_age_days=5,
            ic=0.04,
            rolling_sharpe=1.2,
            regime_distribution={0: 40, 1: 10, 2: 30, 3: 20},
            turnover=0.15,
            execution_costs=0.0012,
        )
        alerts = emitter.check_alerts(ic=0.01, sharpe=-0.3, drawdown=-0.18, regime_2_duration=35)
    """

    def __init__(
        self,
        logger_name: str = "quant_engine.metrics",
        alert_history: Optional[AlertHistory] = None,
    ) -> None:
        """Initialize MetricsEmitter."""
        self.logger = get_logger(logger_name)
        self.alert_history = alert_history or AlertHistory()

    def emit_cycle_metrics(
        self,
        model_age_days: int,
        ic: float,
        rolling_sharpe: float,
        regime_distribution: Dict[int, int],
        turnover: float,
        execution_costs: float,
    ) -> None:
        """Log a structured metrics payload for the current cycle.

        Parameters
        ----------
        model_age_days : int
            Number of days since the current model was trained.
        ic : float
            Information coefficient (rank correlation of predictions vs actuals).
        rolling_sharpe : float
            Rolling annualised Sharpe ratio.
        regime_distribution : dict
            Mapping regime_code -> count of observations.
        turnover : float
            Portfolio turnover as a fraction of NAV.
        execution_costs : float
            Total execution costs (spread + impact) as a fraction of NAV.
        """
        metrics: Dict[str, Any] = {
            "model_age_days": model_age_days,
            "ic": round(ic, 6),
            "rolling_sharpe": round(rolling_sharpe, 4),
            "regime_distribution": {str(k): v for k, v in regime_distribution.items()},
            "turnover": round(turnover, 6),
            "execution_costs": round(execution_costs, 6),
        }
        self.logger.info(
            "cycle_metrics",
            extra={"metrics": metrics},
        )

    def check_alerts(
        self,
        ic: float,
        sharpe: float,
        drawdown: float,
        regime_2_duration: int,
    ) -> List[str]:
        """Check alert thresholds, log, persist to history, and notify.

        Parameters
        ----------
        ic : float
            Current information coefficient.
        sharpe : float
            Current rolling Sharpe ratio.
        drawdown : float
            Current drawdown as a negative fraction (e.g. -0.15 = 15% drawdown).
        regime_2_duration : int
            Consecutive days spent in regime 2 (mean-reverting).

        Returns
        -------
        list of str
            Human-readable alert messages for each threshold breach.
        """
        alerts: List[str] = []
        if ic < 0.02:
            alerts.append(f"IC below 2%: {ic:.4f}")
        if sharpe < 0:
            alerts.append(f"Negative Sharpe: {sharpe:.4f}")
        if drawdown < -0.15:
            alerts.append(f"Drawdown exceeding 15%: {drawdown:.2%}")
        if regime_2_duration > 30:
            alerts.append(f"Regime 2 duration {regime_2_duration} days")

        # Log triggered alerts
        for alert in alerts:
            self.logger.warning(
                alert,
                extra={"metrics": {"alert": alert}},
            )

        # Persist to alert history and send webhook notification
        if alerts:
            context = {
                "ic": round(ic, 6),
                "sharpe": round(sharpe, 4),
                "drawdown": round(drawdown, 4),
                "regime_2_duration": regime_2_duration,
            }
            self.alert_history.record_batch(alerts, context=context)

        return alerts
