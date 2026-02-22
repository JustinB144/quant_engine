"""
ML Retraining Trigger Logic
=============================
Monitors model performance and determines when retraining is needed.

Adapted from automated_portfolio_system/utils/retrain_trigger.py for
the quant_engine regression pipeline (Spearman correlation, not AUC).

Triggers for retraining:
1. SCHEDULE: N calendar days since last training (default: 30)
2. TRADE COUNT: N new trades since last training (default: 50)
3. PERFORMANCE DEGRADATION: Rolling win rate drops below threshold
4. MODEL QUALITY: OOS Spearman correlation was below threshold
5. IC DRIFT: Rolling information coefficient degradation
6. SHARPE DEGRADATION: Rolling Sharpe drops below minimum

Usage:
    trigger = RetrainTrigger()
    should_retrain, reasons = trigger.check()
    if should_retrain:
        print(f"Retraining needed: {reasons}")
        trigger.record_retraining(n_trades=5000, oos_spearman=0.12)
"""
import json
import os
from datetime import datetime, timedelta
from typing import Tuple, List, Dict, Optional

import numpy as np

from ..config import MODEL_DIR


_DEFAULT_METADATA_PATH = str(MODEL_DIR / "retrain_metadata.json")


class RetrainTrigger:
    """Determines when ML model should be retrained."""

    def __init__(
        self,
        metadata_file: str = _DEFAULT_METADATA_PATH,
        max_days_between_retrain: int = 30,
        min_new_trades_for_retrain: int = 50,
        min_acceptable_win_rate: float = 0.45,
        min_acceptable_spearman: float = 0.05,
        min_acceptable_sharpe: float = 0.3,
        min_acceptable_ic: float = 0.02,
        lookback_trades: int = 30,
    ):
        """
        Args:
            metadata_file: Path to retrain metadata JSON
            max_days_between_retrain: Force retrain after this many days
            min_new_trades_for_retrain: Retrain if this many new trades accumulated
            min_acceptable_win_rate: Retrain if rolling win rate drops below this
            min_acceptable_spearman: Retrain if OOS Spearman was below this
            min_acceptable_sharpe: Retrain if rolling Sharpe drops below this
            min_acceptable_ic: Retrain if rolling IC drops below this
            lookback_trades: Number of recent trades for rolling performance
        """
        self.metadata_file = metadata_file
        self.max_days = max_days_between_retrain
        self.min_new_trades = min_new_trades_for_retrain
        self.min_win_rate = min_acceptable_win_rate
        self.min_spearman = min_acceptable_spearman
        self.min_sharpe = min_acceptable_sharpe
        self.min_ic = min_acceptable_ic
        self.lookback = lookback_trades

        self.metadata = self._load_metadata()

    def _load_metadata(self) -> Dict:
        """Load model metadata."""
        if os.path.exists(self.metadata_file):
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        return {
            'last_trained': None,
            'training_trades': 0,
            'oos_spearman': 0.0,
            'retrain_history': [],
            'recent_trade_results': [],
            'recent_ic_values': [],
        }

    def _save_metadata(self):
        """Save model metadata."""
        os.makedirs(os.path.dirname(self.metadata_file) or '.', exist_ok=True)
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2, default=str)

    def add_trade_result(
        self,
        is_winner: bool,
        net_return: float = 0.0,
        predicted_return: float = 0.0,
        actual_return: float = 0.0,
    ):
        """
        Record a completed trade result for performance monitoring.

        Args:
            is_winner: Whether the trade was profitable
            net_return: Net return of the trade (after costs)
            predicted_return: Model's predicted return
            actual_return: Actual realized return
        """
        self.metadata.setdefault('recent_trade_results', [])
        self.metadata['recent_trade_results'].append({
            'timestamp': datetime.now().isoformat(),
            'is_winner': is_winner,
            'net_return': net_return,
            'predicted_return': predicted_return,
            'actual_return': actual_return,
        })

        # Track IC (predicted vs actual correlation proxy)
        self.metadata.setdefault('recent_ic_values', [])
        if predicted_return != 0 and actual_return != 0:
            # Sign agreement is a rough IC proxy for individual trades
            sign_agree = 1.0 if (predicted_return > 0) == (actual_return > 0) else -1.0
            self.metadata['recent_ic_values'].append(sign_agree)

        # Keep only recent history
        max_history = self.lookback * 3
        if len(self.metadata['recent_trade_results']) > max_history:
            self.metadata['recent_trade_results'] = \
                self.metadata['recent_trade_results'][-max_history:]
        if len(self.metadata.get('recent_ic_values', [])) > max_history:
            self.metadata['recent_ic_values'] = \
                self.metadata['recent_ic_values'][-max_history:]

        self._save_metadata()

    def check(self) -> Tuple[bool, List[str]]:
        """
        Check all retraining triggers.

        Returns:
            Tuple of (should_retrain: bool, reasons: List[str])
        """
        reasons = []

        # 1. SCHEDULE CHECK
        last_trained = self.metadata.get('last_trained')
        if last_trained is None:
            reasons.append("Model has never been trained.")
        else:
            last_trained_dt = datetime.fromisoformat(last_trained)
            days_since = (datetime.now() - last_trained_dt).days
            if days_since > self.max_days:
                reasons.append(
                    f"Schedule: {days_since} days since last training "
                    f"(threshold: {self.max_days} days)."
                )

        # 2. TRADE COUNT CHECK
        recent_results = self.metadata.get('recent_trade_results', [])
        trades_since = 0
        if last_trained:
            for result in recent_results:
                if result['timestamp'] > last_trained:
                    trades_since += 1
        else:
            trades_since = len(recent_results)

        if trades_since >= self.min_new_trades:
            reasons.append(
                f"Trade count: {trades_since} new trades since last training "
                f"(threshold: {self.min_new_trades})."
            )

        # 3. PERFORMANCE DEGRADATION CHECK (rolling win rate)
        if len(recent_results) >= self.lookback:
            recent = recent_results[-self.lookback:]
            wins = sum(1 for r in recent if r['is_winner'])
            rolling_win_rate = wins / len(recent)

            if rolling_win_rate < self.min_win_rate:
                reasons.append(
                    f"Performance: Rolling win rate {rolling_win_rate:.1%} "
                    f"is below threshold {self.min_win_rate:.1%} "
                    f"(last {self.lookback} trades)."
                )

        # 4. MODEL QUALITY CHECK (from last training)
        oos_spearman = self.metadata.get('oos_spearman', 0.0)
        if oos_spearman > 0 and oos_spearman < self.min_spearman:
            reasons.append(
                f"Model quality: Last OOS Spearman was {oos_spearman:.3f} "
                f"(minimum: {self.min_spearman:.3f})."
            )

        # 5. IC DRIFT CHECK
        ic_values = self.metadata.get('recent_ic_values', [])
        if len(ic_values) >= self.lookback:
            rolling_ic = np.mean(ic_values[-self.lookback:])
            if rolling_ic < self.min_ic:
                reasons.append(
                    f"IC drift: Rolling IC {rolling_ic:.3f} "
                    f"below threshold {self.min_ic:.3f} "
                    f"(last {self.lookback} trades)."
                )

        # 6. SHARPE DEGRADATION CHECK
        if len(recent_results) >= self.lookback:
            recent = recent_results[-self.lookback:]
            returns = [r.get('net_return', 0) for r in recent]
            if len(returns) > 1:
                ret_arr = np.array(returns)
                mean_ret = np.mean(ret_arr)
                std_ret = np.std(ret_arr, ddof=1)
                if std_ret > 0:
                    # Annualize assuming ~252 trades/year
                    rolling_sharpe = (mean_ret / std_ret) * np.sqrt(252 / max(1, self.lookback))
                    if rolling_sharpe < self.min_sharpe:
                        reasons.append(
                            f"Sharpe degradation: Rolling Sharpe {rolling_sharpe:.2f} "
                            f"below threshold {self.min_sharpe:.2f}."
                        )

        should_retrain = len(reasons) > 0
        return should_retrain, reasons

    def record_retraining(
        self,
        n_trades: int = 0,
        oos_spearman: float = 0.0,
        version_id: str = "",
        notes: str = "",
    ):
        """
        Record that retraining has been completed.

        Args:
            n_trades: Number of trades in training set
            oos_spearman: Out-of-sample Spearman correlation
            version_id: Model version ID
            notes: Any notes about the retraining
        """
        now = datetime.now().isoformat()
        self.metadata['last_trained'] = now
        self.metadata['training_trades'] = n_trades
        self.metadata['oos_spearman'] = oos_spearman

        self.metadata.setdefault('retrain_history', [])
        self.metadata['retrain_history'].append({
            'timestamp': now,
            'n_trades': n_trades,
            'oos_spearman': oos_spearman,
            'version_id': version_id,
            'notes': notes,
        })

        self._save_metadata()

    def status(self) -> str:
        """Human-readable status summary."""
        lines = ["ML Model Retraining Status:"]

        last_trained = self.metadata.get('last_trained')
        if last_trained:
            dt = datetime.fromisoformat(last_trained)
            days_ago = (datetime.now() - dt).days
            lines.append(f"  Last trained: {last_trained[:10]} ({days_ago} days ago)")
        else:
            lines.append("  Last trained: NEVER")

        lines.append(f"  Training trades: {self.metadata.get('training_trades', 0)}")
        lines.append(f"  OOS Spearman: {self.metadata.get('oos_spearman', 0):.3f}")

        recent = self.metadata.get('recent_trade_results', [])
        if recent:
            wins = sum(1 for r in recent[-self.lookback:] if r['is_winner'])
            total = min(len(recent), self.lookback)
            lines.append(f"  Rolling win rate ({total} trades): {wins/total:.1%}")

        ic_values = self.metadata.get('recent_ic_values', [])
        if ic_values:
            n = min(len(ic_values), self.lookback)
            rolling_ic = np.mean(ic_values[-n:])
            lines.append(f"  Rolling IC ({n} trades): {rolling_ic:.3f}")

        history = self.metadata.get('retrain_history', [])
        lines.append(f"  Total retrainings: {len(history)}")

        should, reasons = self.check()
        if should:
            lines.append(f"  RETRAIN NEEDED: {len(reasons)} trigger(s) active")
            for r in reasons:
                lines.append(f"    - {r}")
        else:
            lines.append("  Status: OK (no retraining needed)")

        return '\n'.join(lines)
