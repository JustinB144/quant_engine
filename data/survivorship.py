"""
Survivorship Bias Controls (Tasks 112-117)
==========================================

Prevents survivorship bias by:
1. Tracking historical universe membership
2. Handling delisting events properly
3. Preserving dead company data
4. Reconstructing point-in-time indices

Without this, your backtest only trades "winners" that survived to today.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set
from datetime import datetime, date
from enum import Enum
import json
import sqlite3
import os


class DelistingReason(Enum):
    """Reason for stock delisting."""
    BANKRUPTCY = "bankruptcy"
    MERGER_CASH = "merger_cash"           # Acquired for cash
    MERGER_STOCK = "merger_stock"         # Acquired for stock
    GOING_PRIVATE = "going_private"
    EXCHANGE_DELISTING = "exchange_delisting"  # Failed listing requirements
    VOLUNTARY = "voluntary"
    SPINOFF = "spinoff"
    UNKNOWN = "unknown"


@dataclass
class UniverseMember:
    """
    Task 112: Track a symbol's membership in a universe.
    """
    symbol: str
    universe_name: str  # e.g., "SP500", "NASDAQ100"
    entry_date: date
    exit_date: Optional[date] = None  # None = still active
    entry_reason: str = "addition"
    exit_reason: Optional[str] = None
    metadata: Dict = field(default_factory=dict)

    def is_active_on(self, check_date: date) -> bool:
        """Check if symbol was in universe on given date."""
        if check_date < self.entry_date:
            return False
        if self.exit_date and check_date >= self.exit_date:
            return False
        return True

    def to_dict(self) -> Dict:
        """Serialize UniverseMember to a dictionary."""
        return {
            'symbol': self.symbol,
            'universe_name': self.universe_name,
            'entry_date': self.entry_date.isoformat(),
            'exit_date': self.exit_date.isoformat() if self.exit_date else None,
            'entry_reason': self.entry_reason,
            'exit_reason': self.exit_reason,
            'metadata': self.metadata,
        }


@dataclass
class UniverseChange:
    """
    Task 114: Track a change to universe membership.
    """
    date: date
    universe_name: str
    change_type: str  # "addition" or "removal"
    symbol: str
    reason: str
    replacing_symbol: Optional[str] = None  # For additions that replace another
    metadata: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """Serialize UniverseChange to a dictionary."""
        return {
            'date': self.date.isoformat(),
            'universe_name': self.universe_name,
            'change_type': self.change_type,
            'symbol': self.symbol,
            'reason': self.reason,
            'replacing_symbol': self.replacing_symbol,
            'metadata': self.metadata,
        }


@dataclass
class DelistingEvent:
    """
    Task 113: Track delisting event with proper returns.
    """
    symbol: str
    delisting_date: date
    reason: DelistingReason
    last_price: float
    delisting_return: float  # Return on delisting day (can be large negative)
    cash_distribution: float = 0.0  # For mergers/acquisitions
    acquirer_symbol: Optional[str] = None  # For stock mergers
    exchange_ratio: float = 0.0  # Shares of acquirer per share
    metadata: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """Serialize DelistingEvent to a dictionary."""
        return {
            'symbol': self.symbol,
            'delisting_date': self.delisting_date.isoformat(),
            'reason': self.reason.value,
            'last_price': self.last_price,
            'delisting_return': self.delisting_return,
            'cash_distribution': self.cash_distribution,
            'acquirer_symbol': self.acquirer_symbol,
            'exchange_ratio': self.exchange_ratio,
            'metadata': self.metadata,
        }


@dataclass
class SurvivorshipReport:
    """
    Task 117: Report comparing returns with/without survivorship adjustment.
    """
    period_start: date
    period_end: date
    universe_name: str

    # With survivorship bias (naive)
    naive_return: float
    naive_sharpe: float
    naive_symbols: int  # Only current survivors

    # Without survivorship bias (correct)
    adjusted_return: float
    adjusted_sharpe: float
    adjusted_symbols: int  # Including delisted

    # Bias metrics
    return_bias: float  # naive - adjusted
    sharpe_bias: float
    delisted_count: int
    delisting_impact: float  # Avg return of delisted stocks

    def to_dict(self) -> Dict:
        """Serialize SurvivorshipReport to a dictionary."""
        return {
            'period_start': self.period_start.isoformat(),
            'period_end': self.period_end.isoformat(),
            'universe_name': self.universe_name,
            'naive_return': self.naive_return,
            'naive_sharpe': self.naive_sharpe,
            'naive_symbols': self.naive_symbols,
            'adjusted_return': self.adjusted_return,
            'adjusted_sharpe': self.adjusted_sharpe,
            'adjusted_symbols': self.adjusted_symbols,
            'return_bias': self.return_bias,
            'sharpe_bias': self.sharpe_bias,
            'delisted_count': self.delisted_count,
            'delisting_impact': self.delisting_impact,
        }


class UniverseHistoryTracker:
    """
    Task 112, 114, 115: Track historical universe membership.

    Stores which symbols were in which universes at each point in time.
    """

    def __init__(self, db_path: str = "universe_history.db"):
        """Initialize UniverseHistoryTracker."""
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """Initialize database schema."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS universe_members (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    universe_name TEXT NOT NULL,
                    entry_date DATE NOT NULL,
                    exit_date DATE,
                    entry_reason TEXT,
                    exit_reason TEXT,
                    metadata TEXT,
                    UNIQUE(symbol, universe_name, entry_date)
                )
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS universe_changes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    change_date DATE NOT NULL,
                    universe_name TEXT NOT NULL,
                    change_type TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    reason TEXT,
                    replacing_symbol TEXT,
                    metadata TEXT
                )
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_members_universe_date
                ON universe_members(universe_name, entry_date, exit_date)
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_changes_date
                ON universe_changes(change_date, universe_name)
            """)

            conn.commit()

    def add_member(self, member: UniverseMember):
        """Add a universe member record."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO universe_members
                (symbol, universe_name, entry_date, exit_date,
                 entry_reason, exit_reason, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                member.symbol,
                member.universe_name,
                member.entry_date.isoformat(),
                member.exit_date.isoformat() if member.exit_date else None,
                member.entry_reason,
                member.exit_reason,
                json.dumps(member.metadata)
            ))
            conn.commit()

    def record_change(self, change: UniverseChange):
        """Task 114: Record a universe change."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO universe_changes
                (change_date, universe_name, change_type, symbol,
                 reason, replacing_symbol, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                change.date.isoformat(),
                change.universe_name,
                change.change_type,
                change.symbol,
                change.reason,
                change.replacing_symbol,
                json.dumps(change.metadata)
            ))
            conn.commit()

    def get_universe_on_date(
        self,
        universe_name: str,
        as_of_date: date
    ) -> List[str]:
        """
        Task 115: Reconstruct universe membership on a specific date.

        Returns list of symbols that were in the universe on that date.
        """
        with sqlite3.connect(self.db_path) as conn:
            results = conn.execute("""
                SELECT symbol FROM universe_members
                WHERE universe_name = ?
                  AND entry_date <= ?
                  AND (exit_date IS NULL OR exit_date > ?)
            """, (
                universe_name,
                as_of_date.isoformat(),
                as_of_date.isoformat()
            )).fetchall()

        return [r[0] for r in results]

    def get_changes_in_period(
        self,
        universe_name: str,
        start_date: date,
        end_date: date
    ) -> List[UniverseChange]:
        """Get all changes to a universe in a period."""
        with sqlite3.connect(self.db_path) as conn:
            results = conn.execute("""
                SELECT change_date, universe_name, change_type, symbol,
                       reason, replacing_symbol, metadata
                FROM universe_changes
                WHERE universe_name = ?
                  AND change_date >= ?
                  AND change_date <= ?
                ORDER BY change_date
            """, (
                universe_name,
                start_date.isoformat(),
                end_date.isoformat()
            )).fetchall()

        changes = []
        for r in results:
            changes.append(UniverseChange(
                date=date.fromisoformat(r[0]),
                universe_name=r[1],
                change_type=r[2],
                symbol=r[3],
                reason=r[4],
                replacing_symbol=r[5],
                metadata=json.loads(r[6]) if r[6] else {}
            ))

        return changes

    def bulk_load_universe(
        self,
        universe_name: str,
        symbols: List[str],
        as_of_date: date
    ):
        """Bulk load current universe members."""
        for symbol in symbols:
            member = UniverseMember(
                symbol=symbol,
                universe_name=universe_name,
                entry_date=as_of_date,
                entry_reason="initial_load"
            )
            self.add_member(member)

    def clear_universe(self, universe_name: str):
        """Remove stored membership records for one universe."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "DELETE FROM universe_members WHERE universe_name = ?",
                (universe_name,),
            )
            conn.execute(
                "DELETE FROM universe_changes WHERE universe_name = ?",
                (universe_name,),
            )
            conn.commit()


def hydrate_universe_history_from_snapshots(
    snapshots: pd.DataFrame,
    universe_name: str,
    db_path: Optional[str] = None,
    verbose: bool = False,
) -> int:
    """
    Build point-in-time universe intervals from snapshot rows.

    Expects `snapshots` with columns:
        - date (snapshot date)
        - permno (preferred member id) OR ticker
    """
    if snapshots is None or len(snapshots) == 0:
        return 0

    local_db = db_path
    if local_db is None:
        from ..config import SURVIVORSHIP_DB

        local_db = str(SURVIVORSHIP_DB)

    if "permno" in snapshots.columns and snapshots["permno"].notna().any():
        id_col = "permno"
    elif "ticker" in snapshots.columns:
        id_col = "ticker"
    else:
        return 0

    df = snapshots[["date", id_col]].copy()
    df["date"] = pd.to_datetime(df["date"]).dt.date
    if id_col == "permno":
        df[id_col] = pd.to_numeric(df[id_col], errors="coerce")
        df = df.dropna(subset=["date", id_col])
        df[id_col] = df[id_col].astype(int).astype(str)
    else:
        df[id_col] = df[id_col].astype(str).str.strip().str.upper()
        df = df.dropna(subset=["date", id_col])
        df = df[df[id_col] != ""]
    if len(df) == 0:
        return 0

    tracker = UniverseHistoryTracker(db_path=str(local_db))
    tracker.clear_universe(universe_name)

    by_date: Dict[date, Set[str]] = {}
    for d, grp in df.groupby("date"):
        by_date[d] = set(grp[id_col].tolist())

    snap_dates = sorted(by_date.keys())
    all_symbols = sorted(set(df[id_col].tolist()))
    inserted = 0

    for symbol in all_symbols:
        in_universe = False
        entry: Optional[date] = None

        for snap_date in snap_dates:
            present = symbol in by_date[snap_date]
            if present and not in_universe:
                in_universe = True
                entry = snap_date
            elif (not present) and in_universe and entry is not None:
                tracker.add_member(
                    UniverseMember(
                        symbol=symbol,
                        universe_name=universe_name,
                        entry_date=entry,
                        exit_date=snap_date,  # exit is exclusive
                        entry_reason="snapshot_inferred",
                        exit_reason="snapshot_inferred",
                    ),
                )
                inserted += 1
                in_universe = False
                entry = None

        if in_universe and entry is not None:
            tracker.add_member(
                UniverseMember(
                    symbol=symbol,
                    universe_name=universe_name,
                    entry_date=entry,
                    exit_date=None,
                    entry_reason="snapshot_inferred",
                ),
            )
            inserted += 1

    if verbose:
        print(
            f"  Hydrated {inserted} membership intervals "
            f"for universe={universe_name} ({len(all_symbols)} symbols)",
        )
    return inserted


def hydrate_sp500_history_from_wrds(
    start_date: str,
    end_date: Optional[str] = None,
    db_path: Optional[str] = None,
    freq: str = "quarterly",
    verbose: bool = False,
) -> int:
    """
    Pull historical S&P 500 snapshots from WRDS and hydrate local PIT DB.
    """
    from .wrds_provider import WRDSProvider

    provider = WRDSProvider()
    if not provider.available():
        return 0

    snapshots = provider.get_sp500_history(
        start_date=start_date,
        end_date=end_date,
        freq="quarterly" if str(freq).lower().startswith("q") else "annual",
    )
    if snapshots is None or len(snapshots) == 0:
        return 0

    return hydrate_universe_history_from_snapshots(
        snapshots=snapshots,
        universe_name="SP500",
        db_path=db_path,
        verbose=verbose,
    )


def filter_panel_by_point_in_time_universe(
    panel: pd.DataFrame,
    universe_name: str,
    db_path: Optional[str] = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Filter MultiIndex panel rows by point-in-time universe membership.

    Panel index must be (symbol_id, date), where symbol_id is typically PERMNO.
    """
    if panel is None or len(panel) == 0:
        return panel
    if not isinstance(panel.index, pd.MultiIndex) or panel.index.nlevels < 2:
        return panel

    local_db = db_path
    if local_db is None:
        from ..config import SURVIVORSHIP_DB

        local_db = str(SURVIVORSHIP_DB)

    if not os.path.exists(str(local_db)):
        return panel

    with sqlite3.connect(str(local_db)) as conn:
        rows = conn.execute(
            """
            SELECT symbol, entry_date, COALESCE(exit_date, '9999-12-31')
            FROM universe_members
            WHERE universe_name = ?
            """,
            (universe_name,),
        ).fetchall()

    if not rows:
        return panel

    intervals: Dict[str, List[Tuple[pd.Timestamp, pd.Timestamp]]] = {}
    for symbol, entry_dt, exit_dt in rows:
        symbol_u = str(symbol).strip().upper()
        e = pd.Timestamp(entry_dt)
        x = pd.Timestamp(exit_dt)
        intervals.setdefault(symbol_u, []).append((e, x))

    symbols = panel.index.get_level_values(0).astype(str).str.strip().str.upper().to_numpy()
    dates = pd.to_datetime(panel.index.get_level_values(1)).to_numpy()
    keep = np.zeros(len(panel), dtype=bool)

    by_symbol_pos: Dict[str, List[int]] = {}
    for i, t in enumerate(symbols):
        by_symbol_pos.setdefault(t, []).append(i)

    for symbol, positions in by_symbol_pos.items():
        spans = intervals.get(symbol, [])
        if not spans:
            continue
        d = pd.to_datetime(dates[np.array(positions, dtype=int)])
        active = np.zeros(len(positions), dtype=bool)
        for entry_dt, exit_dt in spans:
            active |= (d >= entry_dt) & (d < exit_dt)
        keep[np.array(positions, dtype=int)] = active

    filtered = panel.iloc[keep]
    if verbose:
        dropped = int((~keep).sum())
        print(
            f"  PIT universe filter ({universe_name}): "
            f"kept={len(filtered)} dropped={dropped}",
        )
    return filtered


class DelistingHandler:
    """
    Task 113, 116: Handle delisting events properly.

    Ensures:
    - Delisting returns are captured (often large negative)
    - Dead company data is retained
    - Symbols don't just "vanish"
    """

    def __init__(self, db_path: str = "delisting_history.db"):
        """Initialize DelistingHandler."""
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """Initialize database schema."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS delisting_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL UNIQUE,
                    delisting_date DATE NOT NULL,
                    reason TEXT NOT NULL,
                    last_price REAL NOT NULL,
                    delisting_return REAL,
                    cash_distribution REAL DEFAULT 0,
                    acquirer_symbol TEXT,
                    exchange_ratio REAL DEFAULT 0,
                    metadata TEXT
                )
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS dead_company_prices (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    date DATE NOT NULL,
                    open REAL,
                    high REAL,
                    low REAL,
                    close REAL,
                    volume REAL,
                    UNIQUE(symbol, date)
                )
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_dead_prices_symbol
                ON dead_company_prices(symbol, date)
            """)

            conn.commit()

    def record_delisting(self, event: DelistingEvent):
        """Task 113: Record a delisting event."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO delisting_events
                (symbol, delisting_date, reason, last_price, delisting_return,
                 cash_distribution, acquirer_symbol, exchange_ratio, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                event.symbol,
                event.delisting_date.isoformat(),
                event.reason.value,
                event.last_price,
                event.delisting_return,
                event.cash_distribution,
                event.acquirer_symbol,
                event.exchange_ratio,
                json.dumps(event.metadata)
            ))
            conn.commit()

    def preserve_price_history(
        self,
        symbol: str,
        prices: pd.DataFrame
    ):
        """
        Task 116: Preserve price history for dead company.

        Keeps full history even after delisting.
        """
        with sqlite3.connect(self.db_path) as conn:
            for idx, row in prices.iterrows():
                date_str = idx.strftime('%Y-%m-%d') if hasattr(idx, 'strftime') else str(idx)
                conn.execute("""
                    INSERT OR REPLACE INTO dead_company_prices
                    (symbol, date, open, high, low, close, volume)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    symbol,
                    date_str,
                    row.get('Open', row.get('open')),
                    row.get('High', row.get('high')),
                    row.get('Low', row.get('low')),
                    row.get('Close', row.get('close')),
                    row.get('Volume', row.get('volume'))
                ))
            conn.commit()

    def get_dead_company_prices(self, symbol: str) -> pd.DataFrame:
        """Task 116: Retrieve preserved price history for dead company."""
        with sqlite3.connect(self.db_path) as conn:
            results = conn.execute("""
                SELECT date, open, high, low, close, volume
                FROM dead_company_prices
                WHERE symbol = ?
                ORDER BY date
            """, (symbol,)).fetchall()

        if not results:
            return pd.DataFrame()

        df = pd.DataFrame(results, columns=['date', 'Open', 'High', 'Low', 'Close', 'Volume'])
        df.loc[:, 'date'] = pd.to_datetime(df['date'])
        df.index = pd.DatetimeIndex(df.pop('date'))
        return df

    def get_delisting_event(self, symbol: str) -> Optional[DelistingEvent]:
        """Get delisting event for a symbol."""
        with sqlite3.connect(self.db_path) as conn:
            result = conn.execute("""
                SELECT symbol, delisting_date, reason, last_price, delisting_return,
                       cash_distribution, acquirer_symbol, exchange_ratio, metadata
                FROM delisting_events
                WHERE symbol = ?
            """, (symbol,)).fetchone()

        if not result:
            return None

        return DelistingEvent(
            symbol=result[0],
            delisting_date=date.fromisoformat(result[1]),
            reason=DelistingReason(result[2]),
            last_price=result[3],
            delisting_return=result[4] or 0,
            cash_distribution=result[5] or 0,
            acquirer_symbol=result[6],
            exchange_ratio=result[7] or 0,
            metadata=json.loads(result[8]) if result[8] else {}
        )

    def get_delisting_return(self, symbol: str) -> float:
        """
        Task 113: Get delisting return for proper backtest accounting.

        This is crucial - bankrupt companies often have -100% returns.
        """
        event = self.get_delisting_event(symbol)
        if event:
            return event.delisting_return
        return 0.0

    def is_delisted(self, symbol: str, as_of_date: date) -> bool:
        """Check if symbol was delisted by a given date."""
        event = self.get_delisting_event(symbol)
        if event and event.delisting_date <= as_of_date:
            return True
        return False

    def get_all_delisted_symbols(self) -> List[str]:
        """Get list of all delisted symbols."""
        with sqlite3.connect(self.db_path) as conn:
            results = conn.execute("""
                SELECT symbol FROM delisting_events
            """).fetchall()
        return [r[0] for r in results]


class SurvivorshipBiasController:
    """
    Task 117: Main controller for survivorship bias analysis.

    Combines universe tracking and delisting handling to:
    - Detect survivorship bias in backtests
    - Correct for it
    - Report the impact
    """

    def __init__(
        self,
        universe_tracker: UniverseHistoryTracker = None,
        delisting_handler: DelistingHandler = None
    ):
        """Initialize SurvivorshipBiasController."""
        self.universe_tracker = universe_tracker or UniverseHistoryTracker()
        self.delisting_handler = delisting_handler or DelistingHandler()

    def get_survivorship_free_universe(
        self,
        universe_name: str,
        as_of_date: date
    ) -> List[str]:
        """
        Get universe membership as it would have been known on a date.

        This is the CORRECT universe for backtesting - includes
        companies that later delisted.
        """
        return self.universe_tracker.get_universe_on_date(
            universe_name, as_of_date
        )

    def calculate_bias_impact(
        self,
        universe_name: str,
        prices: Dict[str, pd.DataFrame],
        start_date: date,
        end_date: date,
        current_survivors_only: List[str] = None
    ) -> SurvivorshipReport:
        """
        Task 117: Calculate the impact of survivorship bias.

        Compares returns using:
        1. Only current survivors (biased)
        2. All historical members (unbiased)

        Args:
            universe_name: Name of universe
            prices: Dict of symbol -> price DataFrame
            start_date: Backtest start
            end_date: Backtest end
            current_survivors_only: List of symbols that exist today

        Returns:
            SurvivorshipReport with bias metrics
        """
        # Get historical universe
        historical_members = self.get_survivorship_free_universe(
            universe_name, start_date
        )

        if not historical_members:
            historical_members = list(prices.keys())

        if current_survivors_only is None:
            # Assume current survivors are those with price data through end_date
            current_survivors_only = [
                s for s, df in prices.items()
                if len(df) > 0 and df.index.max() >= pd.Timestamp(end_date) - pd.Timedelta(days=5)
            ]

        # Calculate naive returns (survivors only)
        naive_returns = []
        for symbol in current_survivors_only:
            if symbol in prices and len(prices[symbol]) > 0:
                df = prices[symbol]
                try:
                    start_price = df.loc[df.index >= pd.Timestamp(start_date), 'Close'].iloc[0]
                    end_price = df.loc[df.index <= pd.Timestamp(end_date), 'Close'].iloc[-1]
                    ret = (end_price - start_price) / start_price
                    naive_returns.append(ret)
                except (IndexError, KeyError):
                    pass

        # Calculate adjusted returns (all historical)
        adjusted_returns = []
        delisted_returns = []
        delisted_count = 0

        for symbol in historical_members:
            if symbol in prices and len(prices[symbol]) > 0:
                df = prices[symbol]
                try:
                    start_price = df.loc[df.index >= pd.Timestamp(start_date), 'Close'].iloc[0]

                    # Check if delisted
                    delisting = self.delisting_handler.get_delisting_event(symbol)
                    if delisting and start_date <= delisting.delisting_date <= end_date:
                        # Use delisting return
                        end_price = delisting.last_price
                        ret = (end_price - start_price) / start_price
                        ret += delisting.delisting_return  # Add delisting day return
                        adjusted_returns.append(ret)
                        delisted_returns.append(ret)
                        delisted_count += 1
                    else:
                        end_price = df.loc[df.index <= pd.Timestamp(end_date), 'Close'].iloc[-1]
                        ret = (end_price - start_price) / start_price
                        adjusted_returns.append(ret)
                except (IndexError, KeyError):
                    pass

        # Calculate metrics
        naive_return = np.mean(naive_returns) if naive_returns else 0
        naive_sharpe = (np.mean(naive_returns) / np.std(naive_returns)
                       if naive_returns and np.std(naive_returns) > 0 else 0)

        adjusted_return = np.mean(adjusted_returns) if adjusted_returns else 0
        adjusted_sharpe = (np.mean(adjusted_returns) / np.std(adjusted_returns)
                         if adjusted_returns and np.std(adjusted_returns) > 0 else 0)

        delisting_impact = np.mean(delisted_returns) if delisted_returns else 0

        return SurvivorshipReport(
            period_start=start_date,
            period_end=end_date,
            universe_name=universe_name,
            naive_return=naive_return,
            naive_sharpe=naive_sharpe,
            naive_symbols=len(current_survivors_only),
            adjusted_return=adjusted_return,
            adjusted_sharpe=adjusted_sharpe,
            adjusted_symbols=len(historical_members),
            return_bias=naive_return - adjusted_return,
            sharpe_bias=naive_sharpe - adjusted_sharpe,
            delisted_count=delisted_count,
            delisting_impact=delisting_impact,
        )

    def format_report(self, report: SurvivorshipReport) -> str:
        """Format survivorship bias report."""
        lines = [
            "=" * 60,
            "SURVIVORSHIP BIAS ANALYSIS",
            "=" * 60,
            "",
            f"Universe: {report.universe_name}",
            f"Period: {report.period_start} to {report.period_end}",
            "",
            "NAIVE (Current Survivors Only):",
            f"  • Symbols: {report.naive_symbols}",
            f"  • Return: {report.naive_return:.2%}",
            f"  • Sharpe: {report.naive_sharpe:.3f}",
            "",
            "ADJUSTED (Including Delisted):",
            f"  • Symbols: {report.adjusted_symbols}",
            f"  • Return: {report.adjusted_return:.2%}",
            f"  • Sharpe: {report.adjusted_sharpe:.3f}",
            "",
            "BIAS IMPACT:",
            f"  • Delisted companies: {report.delisted_count}",
            f"  • Avg delisting return: {report.delisting_impact:.2%}",
            f"  • Return bias: {report.return_bias:+.2%}",
            f"  • Sharpe bias: {report.sharpe_bias:+.3f}",
            "",
        ]

        if report.return_bias > 0.01:
            lines.append("  ⚠️ WARNING: Significant survivorship bias detected!")
            lines.append("     Your backtest may be overestimating returns.")

        lines.append("=" * 60)
        return "\n".join(lines)


# Convenience functions

def reconstruct_historical_universe(
    universe_name: str,
    as_of_date: date,
    tracker: UniverseHistoryTracker = None
) -> List[str]:
    """
    Task 115: Quick function to reconstruct historical universe.
    """
    tracker = tracker or UniverseHistoryTracker()
    return tracker.get_universe_on_date(universe_name, as_of_date)


def calculate_survivorship_bias_impact(
    prices: Dict[str, pd.DataFrame],
    start_date: date,
    end_date: date,
    universe_name: str = "backtest"
) -> SurvivorshipReport:
    """
    Task 117: Quick function to calculate survivorship bias impact.
    """
    controller = SurvivorshipBiasController()
    return controller.calculate_bias_impact(
        universe_name=universe_name,
        prices=prices,
        start_date=start_date,
        end_date=end_date
    )
