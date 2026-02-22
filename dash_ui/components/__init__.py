"""
Reusable Dash components for the Quant Engine Dashboard.

This module provides a collection of professional, production-quality
components for building pages and visualizations:

Components:
  - metric_card: KPI metric card with conditional coloring
  - trade_table: Styled DataTable for trade-level data
  - chart_utils: Plotly chart factory functions (line, area, bar, heatmap, etc.)
  - regime_badge: Market regime indicator badge
  - alert_banner: Styled alert messages
  - sidebar: Navigation sidebar with active state highlighting
  - page_header: Consistent page header with title, subtitle, and actions

All components follow the Bloomberg dark theme and use consistent
typography, spacing, and color palettes.
"""

from .metric_card import metric_card
from .trade_table import trade_table
from .regime_badge import regime_badge
from .alert_banner import alert_banner
from .sidebar import create_sidebar, NAV_ITEMS
from .page_header import create_page_header
from . import chart_utils

__all__ = [
    "metric_card",
    "trade_table",
    "regime_badge",
    "alert_banner",
    "create_sidebar",
    "NAV_ITEMS",
    "chart_utils",
    "create_page_header",
]
