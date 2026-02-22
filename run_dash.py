#!/usr/bin/env python3
"""
Quant Engine -- Dash Dashboard Launcher.

Launch the professional web-based trading system dashboard.

Usage:
    python run_dash.py [--port PORT] [--host HOST] [--no-debug]
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

REQUIRED_PACKAGES = [
    ("dash", "dash"),
    ("dash_bootstrap_components", "dash-bootstrap-components"),
    ("plotly", "plotly"),
    ("flask_caching", "flask-caching"),
]


def _check_dependencies() -> list:
    """Return list of missing package install names."""
    missing = []
    for module_name, pip_name in REQUIRED_PACKAGES:
        try:
            __import__(module_name)
        except ImportError:
            missing.append(pip_name)
    return missing


def main():
    """Launch the Quant Engine Dash Dashboard."""
    parser = argparse.ArgumentParser(description="Quant Engine Dashboard")
    parser.add_argument("--port", type=int, default=8050, help="Server port (default: 8050)")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Server host (default: 127.0.0.1)")
    parser.add_argument("--no-debug", action="store_true", help="Disable debug mode")
    args = parser.parse_args()

    print()
    print("  ╔══════════════════════════════════════════════════════╗")
    print("  ║        QUANT ENGINE -- Professional Dashboard       ║")
    print("  ║              Dash / Plotly Web Interface             ║")
    print("  ╚══════════════════════════════════════════════════════╝")
    print()

    # Check dependencies
    missing = _check_dependencies()
    if missing:
        print(f"  [ERROR] Missing packages: {', '.join(missing)}")
        print(f"  Install with: pip install {' '.join(missing)}")
        sys.exit(1)

    print("  [OK] All dependencies satisfied")

    try:
        from quant_engine.dash_ui.app import create_app
    except ImportError as e:
        print(f"\n  [ERROR] Import error: {e}")
        print("  Ensure quant_engine package is on PYTHONPATH.")
        sys.exit(1)

    app = create_app()

    debug = not args.no_debug
    mode = "DEBUG" if debug else "PRODUCTION"
    print(f"  [OK] Application created ({mode} mode)")
    print(f"  [OK] Dashboard running at: http://{args.host}:{args.port}")
    print()
    print("  Press Ctrl+C to stop.")
    print()

    app.run(debug=debug, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
