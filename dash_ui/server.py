"""
Entry point for the Quant Engine Dash application.

Configures Flask caching and provides the WSGI server instance for
production deployment with Gunicorn or other WSGI servers.

Usage:
    # Development
    $ python server.py

    # Production with Gunicorn
    $ gunicorn -w 4 -b 0.0.0.0:8050 quant_engine.dash_ui.server:server
"""
import tempfile
from pathlib import Path

from flask_caching import Cache

from .app import create_app
from .data.cache import init_cache


# Create the Dash application
app = create_app()

# Extract Flask server instance for WSGI deployments (Gunicorn, etc.)
server = app.server

# Configure Flask caching: FileSystemCache in system temp directory
# Default timeout: 60 seconds for cached data functions
cache_dir = Path(tempfile.gettempdir()) / "qe_dash_cache"
cache_dir.mkdir(parents=True, exist_ok=True)

server.config["CACHE_TYPE"] = "filesystem"
server.config["CACHE_DIR"] = str(cache_dir)
server.config["CACHE_DEFAULT_TIMEOUT"] = 60
server.config["CACHE_THRESHOLD"] = 500

# Initialize cache with Flask app
init_cache(app)


if __name__ == "__main__":
    # Development server: debug=True for auto-reload and better error messages
    # In production, use Gunicorn: gunicorn -w 4 -b 0.0.0.0:8050 quant_engine.dash_ui.server:server
    app.run(
        debug=True,
        host="0.0.0.0",
        port=8050,
        dev_tools_hot_reload=True,
        dev_tools_ui=True,
    )
