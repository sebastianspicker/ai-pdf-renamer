"""Command-line launcher for the loopback Folionym browser application."""

from __future__ import annotations

import argparse
import socket
import threading
import time
import webbrowser
from pathlib import Path


def _open_browser_when_ready(url: str, host: str, port: int) -> None:
    """Wait briefly for Uvicorn to listen before opening the browser."""
    for _attempt in range(100):
        try:
            with socket.create_connection((host, port), timeout=0.1):
                webbrowser.open(url)
                return
        except OSError:
            time.sleep(0.05)


def main(argv: list[str] | None = None) -> None:
    """Launch the packaged local browser frontend."""
    parser = argparse.ArgumentParser(description="Launch Folionym's local browser frontend.")
    parser.add_argument("--port", type=int, default=8765, help="loopback port (default: 8765)")
    parser.add_argument("--no-open", action="store_true", help="do not open the default browser")
    args = parser.parse_args(argv)
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")

    try:
        import uvicorn
    except ImportError as exc:
        raise SystemExit("Install the browser frontend with: pip install 'folionym[web]'") from exc

    from .web_app import create_app

    static_dir = Path(__file__).with_name("web_dist")
    if not (static_dir / "index.html").is_file():
        raise SystemExit("Packaged browser assets are missing. Reinstall Folionym with the web extra.")

    host = "127.0.0.1"
    url = f"http://{host}:{args.port}/source"
    if not args.no_open:
        threading.Thread(
            target=_open_browser_when_ready,
            args=(url, host, args.port),
            name="folionym-browser-open",
            daemon=True,
        ).start()
    uvicorn.run(create_app(static_dir=static_dir), host=host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
