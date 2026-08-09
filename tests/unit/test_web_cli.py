"""Tests for the loopback browser launcher."""

from __future__ import annotations

import sys
from collections.abc import Callable
from contextlib import nullcontext
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

import folionym.web_app as web_app
import folionym.web_cli as web_cli


def test_browser_launcher_retries_then_opens_once(monkeypatch: pytest.MonkeyPatch) -> None:
    """The helper waits for the loopback listener without opening prematurely."""
    attempts: list[tuple[tuple[str, int], float]] = []
    sleeps: list[float] = []
    opened: list[str] = []

    def create_connection(address: tuple[str, int], *, timeout: float) -> Any:
        attempts.append((address, timeout))
        if len(attempts) == 1:
            raise OSError("not ready")
        return nullcontext()

    monkeypatch.setattr("folionym.web_cli.socket.create_connection", create_connection)
    monkeypatch.setattr("folionym.web_cli.time.sleep", sleeps.append)
    monkeypatch.setattr("folionym.web_cli.webbrowser.open", opened.append)

    web_cli._open_browser_when_ready("http://127.0.0.1:8765/source", "127.0.0.1", 8765)

    assert attempts == [(("127.0.0.1", 8765), 0.1), (("127.0.0.1", 8765), 0.1)]
    assert sleeps == [0.05]
    assert opened == ["http://127.0.0.1:8765/source"]


def test_browser_launcher_abandons_unavailable_listener(monkeypatch: pytest.MonkeyPatch) -> None:
    """A permanently unavailable listener is silent and never opens a browser."""
    sleeps: list[float] = []
    opened: list[str] = []

    def unavailable(_address: tuple[str, int], *, timeout: float) -> Any:
        assert timeout == 0.1
        raise OSError("not ready")

    monkeypatch.setattr("folionym.web_cli.socket.create_connection", unavailable)
    monkeypatch.setattr("folionym.web_cli.time.sleep", sleeps.append)
    monkeypatch.setattr("folionym.web_cli.webbrowser.open", opened.append)

    web_cli._open_browser_when_ready("http://127.0.0.1:8765/source", "127.0.0.1", 8765)

    assert sleeps == [0.05] * 100
    assert opened == []


@pytest.mark.parametrize("port", [0, 65536])
def test_main_rejects_ports_outside_the_tcp_range(port: int) -> None:
    """Invalid ports fail before optional dependencies or server startup are touched."""
    with pytest.raises(SystemExit) as exc:
        web_cli.main(["--port", str(port)])

    assert exc.value.code == 2


def test_main_explains_missing_uvicorn(monkeypatch: pytest.MonkeyPatch) -> None:
    """The optional browser dependency has an actionable installation failure."""
    monkeypatch.setitem(sys.modules, "uvicorn", None)

    with pytest.raises(SystemExit, match="folionym\\[web\\]"):
        web_cli.main(["--no-open"])


def test_main_requires_packaged_frontend_assets(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Launching with Uvicorn installed still refuses an incomplete package."""
    monkeypatch.setitem(sys.modules, "uvicorn", ModuleType("uvicorn"))
    monkeypatch.setattr(web_cli, "Path", lambda _value: tmp_path)

    with pytest.raises(SystemExit, match="assets are missing"):
        web_cli.main(["--no-open"])


def test_main_starts_loopback_server_and_browser_thread(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A normal launch serves the packaged app on loopback and schedules browser opening."""
    static_dir = tmp_path / "web_dist"
    static_dir.mkdir()
    (static_dir / "index.html").write_text("<!doctype html>", encoding="utf-8")
    started_threads: list[object] = []
    server_calls: list[dict[str, object]] = []
    created_static_dirs: list[Path] = []

    class BrowserThread:
        def __init__(self, *, target: Callable[..., None], args: tuple[object, ...], name: str, daemon: bool) -> None:
            self.target = target
            self.args = args
            self.name = name
            self.daemon = daemon

        def start(self) -> None:
            started_threads.append(self)

    def create_app(*, static_dir: Path) -> object:
        created_static_dirs.append(static_dir)
        return "app"

    uvicorn = ModuleType("uvicorn")

    def run(app: object, **kwargs: object) -> None:
        server_calls.append({"app": app, **kwargs})

    uvicorn.run = run  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "uvicorn", uvicorn)
    monkeypatch.setattr(web_app, "create_app", create_app)
    monkeypatch.setattr(web_cli, "Path", lambda _value: tmp_path / "web_cli.py")
    monkeypatch.setattr("folionym.web_cli.threading.Thread", BrowserThread)

    web_cli.main(["--port", "9123"])

    assert created_static_dirs == [static_dir]
    assert server_calls == [{"app": "app", "host": "127.0.0.1", "port": 9123, "log_level": "warning"}]
    assert len(started_threads) == 1
    thread = started_threads[0]
    assert isinstance(thread, BrowserThread)
    assert thread.args == ("http://127.0.0.1:9123/source", "127.0.0.1", 9123)
    assert thread.name == "folionym-browser-open"
    assert thread.daemon is True


def test_main_no_open_skips_browser_thread(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """--no-open leaves browser launch entirely out of the startup path."""
    static_dir = tmp_path / "web_dist"
    static_dir.mkdir()
    (static_dir / "index.html").write_text("<!doctype html>", encoding="utf-8")
    uvicorn = ModuleType("uvicorn")
    uvicorn.run = lambda *_args, **_kwargs: None  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "uvicorn", uvicorn)
    monkeypatch.setattr(web_app, "create_app", lambda *, static_dir: object())
    monkeypatch.setattr(web_cli, "Path", lambda _value: tmp_path / "web_cli.py")
    monkeypatch.setattr(
        "folionym.web_cli.threading.Thread", lambda **_kwargs: pytest.fail("--no-open must not start a browser thread")
    )

    web_cli.main(["--no-open"])
