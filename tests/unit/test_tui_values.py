"""Unit tests for Textual form and run-state value access."""

from __future__ import annotations

import threading
from types import SimpleNamespace
from typing import Any

from textual.widgets import Select

from folionym.tui_values import TuiValueAccess


class _ValueHarness(TuiValueAccess):
    """Provide predictable widget values without mounting a Textual app."""

    def __init__(self) -> None:
        self._operation_running = False
        self._stop_event = threading.Event()
        self.values: dict[str, object] = {}

    def query_one(self, selector: str, _widget_type: object) -> Any:
        """Return the configured stand-in for one widget selector."""
        return SimpleNamespace(value=self.values[selector])


def test_value_access_normalizes_widgets_and_run_state() -> None:
    """Readers normalize values while state helpers preserve cancellation semantics."""
    harness = _ValueHarness()
    harness.values = {"#name": "  report  ", "#enabled": 1, "#choice": Select.BLANK}

    assert harness.get_str("name") == "report"
    assert harness.get_bool("enabled") is True
    assert harness.get_select("choice", "fallback") == "fallback"

    harness.run_active = True
    harness._stop_event.set()
    assert harness.run_active is True
    assert harness.stop_requested is True
    harness.clear_stop_request()
    assert harness.stop_requested is False
