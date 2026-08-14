"""Form readers and run-state value access shared by the Textual app controller."""

from __future__ import annotations

import logging
from typing import Any, cast

from textual.css.query import QueryError
from textual.widgets import Checkbox, Input, Select

from .config import RenamerConfig
from .tui_presentation import endpoint_disclosure, snapshot_from_readers
from .tui_state import build_config_from_snapshot

logger = logging.getLogger(__name__)


class TuiValueAccess:
    """Keep configuration reads and operation-state values out of the view controller."""

    def get_str(self, widget_id: str, default: str = "") -> str:
        """Read and trim an input value, returning the supplied default on lookup failure."""
        try:
            widget = cast(Any, self).query_one(f"#{widget_id}", Input)
            return str(widget.value).strip()
        except QueryError:
            logger.debug("Widget query failed for #%s (Input)", widget_id)
            return default

    def get_bool(self, widget_id: str, default: bool = False) -> bool:
        """Read a checkbox value, returning the supplied default on lookup failure."""
        try:
            widget = cast(Any, self).query_one(f"#{widget_id}", Checkbox)
            return bool(widget.value)
        except QueryError:
            logger.debug("Widget query failed for #%s (Checkbox)", widget_id)
            return default

    def get_select(self, widget_id: str, default: str = "") -> str:
        """Read a select value, returning the supplied default for blank or missing widgets."""
        try:
            value = cast(Any, self).query_one(f"#{widget_id}", Select).value
            return str(value) if value is not Select.BLANK else default
        except QueryError:
            logger.debug("Widget query failed for #%s (Select)", widget_id)
            return default

    def snapshot(self) -> dict[str, object]:
        """Collect the current form values in the persisted settings shape."""
        return snapshot_from_readers(self.get_str, self.get_bool, self.get_select)

    @property
    def run_active(self) -> bool:
        """Return whether a preview or apply worker is active."""
        return bool(cast(Any, self)._operation_running)

    @run_active.setter
    def run_active(self, value: bool) -> None:
        """Set the run-active flag used by action guards."""
        cast(Any, self)._operation_running = bool(value)

    @property
    def stop_requested(self) -> bool:
        """Return whether cooperative cancellation has been requested."""
        return bool(cast(Any, self)._stop_event.is_set())

    def clear_stop_request(self) -> None:
        """Clear the cooperative cancellation signal before a run."""
        cast(Any, self)._stop_event.clear()

    def build_config(self, *, dry_run: bool, manual_mode: bool = False) -> RenamerConfig:
        """Build a rename configuration from the current form snapshot."""
        app = cast(Any, self)
        return build_config_from_snapshot(self.snapshot(), app._stop_event, dry_run=dry_run, manual_mode=manual_mode)

    def _endpoint_disclosure(self) -> tuple[str, str]:
        """Describe whether the configured model endpoint is local or external."""
        return endpoint_disclosure(self.get_bool("use_llm", True), self.get_str("llm_url"))
