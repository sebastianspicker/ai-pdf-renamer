"""Confirmation screen for consequential Textual filesystem actions."""

from __future__ import annotations

from typing import ClassVar

from textual import on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Grid
from textual.screen import ModalScreen
from textual.widgets import Button, Static


class ConfirmActionScreen(ModalScreen[bool]):
    """Confirm a consequential filesystem action with cancellation first."""

    BINDINGS: ClassVar[list[Binding | tuple[str, str] | tuple[str, str, str]]] = [Binding("escape", "cancel", "Cancel")]

    def __init__(self, title: str, detail: str, confirm_label: str) -> None:
        """Store action-specific copy for the reusable confirmation layout."""
        super().__init__()
        self._title = title
        self._detail = detail
        self._confirm_label = confirm_label

    def compose(self) -> ComposeResult:
        """Compose one focused confirmation dialog."""
        with Grid(id="confirm-dialog"):
            yield Static(self._title, id="confirm-title", markup=False)
            yield Static(self._detail, id="confirm-detail", markup=False)
            yield Button("Cancel", id="confirm-cancel")
            yield Button(self._confirm_label, id="confirm-action", variant="error")

    def on_mount(self) -> None:
        """Focus the non-destructive action when the dialog mounts."""
        self.query_one("#confirm-cancel", Button).focus()

    def action_cancel(self) -> None:
        """Dismiss the dialog without changing files."""
        self.dismiss(False)

    @on(Button.Pressed, "#confirm-cancel")
    def on_cancel(self) -> None:
        """Cancel the pending action from its button."""
        self.dismiss(False)

    @on(Button.Pressed, "#confirm-action")
    def on_confirm(self) -> None:
        """Return an affirmative result to the owning app."""
        self.dismiss(True)
