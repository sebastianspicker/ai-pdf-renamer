"""Single-PDF selection and setup-error handling for the Textual controller."""

from __future__ import annotations

import contextlib
from pathlib import Path
from typing import Any, cast

from rich.markup import escape as escape_markup
from textual.css.query import QueryError
from textual.widgets import Input, RichLog, Static

from .assets import ERROR_COLOR


class TuiSourceSelection:
    """Keep source validation and its UI feedback out of the main controller."""

    def _show_setup_error(self, widget_id: str, message: str) -> None:
        """Show a source error on the setup tab and focus its input."""
        app = cast(Any, self)
        app._activate_tab("basic")
        error = app.query_one("#setup-error", Static)
        error.update(message)
        error.add_class("visible")
        with contextlib.suppress(QueryError):
            app.query_one(f"#{widget_id}", Input).focus()

    def _selected_single_pdf(self) -> Path | None:
        """Return the selected regular PDF after reporting validation failures."""
        app = cast(Any, self)
        log = app.query_one("#run-log", RichLog)
        file_path = app.get_str("single_file")
        if not file_path:
            message = "Set a single PDF path on the Setup tab first."
            app._show_setup_error("single_file", message)
            log.write(f"[bold {ERROR_COLOR}]No file set.[/bold {ERROR_COLOR}] {message}")
            app.notify(message, title="No file set", severity="error")
            return None
        selected = Path(file_path)
        if selected.is_symlink():
            app._show_setup_error("single_file", "Symbolic links are unsupported; choose the PDF itself.")
            log.write(
                f"[bold {ERROR_COLOR}]Unsupported file:[/bold {ERROR_COLOR}] "
                f"{escape_markup(file_path)} is a symbolic link"
            )
            app.notify(file_path, title="Symbolic links are unsupported", severity="error")
            return None
        if not selected.exists():
            app._show_setup_error("single_file", f"File not found: {file_path}")
            log.write(f"[bold {ERROR_COLOR}]File not found:[/bold {ERROR_COLOR}] {escape_markup(file_path)}")
            app.notify(file_path, title="File not found", severity="error")
            return None
        if selected.suffix.lower() != ".pdf":
            app._show_setup_error("single_file", "Choose a file with a .pdf extension.")
            log.write(
                f"[bold {ERROR_COLOR}]Not a PDF file:[/bold {ERROR_COLOR}] "
                f"{escape_markup(file_path)} (expected .pdf extension)"
            )
            app.notify(file_path, title="Not a PDF file", severity="error")
            return None
        return selected
