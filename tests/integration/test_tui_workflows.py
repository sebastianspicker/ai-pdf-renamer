"""Cover Textual TUI actions, worker messages, and single-file behavior."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from textual.css.query import QueryError

from folionym.config import RenamerConfig
from folionym.tui import ConfirmActionScreen, _RunFinished, _RunLog
from tests.conftest import make_pdf_symlink
from tests.conftest import make_tui_app as _make_app


async def _process_pdf_once(
    app: object,
    pilot: object,
    pdf_file: Path,
    suggestion: tuple[object, object, object],
    apply_result: tuple[bool, Path] | None = None,
) -> tuple[MagicMock, MagicMock]:
    """Drive one single-file TUI worker with controlled suggestion and rename results."""
    from textual.widgets import Input

    app.query_one("#single_file", Input).value = str(pdf_file)
    app.run_active = False
    mock_suggest = MagicMock(return_value=suggestion)
    mock_apply = MagicMock(return_value=apply_result)
    with (
        patch("folionym.tui.suggest_rename_for_file", mock_suggest),
        patch("folionym.tui.apply_single_rename", mock_apply),
        patch("folionym.tui.sanitize_filename_base", return_value="new_name"),
    ):
        app.process_one()
        await app.workers.wait_for_complete()
        await pilot.pause()
    return mock_suggest, mock_apply


async def _run_pdf_worker(
    tmp_path: Path,
    name: str,
    suggestion: tuple[object, object, object],
    apply_result: tuple[bool, Path] | None = None,
) -> tuple[MagicMock, MagicMock]:
    """Create a PDF, mount the app, and finish one single-file worker."""
    pdf_file = tmp_path / name
    pdf_file.write_bytes(b"%PDF-1.4 minimal")
    app = _make_app()
    async with app.run_test(size=(120, 40)) as pilot:
        return await _process_pdf_once(app, pilot, pdf_file, suggestion, apply_result)


async def test_build_config_returns_renamer_config() -> None:
    """_build_config() returns a RenamerConfig instance."""
    app = _make_app()
    async with app.run_test(size=(120, 40)):
        config = app.build_config(dry_run=True)
        assert isinstance(config, RenamerConfig)
        assert config.output.mode.dry_run is True


async def test_build_config_maps_directory() -> None:
    """Setting the #directory Input is reflected in the snapshot; dry_run maps correctly."""
    from textual.widgets import Input

    app = _make_app()
    async with app.run_test(size=(120, 40)):
        app.query_one("#directory", Input).value = "test-pdfs"
        config = app.build_config(dry_run=False)
        assert config.output.mode.dry_run is False
        snap = app.snapshot()
        assert snap["directory"] == "test-pdfs"


async def test_build_config_dry_run_false() -> None:
    """_build_config with dry_run=False produces a grouped false mode flag."""
    app = _make_app()
    async with app.run_test(size=(120, 40)):
        config = app.build_config(dry_run=False)
        assert isinstance(config, RenamerConfig)
        assert config.output.mode.dry_run is False


async def test_build_config_manual_mode() -> None:
    """_build_config with manual_mode=True propagates the flag."""
    app = _make_app()
    async with app.run_test(size=(120, 40)):
        config = app.build_config(dry_run=False, manual_mode=True)
        assert isinstance(config, RenamerConfig)
        assert config.output.mode.manual_mode is True
        assert config.output.mode.interactive is True


async def test_tui_omits_local_backend_controls() -> None:
    """The TUI is HTTP-only and must not expose local backend configuration."""
    app = _make_app()
    async with app.run_test(size=(120, 40)):
        with pytest.raises(QueryError):
            app.query_one("#llm_backend")
        with pytest.raises(QueryError):
            app.query_one("#llm_model_path")


async def test_workflow_rail_switches_between_existing_task_panes() -> None:
    """The reference-aligned rail remains real navigation, not decorative chrome."""
    from textual.widgets import TabbedContent

    app = _make_app()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.click("#nav-advanced")
        assert app.query_one("#workflow-tabs", TabbedContent).active == "advanced"
        assert app.query_one("#nav-advanced").has_class("active")

        await pilot.click("#nav-run")
        assert app.query_one("#workflow-tabs", TabbedContent).active == "run"
        assert app.query_one("#nav-run").has_class("active")


async def test_dry_run_output_populates_preview_table_and_inspector() -> None:
    """Recognized preview output is projected into the structured review surface."""
    from textual.widgets import DataTable, Static

    app = _make_app()
    async with app.run_test(size=(120, 40)) as pilot:
        app._activate_tab("run")
        app.post_message(_RunLog("Dry-run: would rename 'invoice.pdf' to '20240516-invoice-energy.pdf'"))
        await pilot.pause()

        table = app.query_one("#preview-table", DataTable)
        assert table.row_count == 1
        assert table.get_row_at(0)[1:] == ["invoice.pdf", "20240516-invoice-energy.pdf"]
        assert "invoice.pdf" in str(app.query_one("#inspector-source", Static).content)
        assert "20240516-invoice-energy.pdf" in str(app.query_one("#inspector-proposed", Static).content)


async def test_review_disclosure_distinguishes_external_http_model() -> None:
    """The trust panel must not claim document text stays local for an external endpoint."""
    from textual.widgets import Input, Static

    app = _make_app()
    async with app.run_test(size=(120, 40)):
        app.query_one("#llm_url", Input).value = "https://models.example.invalid/v1/completions"
        app._update_effective_configuration()

        disclosure = str(app.query_one("#privacy-disclosure", Static).content)
        assert "EXTERNAL HTTP MODEL" in disclosure
        assert "may leave this machine" in disclosure


async def test_start_run_empty_dir_logs_error() -> None:
    """_start_run with no directory does not schedule a Textual worker."""
    app = _make_app()
    async with app.run_test(size=(120, 40)):
        app.run_active = False

        with patch.object(app, "run_directory_worker") as mock_worker:
            app.start_run(dry_run=True)
            mock_worker.assert_not_called()


async def test_start_run_valid_dir_uses_textual_worker(tmp_path: Path) -> None:
    """A valid directory schedules the Textual-managed directory worker."""
    from textual.widgets import Input

    app = _make_app()
    async with app.run_test(size=(120, 40)) as pilot:
        app.query_one("#directory", Input).value = str(tmp_path)
        app.run_active = False

        with (
            patch.object(app, "run_directory_worker") as mock_worker,
            patch("folionym.tui._save_settings"),
        ):
            app.start_run(dry_run=True)
        await pilot.pause()

        assert app.run_active is True
        assert app.query_one("TabbedContent").active == "run"
        assert app.focused is not None
        assert app.focused.id == "btn-cancel"
        mock_worker.assert_called_once()


async def test_start_run_already_running() -> None:
    """_start_run while already running writes a warning and does nothing else."""
    app = _make_app()
    async with app.run_test(size=(120, 40)):
        app.run_active = True
        with patch.object(app, "run_directory_worker") as mock_worker:
            app.start_run(dry_run=True)
            mock_worker.assert_not_called()
        assert app.run_active is True


async def test_process_one_no_file_set() -> None:
    """_process_one with no single_file path writes an error to the log."""
    app = _make_app()
    async with app.run_test(size=(120, 40)):
        app.run_active = False

        with patch("folionym.tui.suggest_rename_for_file") as mock_suggest:
            app.process_one()
            mock_suggest.assert_not_called()


async def test_process_one_success(tmp_path: Path) -> None:
    """_process_one with a valid PDF and successful suggestion logs success."""
    mock_suggest, mock_apply = await _run_pdf_worker(
        tmp_path, "test-doc.pdf", ("new_name", {"category": "invoice"}, None), (True, tmp_path / "new_name.pdf")
    )

    mock_suggest.assert_called_once()
    mock_apply.assert_called_once()


async def test_process_one_failure(tmp_path: Path) -> None:
    """_process_one logs an error when suggest_rename_for_file returns an error."""
    mock_suggest, _ = await _run_pdf_worker(tmp_path, "fail-doc.pdf", (None, None, ValueError("extraction error")))

    mock_suggest.assert_called_once()


async def test_process_one_already_running() -> None:
    """_process_one while already running writes a warning and exits early."""
    app = _make_app()
    async with app.run_test(size=(120, 40)):
        app.run_active = True
        with patch("folionym.tui.suggest_rename_for_file") as mock_suggest:
            app.process_one()
            mock_suggest.assert_not_called()


async def test_process_one_not_pdf(tmp_path: Path) -> None:
    """_process_one with a non-PDF file writes an error to the log."""
    from textual.widgets import Input

    txt_file = tmp_path / "readme.txt"
    txt_file.write_text("not a PDF")

    app = _make_app()
    async with app.run_test(size=(120, 40)):
        app.query_one("#single_file", Input).value = str(txt_file)
        app.run_active = False

        with patch("folionym.tui.suggest_rename_for_file") as mock_suggest:
            app.process_one()
            mock_suggest.assert_not_called()


async def test_process_one_skipped(tmp_path: Path) -> None:
    """_process_one logs 'Skipped' when suggest returns None base with no error."""
    mock_suggest, _ = await _run_pdf_worker(tmp_path, "empty-doc.pdf", (None, None, None))

    mock_suggest.assert_called_once()


async def test_process_one_rename_fails(tmp_path: Path) -> None:
    """_process_one logs failure when apply_single_rename returns (False, ...)."""
    pdf_file = tmp_path / "test-doc.pdf"
    mock_suggest, mock_apply = await _run_pdf_worker(
        tmp_path, "test-doc.pdf", ("new_name", None, None), (False, pdf_file)
    )

    mock_suggest.assert_called_once()
    mock_apply.assert_called_once()


async def test_process_one_nonexistent_file(tmp_path: Path) -> None:
    """_process_one with a path to a file that does not exist writes an error."""
    from textual.widgets import Input

    pdf_file = tmp_path / "ghost.pdf"

    app = _make_app()
    async with app.run_test(size=(120, 40)):
        app.query_one("#single_file", Input).value = str(pdf_file)
        app.run_active = False

        with patch("folionym.tui.suggest_rename_for_file") as mock_suggest:
            app.process_one()
            mock_suggest.assert_not_called()


async def test_process_one_rejects_symlink(tmp_path: Path) -> None:
    """Single-file TUI mode must not dereference a PDF symlink."""
    from textual.widgets import Input

    _target, link = make_pdf_symlink(tmp_path)

    app = _make_app()
    async with app.run_test(size=(120, 40)):
        app.query_one("#single_file", Input).value = str(link)

        with patch("folionym.tui.suggest_rename_for_file") as mock_suggest:
            app.process_one()
            mock_suggest.assert_not_called()


async def test_preview_button_calls_start_run() -> None:
    """on_preview invokes _start_run(dry_run=True)."""
    app = _make_app()
    async with app.run_test(size=(120, 40)):
        with patch.object(app, "start_run") as mock_start:
            app.on_preview()
            mock_start.assert_called_once_with(dry_run=True)


async def test_ctrl_p_binding_starts_preview_instead_of_command_palette() -> None:
    """The documented preview shortcut must not be shadowed by Textual's palette."""
    app = _make_app()
    async with app.run_test(size=(120, 40)) as pilot:
        with patch.object(app, "start_run") as mock_start:
            await pilot.press("ctrl+p")
            mock_start.assert_called_once_with(dry_run=True)


async def test_apply_button_requests_confirmation() -> None:
    """The apply button enters the confirmation policy instead of mutating directly."""
    app = _make_app()
    async with app.run_test(size=(120, 40)):
        with patch.object(app, "request_apply") as mock_request:
            app.on_apply()
            mock_request.assert_called_once()


async def test_one_button_requests_confirmation() -> None:
    """The one-PDF button enters the confirmation policy instead of mutating directly."""
    app = _make_app()
    async with app.run_test(size=(120, 40)):
        with patch.object(app, "request_process_one") as mock_request:
            app.on_one()
            mock_request.assert_called_once()


async def test_apply_confirmation_defaults_to_cancel_and_requires_explicit_action(tmp_path: Path) -> None:
    """A batch apply must not start until the destructive modal action is chosen."""
    from textual.widgets import Input

    app = _make_app()
    async with app.run_test(size=(120, 40)) as pilot:
        app.query_one("#directory", Input).value = str(tmp_path)
        with patch.object(app, "start_run") as mock_start:
            app.request_apply()
            await pilot.pause()
            assert isinstance(app.screen, ConfirmActionScreen)

            await pilot.press("enter")
            await pilot.pause()
            mock_start.assert_not_called()

            app.request_apply()
            await pilot.pause()
            await pilot.click("#confirm-action")
            await pilot.pause()
            mock_start.assert_called_once_with(dry_run=False)


async def test_single_pdf_confirmation_starts_only_after_affirmative_result() -> None:
    """The single-file confirmation callback preserves cancellation as a no-op."""
    app = _make_app()
    async with app.run_test(size=(120, 40)):
        with patch.object(app, "process_one") as mock_process:
            app._finish_single_confirmation(False)
            mock_process.assert_not_called()
            app._finish_single_confirmation(True)
            mock_process.assert_called_once()


async def test_worker_messages_render_and_finish_run() -> None:
    """Worker messages update the UI without application-owned polling queues."""
    app = _make_app()
    async with app.run_test(size=(120, 40)) as pilot:
        app.run_active = True
        app.post_message(_RunLog("Processing 2/3: file2.pdf\n"))
        app.post_message(_RunFinished(True, "Completed"))
        await pilot.pause()

        assert app.run_active is False


async def test_preview_completion_uses_suggestion_language() -> None:
    """Preview counters describe proposed changes rather than completed renames."""
    from textual.widgets import Static

    app = _make_app()
    async with app.run_test(size=(120, 40)):
        app._run_is_preview = True
        app._run_counts = {"renamed": 2, "skipped": 1, "failed": 0}
        app._update_summary()

        assert "2 suggestions" in str(app.query_one("#run-summary", Static).content)
        assert "2 suggestions" in app._completion_summary_line()


async def test_supported_compact_viewport_keeps_run_actions_and_modal_visible(tmp_path: Path) -> None:
    """The documented 80x24 target keeps primary actions and confirmation reachable."""
    from textual.widgets import Button, Input, TabbedContent

    app = _make_app()
    async with app.run_test(size=(80, 24)) as pilot:
        app.query_one(TabbedContent).active = "run"
        await pilot.pause()
        assert app.query_one("#workflow-header").display is True
        assert app.query_one("#review-inspector").display is False
        for button_id in ("#btn-preview", "#btn-apply", "#btn-one", "#btn-cancel"):
            region = app.query_one(button_id, Button).region
            assert region.width > 0
            assert region.y + region.height <= 24

        app.query_one("#directory", Input).value = str(tmp_path)
        app.request_apply()
        await pilot.pause()
        assert isinstance(app.screen, ConfirmActionScreen)
        dialog_region = app.screen.query_one("#confirm-dialog").region
        assert dialog_region.width <= 80
        assert dialog_region.y + dialog_region.height <= 24
        assert app.screen.focused is app.screen.query_one("#confirm-cancel", Button)


async def test_preview_ledger_switches_between_empty_and_result_states() -> None:
    """The open ledger shows guidance before Preview and rows after recognized output."""
    from textual.widgets import DataTable, Static

    app = _make_app()
    async with app.run_test(size=(120, 40)):
        table = app.query_one("#preview-table", DataTable)
        empty = app.query_one("#preview-empty", Static)
        assert table.display is False
        assert empty.display is True

        app._record_preview_result("Dry-run: would rename 'scan.pdf' to '20260723-invoice-nordlicht.pdf'")
        assert table.display is True
        assert empty.display is False

        app._clear_preview_records()
        assert table.display is False
        assert empty.display is True


async def test_worker_failure_message_finishes_run() -> None:
    app = _make_app()
    async with app.run_test(size=(120, 40)) as pilot:
        app.run_active = True
        app.post_message(_RunFinished(False, "something broke"))
        await pilot.pause()

        assert app.run_active is False


async def test_quit_during_run_cancels_then_exits_after_worker_finishes() -> None:
    app = _make_app()
    async with app.run_test(size=(120, 40)) as pilot:
        app.run_active = True
        with patch.object(app, "exit") as mock_exit:
            await app.action_quit()
            assert app.stop_requested
            mock_exit.assert_not_called()
            app.post_message(_RunFinished(True, "Completed"))
            await pilot.pause()
            mock_exit.assert_called_once()

        assert app.run_active is False


async def test_get_bool_missing_widget_returns_default() -> None:
    """_get_bool returns default when the widget does not exist."""
    app = _make_app()
    async with app.run_test(size=(120, 40)):
        assert app.get_bool("nonexistent_checkbox") is False
        assert app.get_bool("nonexistent_checkbox", True) is True


async def test_get_select_missing_widget_returns_default() -> None:
    """_get_select returns default when the widget does not exist."""
    app = _make_app()
    async with app.run_test(size=(120, 40)):
        assert app.get_select("nonexistent_select") == ""
        assert app.get_select("nonexistent_select", "fallback") == "fallback"


def test_main_function_exists() -> None:
    """tui.main is a callable function."""
    from folionym import tui

    assert callable(tui.main)
