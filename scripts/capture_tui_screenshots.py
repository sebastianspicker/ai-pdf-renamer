#!/usr/bin/env python3
"""Capture privacy-safe SVG screenshots of the Textual UI."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import os
import re
import tempfile
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

from textual.containers import ScrollableContainer
from textual.widgets import Checkbox, Footer, Input, ProgressBar, RichLog, Static, TabbedContent, TabPane

import folionym.tui as tui
import folionym.tui_state as tui_state
from folionym.tui_assets import PREVIEW_COLOR, SUCCESS_COLOR, WARNING_COLOR

_FONT_FACE_RE = re.compile(r"\n    @font-face \{.*?\n    \}\n    @font-face \{.*?\n    \}\n", re.DOTALL)
_TERMINAL_ID_RE = re.compile(r"terminal-\d+")
_SVG_ROOT_RE = re.compile(r"<svg\b[^>]*>")
_CAPTURE_SIZE = (120, 40)
_ADVANCED_SCROLL_Y = 18
_SCREENSHOT_METADATA = {
    "settings": (
        "Folionym setup",
        "Setup tab with the PDF naming pipeline and illustrative source paths.",
    ),
    "advanced": (
        "Folionym fine-tune settings",
        "Fine-tune tab with illustrative local model and processing configuration.",
    ),
    "preview": (
        "Folionym review and rename",
        "Review and rename tab showing an illustrative completed preview.",
    ),
}


def _self_contained_svg(svg: str, *, screen_name: str) -> str:
    """Remove remote dependencies and add stable metadata to a Rich SVG export."""
    try:
        title, description = _SCREENSHOT_METADATA[screen_name]
    except KeyError as error:
        raise ValueError(f"Unknown screenshot name: {screen_name}") from error
    svg, font_blocks = _FONT_FACE_RE.subn("\n", svg, count=1)
    if font_blocks != 1:
        raise RuntimeError("Textual screenshot font block changed; review the generated SVG before publishing")
    svg = svg.replace(
        "font-family: Fira Code, monospace;",
        "font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;",
    )
    svg = re.sub(r"<!--[^>]*textualize\.io[^>]*-->", "", svg, count=1)
    terminal_ids = set(_TERMINAL_ID_RE.findall(svg))
    if len(terminal_ids) != 1:
        raise RuntimeError("Textual screenshot ID format changed; review the generated SVG before publishing")
    svg = _TERMINAL_ID_RE.sub(f"folionym-{screen_name}", svg)
    metadata_id = f"folionym-{screen_name}-description"
    title_id = f"folionym-{screen_name}-title"
    root_match = _SVG_ROOT_RE.match(svg)
    if root_match is None:
        raise RuntimeError("Textual screenshot SVG root changed; review the generated SVG before publishing")
    accessible_root = f'{root_match.group()[:-1]} role="img" aria-labelledby="{title_id} {metadata_id}">'
    metadata = f'\n    <title id="{title_id}">{title}</title>\n    <desc id="{metadata_id}">{description}</desc>'
    svg = f"{accessible_root}{metadata}{svg[root_match.end() :]}"
    return svg


def _write_screenshot(app: tui.FolionymTUI, output_dir: Path, name: str) -> None:
    """Export one screenshot, embed stable accessible SVG metadata, and write it to the output directory."""
    svg = app.export_screenshot(simplify=True)
    output_dir.joinpath(f"tui-{name}.svg").write_text(
        _self_contained_svg(svg, screen_name=name),
        encoding="utf-8",
    )


async def _stabilize_footer(app: tui.FolionymTUI, pilot: Any) -> None:
    """Compose the binding footer before exporting the first frame."""
    footer = app.query_one(Footer)
    app.screen.refresh_bindings()
    await _wait_until(
        lambda: bool(footer.children),
        pilot,
        description="binding footer",
    )


async def _wait_until(
    condition: Callable[[], bool],
    pilot: Any,
    *,
    description: str,
    attempts: int = 250,
) -> None:
    """Wait for a Textual render condition with a bounded deterministic retry."""
    for _ in range(attempts):
        if condition():
            await pilot.pause()
            return
        await pilot.pause(0.02)
    raise RuntimeError(f"Timed out while stabilizing the {description}")


@contextlib.contextmanager
def _temporary_settings_path() -> Iterator[None]:
    """Temporarily redirect TUI settings to an isolated directory and restore the global path."""
    original_settings_path = tui_state.SETTINGS_PATH
    with tempfile.TemporaryDirectory(prefix="folionym-screenshots-") as settings_dir:
        tui_state.SETTINGS_PATH = Path(settings_dir) / "settings.json"
        try:
            yield
        finally:
            tui_state.SETTINGS_PATH = original_settings_path


@contextlib.contextmanager
def _stable_color_environment() -> Iterator[None]:
    """Force true-color exports without leaking changes into the caller's environment."""
    original_no_color = os.environ.pop("NO_COLOR", None)
    original_force_color = os.environ.get("FORCE_COLOR")
    os.environ["FORCE_COLOR"] = "1"
    try:
        yield
    finally:
        if original_no_color is not None:
            os.environ["NO_COLOR"] = original_no_color
        else:
            os.environ.pop("NO_COLOR", None)
        if original_force_color is not None:
            os.environ["FORCE_COLOR"] = original_force_color
        else:
            os.environ.pop("FORCE_COLOR", None)


async def _capture_settings(app: tui.FolionymTUI, pilot: Any, output_dir: Path) -> None:
    """Capture the settings UI state using illustrative, privacy-safe data."""
    app.query_one("#directory", Input).value = "~/Documents/PDFs"
    app.query_one("#single_file", Input).value = "~/Documents/PDFs/invoice.pdf"
    app.query_one("#project", Input).value = "personal-documents"
    app.query_one("#use_llm", Checkbox).value = False
    app.set_focus(None)
    await _stabilize_footer(app, pilot)
    _write_screenshot(app, output_dir, "settings")


async def _capture_advanced(app: tui.FolionymTUI, pilot: Any, output_dir: Path) -> None:
    """Capture the advanced UI state using illustrative, privacy-safe data."""
    tabs = app.query_one(TabbedContent)
    tabs.active = "advanced"
    await pilot.pause()
    app.query_one("#template", Input).value = "{date}_{category}_{keywords}"
    app.query_one("#backup_dir", Input).value = "~/Documents/PDF-backups"
    app.query_one("#llm_url", Input).value = "http://127.0.0.1:11434/v1/completions"
    app.query_one("#max_content_chars", Input).value = "12000"
    app.query_one("#workers", Input).value = "2"
    app.set_focus(None)
    await pilot.pause()
    container = app.query_one("#advanced", TabPane).query_one(ScrollableContainer)
    container.scroll_to(x=0, y=_ADVANCED_SCROLL_Y, animate=False, force=True, immediate=True)
    await pilot.pause()
    if container.scroll_y != _ADVANCED_SCROLL_Y:
        raise RuntimeError("Advanced screenshot scroll position could not be stabilized")
    _write_screenshot(app, output_dir, "advanced")


async def _capture_preview(app: tui.FolionymTUI, pilot: Any, output_dir: Path) -> None:
    """Capture the preview UI state using illustrative, privacy-safe data."""
    app.query_one(TabbedContent).active = "run"
    await pilot.pause()
    await pilot.resize_terminal(_CAPTURE_SIZE[0] + 1, _CAPTURE_SIZE[1])
    await pilot.resize_terminal(*_CAPTURE_SIZE)
    log = app.query_one("#run-log", RichLog)
    log.display = True
    await _wait_until(
        lambda: log.size.width > 0 and log.size.height > 0,
        pilot,
        description="preview log layout",
    )
    # RichLog normally flips this flag in its first Resize handler. A hidden
    # tab can miss that notification under a heavily loaded headless run even
    # though the final layout size is known; there are no deferred writes yet.
    log._size_known = True
    app.query_one("#run-status", Static).update(f"[bold {SUCCESS_COLOR}]DONE[/bold {SUCCESS_COLOR}]  Preview completed")
    app.query_one("#run-summary", Static).update(
        f"[{SUCCESS_COLOR}]2 rename suggestions[/{SUCCESS_COLOR}]  |  [{WARNING_COLOR}]1 skipped[/{WARNING_COLOR}]"
    )
    app.query_one("#metric-files", Static).update("[b]3[/b] PDFs")
    app.query_one("#metric-suggestions", Static).update("[b]2[/b] suggestions")
    app.query_one("#metric-skipped", Static).update("[b]1[/b] skipped")
    app._record_preview_result("Dry-run: would rename 'invoice.pdf' to '20260714-invoice-acme-office-supplies.pdf'")
    app._record_preview_result("Dry-run: would rename 'statement.pdf' to '20260701-bank-statement-june.pdf'")
    progress = app.query_one("#run-progress", ProgressBar)
    progress.display = True
    progress_bar = progress.query_one("#bar")
    progress.update(total=3, progress=3)
    await _wait_until(
        lambda: progress.percentage == 1.0 and getattr(progress_bar, "percentage", None) == 1.0,
        pilot,
        description="completed progress bar",
    )
    preview_lines = _preview_log_lines()
    for line in preview_lines:
        log.write(line)
    app.set_focus(None)
    await _wait_until(
        lambda: len(log.lines) >= len(preview_lines),
        pilot,
        description="preview log content",
    )
    _write_screenshot(app, output_dir, "preview")


def _preview_log_lines() -> tuple[str, ...]:
    """Return deterministic illustrative log lines for the dry-run preview."""
    return (
        f"[bold {PREVIEW_COLOR}]Preview (dry run)[/bold {PREVIEW_COLOR}]",
        "[dim]Directory: ~/Documents/PDFs[/dim]",
        "[dim]Processing 1/3: invoice.pdf[/dim]",
        f"[{PREVIEW_COLOR}]Dry-run[/{PREVIEW_COLOR}] [dim]invoice.pdf[/dim] "
        f"[{PREVIEW_COLOR} bold]->[/{PREVIEW_COLOR} bold] "
        "[bold]20260714-invoice-acme-office-supplies.pdf[/bold]",
        "[dim]Processing 2/3: statement.pdf[/dim]",
        f"[{PREVIEW_COLOR}]Dry-run[/{PREVIEW_COLOR}] [dim]statement.pdf[/dim] "
        f"[{PREVIEW_COLOR} bold]->[/{PREVIEW_COLOR} bold] "
        "[bold]20260701-bank-statement-june.pdf[/bold]",
        "[dim]Processing 3/3: scan.pdf[/dim]",
        f"[{WARNING_COLOR}]SKIP[/{WARNING_COLOR}] [dim]scan.pdf: no extractable text (OCR disabled)[/dim]",
        "[bold]Summary: 2 suggestions, 1 skipped, 0 failures[/bold]",
    )


async def capture(output_dir: Path) -> None:
    """Render the three public screenshot states without reading user settings."""
    output_dir.mkdir(parents=True, exist_ok=True)
    with _stable_color_environment(), _temporary_settings_path():
        app = tui.FolionymTUI()
        # Tab underline animations can be captured mid-transition, producing
        # different SVGs from identical headless runs.
        app.animation_level = "none"
        async with app.run_test(size=_CAPTURE_SIZE) as pilot:
            await _capture_settings(app, pilot, output_dir)
            await _capture_advanced(app, pilot, output_dir)
            await _capture_preview(app, pilot, output_dir)


def main() -> int:
    """Parse the output directory and capture all public TUI screenshots."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("docs/screenshots"),
        help="directory for the generated SVG files",
    )
    args = parser.parse_args()
    asyncio.run(capture(args.output_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
