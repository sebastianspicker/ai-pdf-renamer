"""Ensure public TUI screenshots are private-data-free and reproducible."""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
from pathlib import Path

import pytest

import folionym.tui_state as tui_state
from scripts.capture_tui_screenshots import capture

REPO_ROOT = Path(__file__).resolve().parents[2]
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


def _run_capture_subprocess(output_dir: Path) -> None:
    env = {**os.environ, "PYTHONHASHSEED": "0"}
    subprocess.run(
        [sys.executable, "scripts/capture_tui_screenshots.py", "--output-dir", str(output_dir)],
        check=True,
        cwd=REPO_ROOT,
        env=env,
    )


def test_capture_is_reproducible_across_seeded_subprocesses(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    _run_capture_subprocess(first)
    _run_capture_subprocess(second)

    for screen_name, (title, description) in _SCREENSHOT_METADATA.items():
        name = f"tui-{screen_name}.svg"
        first_svg = first.joinpath(name).read_text(encoding="utf-8")
        assert first_svg.encode() == second.joinpath(name).read_bytes()
        assert first_svg.encode() == REPO_ROOT.joinpath("docs", "screenshots", name).read_bytes()
        title_id = f"folionym-{screen_name}-title"
        description_id = f"folionym-{screen_name}-description"
        assert (
            f'<svg class="rich-terminal" viewBox="0 0 1482 1026.0" '
            f'xmlns="http://www.w3.org/2000/svg" role="img" '
            f'aria-labelledby="{title_id} {description_id}">' in first_svg
        )
        assert f'<title id="{title_id}">{title}</title>' in first_svg
        assert f'<desc id="{description_id}">{description}</desc>' in first_svg
        assert "Preview&#160;(dry&#160;run)" in first_svg


def test_capture_restores_settings_path(tmp_path: Path) -> None:
    original_settings_path = tui_state.SETTINGS_PATH
    asyncio.run(capture(tmp_path / "screenshots"))

    assert original_settings_path == tui_state.SETTINGS_PATH


def test_capture_restores_color_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Screenshot color forcing must not change the caller's environment."""
    monkeypatch.setenv("NO_COLOR", "1")
    monkeypatch.delenv("FORCE_COLOR", raising=False)

    asyncio.run(capture(tmp_path / "screenshots"))

    assert os.environ["NO_COLOR"] == "1"
    assert "FORCE_COLOR" not in os.environ
