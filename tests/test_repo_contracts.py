from __future__ import annotations

import ast
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _read_repo_file(path: str) -> str:
    return (REPO_ROOT / path).read_text(encoding="utf-8")


def _extract_literal_choices(source: str, *, flag: str) -> list[str]:
    pattern = re.compile(rf'{re.escape(flag)}".*?choices=\[(.*?)\]', re.DOTALL)
    match = pattern.search(source)
    assert match is not None
    return ast.literal_eval("[" + match.group(1) + "]")


def test_readme_documents_current_defaults_and_precedence() -> None:
    readme = _read_repo_file("README.md")

    assert "88% coverage" not in readme
    assert "current floor: 85%" in readme
    assert "1. CLI flags" in readme
    assert "2. Environment defaults (for supported settings)" in readme
    assert "3. Config file values (`--config` JSON/YAML)" in readme
    assert "~/.local/share/ai-pdf-renamer/error.log" in readme
    assert "`--vision-fallback`, `--vision-first`" in readme
    assert "`--use-vision-fallback`" not in readme
    assert "AI_PDF_RENAMER_CACHE_DIR" in readme


def test_readme_and_tui_match_current_cli_preset_surface() -> None:
    readme = _read_repo_file("README.md")
    cli_parser = _read_repo_file("src/ai_pdf_renamer/cli_parser.py")
    tui_source = _read_repo_file("src/ai_pdf_renamer/tui.py")

    cli_presets = _extract_literal_choices(cli_parser, flag="--preset")
    assert "`--preset` (`high-confidence-heuristic`, `scanned`, `fast`, `accurate`, `batch`)" in readme

    tui_match = re.search(r"_PRESETS\s*=\s*(\[[^\]]*\])", tui_source, re.DOTALL)
    assert tui_match is not None
    tui_presets = [value for _label, value in ast.literal_eval(tui_match.group(1)) if value]

    assert tui_presets == cli_presets


def test_contributing_matches_current_ci_job_shape() -> None:
    contributing = _read_repo_file("CONTRIBUTING.md")

    assert "Python-version matrix" not in contributing
    assert "single Python 3.11 job" in contributing


def test_changelog_tracks_current_coverage_gate() -> None:
    changelog = _read_repo_file("CHANGELOG.md")

    assert "88% coverage" not in changelog
    assert "Coverage threshold raised from 50% to 85%." in changelog


def test_security_documents_cli_and_backend_llm_defaults() -> None:
    security = _read_repo_file("SECURITY.md")

    assert "http://127.0.0.1:11434/v1/completions" in security
    assert "http://127.0.0.1:8080/v1/completions" in security
