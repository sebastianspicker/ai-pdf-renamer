from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _read_repo_file(path: str) -> str:
    return (REPO_ROOT / path).read_text(encoding="utf-8")


def test_readme_documents_current_defaults_and_precedence() -> None:
    readme = _read_repo_file("README.md")

    assert "88% coverage" not in readme
    assert "current floor: 85%" in readme
    assert "1. CLI flags" in readme
    assert "2. Environment defaults (for supported settings)" in readme
    assert "3. Config file values (`--config` JSON/YAML)" in readme
    assert "~/.local/share/ai-pdf-renamer/error.log" in readme


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
