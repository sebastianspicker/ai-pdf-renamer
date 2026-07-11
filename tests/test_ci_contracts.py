from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _read(path: str) -> str:
    return (REPO_ROOT / path).read_text(encoding="utf-8")


def test_ci_runs_release_gate_without_path_trigger_exclusions() -> None:
    workflow = _read(".github/workflows/ci.yml")

    assert "paths-ignore:" not in workflow
    assert "name: Python 3.11 (lint + tests)" in workflow
    assert "uv sync --frozen --extra dev --extra pdf --extra tui" in workflow
    assert "make release-check" in workflow
    assert "git ls-files | grep" not in workflow


def test_security_scanning_is_unconditional_and_has_no_trufflehog_exclusions() -> None:
    workflow = _read(".github/workflows/security.yml")

    assert "paths-ignore:" not in workflow
    assert "exclude-paths" not in workflow
    assert workflow.count("extra_args: --only-verified") == 4
    assert not (REPO_ROOT / ".github/trufflehog-exclude-paths.txt").exists()


def test_codeql_exclusions_do_not_control_workflow_or_secret_scan_execution() -> None:
    codeql = _read(".github/codeql/codeql-config.yml")
    security_workflow = _read(".github/workflows/security.yml")

    assert "paths-ignore:" in codeql
    assert "config-file: ./.github/codeql/codeql-config.yml" in security_workflow
    assert ".github/trufflehog-exclude-paths.txt" not in security_workflow
