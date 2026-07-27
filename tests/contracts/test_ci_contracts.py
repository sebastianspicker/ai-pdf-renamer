"""Pin release-critical CI and security workflow behavior."""

from __future__ import annotations

import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def _read(path: str) -> str:
    return (REPO_ROOT / path).read_text(encoding="utf-8")


def test_ci_runs_release_gate_without_path_trigger_exclusions() -> None:
    workflow = _read(".github/workflows/ci.yml")

    assert "paths-ignore:" not in workflow
    assert "workflow_dispatch:" in workflow
    assert "name: Linux / Python 3.14.6 (release gate)" in workflow
    assert workflow.count('python-version: "3.14.6"') == 2
    assert "uv sync --frozen --extra dev --extra pdf --extra tui" in workflow
    assert "make release-check" in workflow
    assert "git ls-files | grep" not in workflow


def test_ci_has_targeted_macos_and_windows_smoke_jobs() -> None:
    workflow = _read(".github/workflows/ci.yml")

    assert "os: [macos-latest, windows-latest]" in workflow
    assert "uv sync --frozen --extra dev --extra pdf --extra tui" in workflow
    assert "tests/e2e/test_cli_e2e.py" in workflow
    assert "tests/unit/test_rename_ops.py" in workflow
    assert "tests/unit/test_undo_cli.py" in workflow


def test_security_uses_exact_release_patch_python() -> None:
    security = _read(".github/workflows/security.yml")

    assert 'python-version: "3.14.6"' in security
    assert "python -m pip install -e '.[pdf,tokens,ocr,tui]'" in security


def test_typecheck_gate_covers_runtime_scripts() -> None:
    makefile = _read("Makefile")
    pyproject = _read("pyproject.toml")

    assert "mypy src/folionym/ scripts/" in makefile
    assert 'mypy_path = ["src", "scripts"]' in pyproject
    assert "explicit_package_bases = true" in pyproject
    assert 'python_version = "3.14"' in pyproject


def test_project_declares_python_314_and_complexity_guards() -> None:
    config = tomllib.loads(_read("pyproject.toml"))

    assert config["project"]["requires-python"] == ">=3.14"
    assert "PyYAML>=6.0.0" in config["project"]["dependencies"]
    assert config["tool"]["ruff"]["target-version"] == "py314"
    for rule in ("C901", "PLR0911", "PLR0912", "PLR0913", "PLR0915"):
        assert rule in config["tool"]["ruff"]["lint"]["select"]


def test_exact_release_interpreter_is_used_by_local_build() -> None:
    assert _read(".python-version").strip() == "3.14.6"
    assert "$(UV) python find 3.14.6" in _read("Makefile")


def test_sdist_uses_an_explicit_public_file_allowlist() -> None:
    config = tomllib.loads(_read("pyproject.toml"))

    assert config["tool"]["hatch"]["build"]["targets"]["sdist"]["include"] == [
        "/CHANGELOG.md",
        "/CONTRIBUTING.md",
        "/DESIGN.md",
        "/LICENSE",
        "/Makefile",
        "/PRODUCT.md",
        "/README.md",
        "/RELEASING.md",
        "/SECURITY.md",
        "/docs/README.md",
        "/docs/frontend.md",
        "/docs/releases",
        "/docs/screenshots",
        "/docs/tui.md",
        "/frontend",
        "/pyproject.toml",
        "/scripts/capture_tui_screenshots.py",
        "/scripts/repository_hygiene.py",
        "/scripts/verify_distributions.py",
        "/src/folionym",
        "/tests",
        "/uv.lock",
    ]


def test_security_scanning_is_unconditional_and_has_no_trufflehog_exclusions() -> None:
    workflow = _read(".github/workflows/security.yml")

    assert "paths-ignore:" not in workflow
    assert "exclude-paths" not in workflow
    assert workflow.count("extra_args: --only-verified") == 4
    assert not (REPO_ROOT / ".github/trufflehog-exclude-paths.txt").exists()


def test_version_tags_run_ci_and_security_gates() -> None:
    assert 'tags: ["v*"]' in _read(".github/workflows/ci.yml")
    assert 'tags: ["v*"]' in _read(".github/workflows/security.yml")


def test_codeql_exclusions_do_not_control_workflow_or_secret_scan_execution() -> None:
    codeql = _read(".github/codeql/codeql-config.yml")
    security_workflow = _read(".github/workflows/security.yml")

    assert "paths-ignore:" in codeql
    assert "config-file: ./.github/codeql/codeql-config.yml" in security_workflow
    assert ".github/trufflehog-exclude-paths.txt" not in security_workflow
