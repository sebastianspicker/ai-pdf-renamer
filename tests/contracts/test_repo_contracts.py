"""Pin public documentation, packaging, API, and repository-wide code contracts."""

from __future__ import annotations

import ast
import re
import tomllib
from pathlib import Path
from typing import cast

from folionym import rename_ops, renamer
from folionym.config import RenamerConfig
from folionym.filename import generate_filename
from folionym.heuristics import CategoryCombineParams
from folionym.web_schema import UISettingsPayload

REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_SOURCE_ROOTS = (REPO_ROOT / "src", REPO_ROOT / "scripts", REPO_ROOT / "tests")
PRODUCTION_SOURCE_ROOTS = (REPO_ROOT / "src", REPO_ROOT / "scripts")
AUTHORED_CODE_SUFFIXES = {
    ".bash",
    ".cjs",
    ".css",
    ".js",
    ".jsx",
    ".mjs",
    ".ps1",
    ".py",
    ".pyi",
    ".sh",
    ".tcss",
    ".ts",
    ".tsx",
    ".zsh",
}
AUTHORED_CODE_EXCLUDED_DIRS = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".repowise",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "archive",
    "build",
    "coverage",
    "dist",
    "htmlcov",
    "node_modules",
    "web_dist",
}
MAX_AUTHORED_CODE_LINES = 600
IDENTITY_SCAN_ROOTS = (
    REPO_ROOT / ".github",
    REPO_ROOT / "docs",
    REPO_ROOT / "scripts",
    REPO_ROOT / "src",
    REPO_ROOT / "tests",
)
LOCAL_DOC_DIRECTORY_NAMES = {
    "agent",
    "archive",
    "deprecated",
    "ledgers",
    "reports",
    "source-audit",
    "status",
    "tmp",
}
IDENTITY_SCAN_ROOT_FILES = (
    ".gitignore",
    "CHANGELOG.md",
    "CONTRIBUTING.md",
    "Makefile",
    "README.md",
    "RELEASING.md",
    "SECURITY.md",
    "pyproject.toml",
    "uv.lock",
)
LEGACY_IDENTITIES = (
    "AI" + "-PDF-Renamer",
    "AI" + " PDF Renamer",
    "ai" + "-pdf-renamer",
    "ai" + "_pdf_renamer",
    "AI" + "_PDF_RENAMER",
    "AI" + "RenamerTUI",
)
CURRENT_REPOSITORY_URL = "https://github.com/sebastianspicker/folionym"
RENAME_OPS_PUBLIC_EXPORTS = {
    "FILENAME_RESERVED_WIN",
    "FILENAME_UNSAFE_RE",
    "MAX_LLM_FILENAME_LEN",
    "MAX_RENAME_RETRIES",
    "RenameApplyOptions",
    "RenameAttemptState",
    "RenameRetryContext",
    "apply_single_rename",
    "is_path_within",
    "sanitize_filename_base",
    "sanitize_filename_from_llm",
}


def _read_repo_file(path: str) -> str:
    return (REPO_ROOT / path).read_text(encoding="utf-8")


def _typescript_string_union(source: str, type_name: str) -> set[str]:
    """Extract string members from one exported TypeScript union declaration."""
    match = re.search(rf"export type {re.escape(type_name)}\s*=\s*(.*?);", source, re.DOTALL)
    assert match is not None, f"Missing TypeScript union: {type_name}"
    return set(re.findall(r'"([^"]+)"', match.group(1)))


def _extract_literal_choices(source: str, *, flag: str) -> list[str]:
    pattern = re.compile(rf'{re.escape(flag)}".*?choices=\[(.*?)\]', re.DOTALL)
    match = pattern.search(source)
    assert match is not None
    choices: object = ast.literal_eval("[" + match.group(1) + "]")
    assert isinstance(choices, list)
    assert all(isinstance(choice, str) for choice in choices)
    return cast(list[str], choices)


def _python_files(roots: tuple[Path, ...]) -> list[Path]:
    """Return deterministic Python source paths below the requested roots."""
    return sorted(path for root in roots for path in root.rglob("*.py"))


def _authored_code_files() -> list[Path]:
    """Return repository-wide maintained code while excluding archived, generated, and vendored trees."""
    return sorted(
        path
        for path in REPO_ROOT.rglob("*")
        if path.is_file()
        and path.suffix in AUTHORED_CODE_SUFFIXES
        and not AUTHORED_CODE_EXCLUDED_DIRS.intersection(path.relative_to(REPO_ROOT).parts)
    )


def _identity_surface_files() -> list[Path]:
    """Return public and executable text files that must use the Folionym identity."""
    suffixes = {".json", ".lock", ".md", ".py", ".svg", ".toml", ".yaml", ".yml"}
    nested = [
        path
        for root in IDENTITY_SCAN_ROOTS
        for path in root.rglob("*")
        if path.suffix in suffixes and not _is_local_documentation(path)
    ]
    root_files = [REPO_ROOT / name for name in IDENTITY_SCAN_ROOT_FILES]
    return sorted((*nested, *root_files))


def _is_local_documentation(path: Path) -> bool:
    """Return whether a documentation path belongs to an ignored local-only lane."""
    try:
        relative = path.relative_to(REPO_ROOT / "docs")
    except ValueError:
        return False
    return bool(relative.parts) and (
        relative.parts[0] in LOCAL_DOC_DIRECTORY_NAMES or relative.parts[0].startswith("audit-")
    )


def test_legacy_product_identity_is_absent_from_paths_and_content() -> None:
    """Prevent retired product names while allowing the current GitHub repository slug."""
    findings = []
    for path in _identity_surface_files():
        relative = str(path.relative_to(REPO_ROOT))
        content = path.read_text(encoding="utf-8").replace(CURRENT_REPOSITORY_URL, "")
        for legacy in LEGACY_IDENTITIES:
            if legacy in relative or legacy in content:
                findings.append(f"{relative}: {legacy}")

    assert findings == []


def test_all_python_modules_explain_their_scope() -> None:
    """Require a module docstring so each file states its architectural role."""
    missing = []
    for path in _python_files(PYTHON_SOURCE_ROOTS):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        if ast.get_docstring(tree, clean=False) is None:
            missing.append(str(path.relative_to(REPO_ROOT)))

    assert missing == []


def test_authored_code_files_stay_within_maintainability_budget() -> None:
    """Keep authored code reviewable by requiring cohesive files at or below 600 lines."""
    oversized = []
    for path in _authored_code_files():
        line_count = len(path.read_text(encoding="utf-8").splitlines())
        if line_count > MAX_AUTHORED_CODE_LINES:
            oversized.append(f"{path.relative_to(REPO_ROOT)}: {line_count}")

    assert oversized == []


def test_browser_settings_keys_match_the_backend_schema() -> None:
    """Keep the manually mirrored browser key unions aligned with Pydantic field types."""
    source = _read_repo_file("frontend/src/types.ts")
    backend_fields = UISettingsPayload.model_fields
    backend_text = {name for name, field in backend_fields.items() if field.annotation is str}
    backend_boolean = {name for name, field in backend_fields.items() if field.annotation is bool}

    assert _typescript_string_union(source, "SettingsTextKey") == backend_text
    assert _typescript_string_union(source, "SettingsBooleanKey") == backend_boolean
    assert backend_text | backend_boolean == set(backend_fields)


def test_production_declarations_explain_their_contract() -> None:
    """Require production classes and callables to document purpose or invariants."""
    missing = []
    declaration_types = (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
    for path in _python_files(PRODUCTION_SOURCE_ROOTS):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, declaration_types) and ast.get_docstring(node, clean=False) is None:
                relative_path = path.relative_to(REPO_ROOT)
                missing.append(f"{relative_path}:{node.lineno}:{node.name}")

    assert missing == []


def test_readme_documents_current_defaults_and_precedence() -> None:
    readme = _read_repo_file("README.md")

    assert "88% coverage" not in readme
    assert "85%" not in readme
    assert "Current release:" not in readme
    assert "1. CLI flags" in readme
    assert "2. Environment defaults (for supported settings)" in readme
    assert "3. Config file values (`--config` JSON/YAML)" in readme
    assert "~/.local/share/folionym/error.log" in readme
    assert "`--vision-fallback`, `--vision-first`" in readme
    assert "`--use-vision-fallback`" not in readme
    assert "FOLIONYM_CACHE_DIR" in readme
    assert "--preset scanned --vision-model llava" in readme
    assert "--vision-first --vision-model llava" in readme
    assert "Apply and Rename one PDF require explicit confirmation" in readme


def test_tui_docs_describe_independent_runs_and_settings_save_trigger() -> None:
    tui_docs = _read_repo_file("docs/tui.md")

    assert "Apply starts a separate run" in tui_docs
    assert "confirmation whose first action is Cancel" in tui_docs
    assert "Starting a valid directory Preview or confirmed Apply run saves" in tui_docs
    assert "Edits are not autosaved" in tui_docs


def test_readme_and_tui_match_current_cli_preset_surface() -> None:
    readme = _read_repo_file("README.md")
    cli_parser = _read_repo_file("src/folionym/cli_parser.py")
    tui_assets_source = _read_repo_file("src/folionym/tui_assets.py")

    cli_presets = _extract_literal_choices(cli_parser, flag="--preset")
    assert "`--preset` (`high-confidence-heuristic`, `scanned`, `fast`, `accurate`, `batch`)" in readme

    tui_match = re.search(r"_PRESETS\s*=\s*(\[[^\]]*\])", tui_assets_source, re.DOTALL)
    assert tui_match is not None
    tui_presets = [value for _label, value in ast.literal_eval(tui_match.group(1)) if value]

    assert tui_presets == cli_presets


def test_contributing_matches_current_ci_job_shape() -> None:
    contributing = _read_repo_file("CONTRIBUTING.md")

    assert "Python-version matrix" not in contributing
    assert "Linux Python 3.14.6 release gate" in contributing
    assert "macOS and Windows targeted smoke jobs" in contributing


def test_changelog_tracks_current_package_version() -> None:
    changelog = _read_repo_file("CHANGELOG.md")
    init_source = _read_repo_file("src/folionym/__init__.py")

    version_match = re.search(r'^__version__ = "([^"]+)"$', init_source, re.MULTILINE)
    assert version_match is not None
    assert "## [Unreleased]" in changelog
    assert f"Target prerelease: `{version_match.group(1)}`" in changelog
    assert f"## [{version_match.group(1)}]" not in changelog


def test_public_docs_do_not_duplicate_version_or_coverage_floor() -> None:
    docs_index = _read_repo_file("docs/README.md")

    assert "Current release:" not in docs_index
    assert "coverage floor" not in docs_index.lower()


def test_security_documents_cli_and_backend_llm_defaults() -> None:
    security = _read_repo_file("SECURITY.md")

    assert "http://127.0.0.1:11434/v1/completions" in security
    assert "http://127.0.0.1:8080/v1/completions" in security


def test_documented_public_api_uses_owning_modules_without_compatibility_reexports() -> None:
    assert callable(generate_filename)
    assert callable(renamer.rename_pdfs_in_directory)
    assert RenamerConfig.__module__ == "folionym.config"
    assert CategoryCombineParams.__module__ == "folionym.heuristics"
    assert not hasattr(renamer, "RenamerConfig")
    assert not hasattr(renamer, "generate_filename")
    assert not hasattr(renamer, "collect_pdf_files")
    assert not hasattr(renamer, "CategoryCombineParams")


def test_rename_ops_facade_preserves_its_documented_exports() -> None:
    assert set(rename_ops.__all__) == RENAME_OPS_PUBLIC_EXPORTS
    assert all(hasattr(rename_ops, name) for name in RENAME_OPS_PUBLIC_EXPORTS)


def test_public_tui_screenshots_are_accessible_and_self_contained() -> None:
    for name in ("tui-settings.svg", "tui-advanced.svg", "tui-preview.svg"):
        screenshot = _read_repo_file(f"docs/screenshots/{name}")

        assert screenshot.startswith("<svg")
        assert "<title " in screenshot
        assert "<desc " in screenshot
        assert "Folionym" in screenshot
        assert "cdnjs.cloudflare.com" not in screenshot

    advanced = _read_repo_file("docs/screenshots/tui-advanced.svg")
    assert "LLM&#160;backend" not in advanced
    assert "LLM&#160;model&#160;path" not in advanced


def test_alpha_public_docs_match_current_package_version() -> None:
    init_source = _read_repo_file("src/folionym/__init__.py")
    version_match = re.search(r'^__version__ = "([^"]+)"$', init_source, re.MULTILINE)
    assert version_match is not None
    version = version_match.group(1)

    assert version == "0.4.0a1"
    for path in (
        "README.md",
        "docs/README.md",
        "docs/releases/0.4.0a1.md",
        "docs/tui.md",
        "RELEASING.md",
        "CHANGELOG.md",
    ):
        assert version in _read_repo_file(path)


def test_release_contract_removes_gui_alias_and_retired_llm_modes() -> None:
    config = tomllib.loads(_read_repo_file("pyproject.toml"))
    readme = _read_repo_file("README.md")
    contributing = _read_repo_file("CONTRIBUTING.md")

    assert "folionym-gui" not in config["project"]["scripts"]
    assert "in-process" not in readme
    assert "embedding-assisted" not in readme.lower()
    assert "HTTP-only LLM client" in readme
    assert "standard-library argparse" in readme
    assert "compatibility imports" not in contributing
