from __future__ import annotations

from pathlib import Path

import pytest

from ai_pdf_renamer import data_paths
from ai_pdf_renamer.data_paths import (
    _discover_repo_root,
    category_aliases_path,
    data_dir,
    data_path,
    project_root,
)
from ai_pdf_renamer.llm_prompts import (
    _summary_doc_type_hint,
    build_analysis_prompt,
)
from ai_pdf_renamer.llm_schema import (
    DEFAULT_LLM_CATEGORY,
    DEFAULT_LLM_SUMMARY,
    validate_llm_document_result,
)
from ai_pdf_renamer.loaders import _file_mtime, load_meta_stopwords
from ai_pdf_renamer.logging_utils import StructuredLogFormatter, setup_logging
from ai_pdf_renamer.renamer_files import _is_safe_path, collect_pdf_files
from ai_pdf_renamer.rules import ProcessingRules, load_processing_rules


def test_data_path_falls_back_to_package_data(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(data_paths, "data_dir", lambda: tmp_path)

    path = data_paths.data_path("meta_stopwords.json")

    assert path.exists()
    assert path.name == "meta_stopwords.json"
    assert path.parent.name == "data"


def test_data_dir_uses_packaged_data_when_repo_root_not_found(monkeypatch) -> None:
    module_file = data_paths.Path(data_paths.__file__).resolve()
    expected = (module_file.parent / "data").resolve()

    monkeypatch.setattr(data_paths, "_discover_repo_root", lambda start=None: None)
    monkeypatch.setattr(data_paths.Path, "cwd", classmethod(lambda cls: data_paths.Path("/tmp/not-used-cwd")))

    resolved = data_paths.data_dir()
    assert resolved == expected


def test_data_path_ignores_non_file_candidate_and_falls_back_to_package(monkeypatch, tmp_path) -> None:
    bad_candidate = tmp_path / "meta_stopwords.json"
    bad_candidate.mkdir()

    monkeypatch.setattr(data_paths, "data_dir", lambda: tmp_path)
    path = data_paths.data_path("meta_stopwords.json")
    assert path.exists()
    assert path.is_file()
    assert path.parent.name == "data"


def test_category_aliases_path_ignores_directory_override(monkeypatch, tmp_path) -> None:
    bad_candidate = tmp_path / "category_aliases.json"
    bad_candidate.mkdir()

    monkeypatch.setattr(data_paths, "data_dir", lambda: tmp_path)
    path = data_paths.category_aliases_path()
    assert path.exists()
    assert path.is_file()
    assert path.name == "category_aliases.json"


# --- Merged from test_round6_small_modules.py ---

# ---------------------------------------------------------------------------
# 1. renamer_files.py
# ---------------------------------------------------------------------------


def test_collect_files_override(tmp_path: Path) -> None:
    """files_override list is returned as-is (no directory scan)."""
    # Create files in tmp_path but pass explicit override list
    real_pdf = tmp_path / "override.pdf"
    real_pdf.write_bytes(b"%PDF-1.4 dummy")
    non_pdf = tmp_path / "readme.txt"
    non_pdf.write_text("hi")

    result = collect_pdf_files(
        tmp_path,
        files_override=[real_pdf, non_pdf],
    )
    # Only the .pdf file survives the suffix filter
    assert result == [real_pdf]


def test_collect_skip_already_named(tmp_path: Path) -> None:
    """Files matching YYYYMMDD-*.pdf are skipped when skip_if_already_named=True."""
    already = tmp_path / "20250101-invoice.pdf"
    already.write_bytes(b"%PDF")
    normal = tmp_path / "report.pdf"
    normal.write_bytes(b"%PDF")

    result = collect_pdf_files(tmp_path, skip_if_already_named=True)
    assert normal in result
    assert already not in result


def test_collect_max_depth(tmp_path: Path) -> None:
    """max_depth=1 returns only top-level and depth-1 PDFs in recursive mode."""
    top = tmp_path / "top.pdf"
    top.write_bytes(b"%PDF")
    lvl1 = tmp_path / "sub" / "lvl1.pdf"
    lvl1.parent.mkdir()
    lvl1.write_bytes(b"%PDF")
    lvl2 = tmp_path / "sub" / "deep" / "lvl2.pdf"
    lvl2.parent.mkdir(parents=True)
    lvl2.write_bytes(b"%PDF")

    result = collect_pdf_files(tmp_path, recursive=True, max_depth=1)
    names = {p.name for p in result}
    assert "top.pdf" in names
    assert "lvl1.pdf" in names
    assert "lvl2.pdf" not in names


def test_collect_hidden_files_skipped(tmp_path: Path) -> None:
    """Dot-prefixed files are excluded in non-recursive mode."""
    hidden = tmp_path / ".hidden.pdf"
    hidden.write_bytes(b"%PDF")
    visible = tmp_path / "visible.pdf"
    visible.write_bytes(b"%PDF")

    result = collect_pdf_files(tmp_path)
    names = {p.name for p in result}
    assert "visible.pdf" in names
    assert ".hidden.pdf" not in names


def test_collect_include_pattern(tmp_path: Path) -> None:
    """include_patterns filters to matching filenames only."""
    inv = tmp_path / "invoice_123.pdf"
    inv.write_bytes(b"%PDF")
    other = tmp_path / "report.pdf"
    other.write_bytes(b"%PDF")

    result = collect_pdf_files(tmp_path, include_patterns=["invoice*"])
    assert [p.name for p in result] == ["invoice_123.pdf"]


def test_is_safe_path_symlink_outside_root(tmp_path: Path) -> None:
    """Symlink pointing outside the root directory is rejected."""
    import tempfile

    with tempfile.TemporaryDirectory() as outside:
        target = Path(outside) / "secret.pdf"
        target.write_bytes(b"%PDF")
        link = tmp_path / "link.pdf"
        link.symlink_to(target)
        assert not _is_safe_path(link, tmp_path)


def test_is_safe_path_symlink_inside_root(tmp_path: Path) -> None:
    """Symlink pointing inside the root directory is accepted."""
    target = tmp_path / "real.pdf"
    target.write_bytes(b"%PDF")
    link = tmp_path / "link.pdf"
    link.symlink_to(target)
    assert _is_safe_path(link, tmp_path)


def test_is_safe_path_not_symlink(tmp_path: Path) -> None:
    """Regular file is always safe."""
    f = tmp_path / "normal.pdf"
    f.write_bytes(b"%PDF")
    assert _is_safe_path(f, tmp_path)


def test_collect_recursive_skips_hidden(tmp_path: Path) -> None:
    """Hidden files are skipped in recursive mode."""
    hidden = tmp_path / ".hidden.pdf"
    hidden.write_bytes(b"%PDF")
    visible = tmp_path / "visible.pdf"
    visible.write_bytes(b"%PDF")

    result = collect_pdf_files(tmp_path, recursive=True)
    names = {p.name for p in result}
    assert "visible.pdf" in names
    assert ".hidden.pdf" not in names


def test_collect_recursive_skips_unsafe_symlinks(tmp_path: Path) -> None:
    """Symlinks pointing outside root are skipped in recursive mode."""
    import tempfile

    visible = tmp_path / "real.pdf"
    visible.write_bytes(b"%PDF")
    with tempfile.TemporaryDirectory() as outside:
        target = Path(outside) / "external.pdf"
        target.write_bytes(b"%PDF")
        link = tmp_path / "link.pdf"
        link.symlink_to(target)
        result = collect_pdf_files(tmp_path, recursive=True)
        names = {p.name for p in result}
        assert "real.pdf" in names
        assert "link.pdf" not in names


def test_collect_with_rules_filter(tmp_path: Path) -> None:
    """Rules skip_files_by_pattern filters out matching files."""
    keep = tmp_path / "important.pdf"
    keep.write_bytes(b"%PDF")
    skip = tmp_path / "TEMP_draft.pdf"
    skip.write_bytes(b"%PDF")

    rules = ProcessingRules(
        skip_llm_if_heuristic_category=[],
        force_category_by_pattern=[],
        skip_files_by_pattern=["TEMP_*"],
        allowed_categories=[],
    )
    result = collect_pdf_files(tmp_path, rules=rules)
    names = {p.name for p in result}
    assert "important.pdf" in names
    assert "TEMP_draft.pdf" not in names


# ---------------------------------------------------------------------------
# 1b. loaders.py
# ---------------------------------------------------------------------------


def test_load_meta_stopwords_missing_file(tmp_path: Path) -> None:
    """Missing file raises ValueError."""
    with pytest.raises(ValueError, match="Could not read"):
        load_meta_stopwords(tmp_path / "nonexistent.json")


def test_load_meta_stopwords_invalid_json(tmp_path: Path) -> None:
    """Invalid JSON raises ValueError."""
    bad = tmp_path / "bad.json"
    bad.write_text("{not valid json", encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid JSON"):
        load_meta_stopwords(bad)


def test_load_meta_stopwords_non_list(tmp_path: Path) -> None:
    """Non-list stopwords value falls back to empty set."""
    f = tmp_path / "data.json"
    f.write_text('{"stopwords": "not a list"}', encoding="utf-8")
    result = load_meta_stopwords(f)
    assert len(result.words) == 0


def test_file_mtime_missing() -> None:
    """Missing file returns 0.0."""
    assert _file_mtime("/nonexistent/path/file.json") == 0.0


# ---------------------------------------------------------------------------
# 1c. logging_utils.py
# ---------------------------------------------------------------------------


def test_structured_log_formatter_basic() -> None:
    """StructuredLogFormatter produces valid JSON with expected fields."""
    import json
    import logging

    fmt = StructuredLogFormatter()
    record = logging.LogRecord(
        name="test", level=logging.INFO, pathname="", lineno=0, msg="hello %s", args=("world",), exc_info=None
    )
    output = fmt.format(record)
    data = json.loads(output)
    assert data["message"] == "hello world"
    assert data["level"] == "INFO"


def test_structured_log_formatter_with_exception() -> None:
    """StructuredLogFormatter includes exception info."""
    import json
    import logging
    import sys

    fmt = StructuredLogFormatter()
    try:
        raise RuntimeError("boom")
    except RuntimeError:
        exc_info = sys.exc_info()
    record = logging.LogRecord(
        name="mylogger", level=logging.ERROR, pathname="", lineno=0, msg="fail", args=(), exc_info=exc_info
    )
    output = fmt.format(record)
    data = json.loads(output)
    assert "exception" in data
    assert "logger" in data
    assert data["logger"] == "mylogger"


def test_setup_logging_file_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """setup_logging handles OSError when log file can't be created."""
    import logging

    # Clear existing handlers to force fresh setup
    root = logging.getLogger()
    original_handlers = root.handlers[:]
    root.handlers = []
    try:
        # Use /dev/null/impossible to trigger OSError on file creation
        setup_logging(log_file="/dev/null/impossible/error.log", level=logging.DEBUG)
    finally:
        root.handlers = original_handlers


# ---------------------------------------------------------------------------
# 2. data_paths.py
# ---------------------------------------------------------------------------


def test_discover_repo_root_not_found(tmp_path: Path) -> None:
    """Returns None when no pyproject.toml exists in the tree."""
    leaf = tmp_path / "a" / "b" / "c"
    leaf.mkdir(parents=True)
    assert _discover_repo_root(leaf) is None


def test_project_root_fallback(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """project_root falls back to CWD when no pyproject.toml found."""
    monkeypatch.chdir(tmp_path)
    leaf = tmp_path / "no_project"
    leaf.mkdir()
    result = project_root(start=leaf)
    assert result == Path.cwd()


def test_load_rules_non_dict_json(tmp_path: Path) -> None:
    """JSON that is not an object returns None."""
    f = tmp_path / "rules.json"
    f.write_text('["not", "a", "dict"]', encoding="utf-8")
    assert load_processing_rules(f) is None


def test_load_rules_non_list_force_cat(tmp_path: Path) -> None:
    """Non-list force_category_by_pattern defaults to empty list."""
    f = tmp_path / "rules.json"
    f.write_text('{"force_category_by_pattern": "not a list"}', encoding="utf-8")
    result = load_processing_rules(f)
    assert result is not None
    assert result.force_category_by_pattern == []


def test_category_aliases_path_returns_path() -> None:
    """category_aliases_path returns an existing Path."""
    result = category_aliases_path()
    assert isinstance(result, Path)
    assert result.name == "category_aliases.json"


def test_data_path_env_override(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """AI_PDF_RENAMER_DATA_DIR env var redirects data_dir()."""
    monkeypatch.setenv("AI_PDF_RENAMER_DATA_DIR", str(tmp_path))
    result = data_dir()
    assert result == tmp_path.resolve()


def test_data_path_unknown_file() -> None:
    """data_path() raises ValueError for an unrecognised filename."""
    with pytest.raises(ValueError, match="Unsupported data file"):
        data_path("totally_bogus.json")  # type: ignore[arg-type]


def test_data_dir_fallback() -> None:
    """data_dir() always returns a valid Path (even without env override)."""
    result = data_dir()
    assert isinstance(result, Path)


# ---------------------------------------------------------------------------
# 3. llm_prompts.py
# ---------------------------------------------------------------------------


def test_build_analysis_prompt_german() -> None:
    """German language analysis prompt contains German instruction text."""
    prompt = build_analysis_prompt("de", "Testdokument")
    assert "Analysiere" in prompt
    assert "reines JSON" in prompt


def test_build_analysis_prompt_english() -> None:
    """English language analysis prompt contains English instruction text."""
    prompt = build_analysis_prompt("en", "Test document")
    assert "Analyze" in prompt
    assert "pure JSON" in prompt


def test_summary_doc_type_hint_german() -> None:
    """German doc-type hint references the suggested type."""
    hint = _summary_doc_type_hint("de", "Rechnung")
    assert "Rechnung" in hint
    assert "heuristisch" in hint


# ---------------------------------------------------------------------------
# 4. llm_schema.py
# ---------------------------------------------------------------------------


def test_validate_result_string_tokens() -> None:
    """Comma-separated string for final_summary_tokens is split into a list."""
    parsed: dict[str, object] = {
        "summary": "A summary",
        "keywords": ["a", "b"],
        "category": "invoice",
        "final_summary_tokens": "token1, token2, token3",
    }
    result = validate_llm_document_result(parsed)
    assert result.final_summary_tokens == ("token1", "token2", "token3")


def test_validate_result_category_na() -> None:
    """Category 'NA' is normalised to the default."""
    parsed: dict[str, object] = {
        "summary": "Some summary",
        "keywords": [],
        "category": "NA",
    }
    result = validate_llm_document_result(parsed)
    assert result.category == DEFAULT_LLM_CATEGORY


def test_validate_result_empty_summary() -> None:
    """Empty summary string falls back to default."""
    parsed: dict[str, object] = {
        "summary": "",
        "keywords": ["x"],
        "category": "invoice",
    }
    result = validate_llm_document_result(parsed)
    assert result.summary == DEFAULT_LLM_SUMMARY


def test_validate_result_valid() -> None:
    """All valid fields are preserved as-is."""
    parsed: dict[str, object] = {
        "summary": "An invoice from Acme Corp",
        "keywords": ["invoice", "acme"],
        "category": "invoice",
        "final_summary_tokens": ["token1", "token2"],
    }
    result = validate_llm_document_result(parsed)
    assert result.summary == "An invoice from Acme Corp"
    assert result.keywords == ("invoice", "acme")
    assert result.category == "invoice"
    assert result.final_summary_tokens == ("token1", "token2")
