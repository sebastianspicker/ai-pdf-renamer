"""Cover interactive CLI option resolution and initial command behavior."""

from __future__ import annotations

import json

import pytest

import folionym.cli as cli_mod
from folionym.cli import OptionPromptSpec, OptionResolutionRequest, _resolve_option
from tests.conftest import make_cli_main_args
from tests.conftest import make_cli_namespace as _ns

_PDF_TEST_SETUP_EXCEPTIONS = (AttributeError, ImportError, OSError, RuntimeError, TypeError, ValueError)


def _capture_cli_rename(monkeypatch: pytest.MonkeyPatch, cli: object) -> dict[str, object]:
    """Record the directory and resolved configuration passed into a CLI rename."""
    captured: dict[str, object] = {}

    def fake_rename(directory: object, *, config: object, files_override: object = None) -> None:
        captured["directory"] = directory
        captured["config"] = config

    monkeypatch.setattr(cli, "rename_pdfs_in_directory", fake_rename)
    return captured


def test_cli_rejects_empty_dir(monkeypatch) -> None:
    import folionym.cli as cli

    monkeypatch.setattr(cli, "setup_logging", lambda **k: None)

    with pytest.raises(SystemExit) as excinfo:
        cli.main(make_cli_main_args(""))

    assert excinfo.value.code == 1


def test_cli_exits_on_missing_directory(monkeypatch, tmp_path) -> None:
    import folionym.cli as cli

    monkeypatch.setattr(cli, "setup_logging", lambda **k: None)

    missing = tmp_path / "missing"
    with pytest.raises(SystemExit) as excinfo:
        cli.main(make_cli_main_args(missing))

    assert excinfo.value.code == 1


def test_cli_reprompts_on_invalid_choices(monkeypatch, tmp_path) -> None:
    import builtins

    import folionym.cli as cli

    monkeypatch.setattr(cli, "setup_logging", lambda **k: None)
    monkeypatch.setattr(cli, "_is_interactive", lambda: True)

    inputs = iter(["fr", "en", "badcase", "snakecase"])
    monkeypatch.setattr(builtins, "input", lambda _prompt: next(inputs))

    captured = _capture_cli_rename(monkeypatch, cli)

    cli.main(
        [
            "--dir",
            str(tmp_path),
            "--project",
            "",
            "--version",
            "",
        ]
    )

    config = captured["config"]
    assert config.output.naming.language == "en"
    assert config.output.naming.desired_case == "snakeCase"


def test_cli_exits_on_invalid_json_in_data_file(monkeypatch, tmp_path) -> None:
    import folionym.cli as cli

    monkeypatch.setattr(cli, "setup_logging", lambda **k: None)

    def _raise_json_error(*args, **kwargs):
        raise json.JSONDecodeError("Expecting value", doc="", pos=0)

    monkeypatch.setattr(cli, "rename_pdfs_in_directory", _raise_json_error)

    with pytest.raises(SystemExit) as excinfo:
        cli.main(make_cli_main_args(tmp_path))

    assert excinfo.value.code == 1


def test_cli_exits_on_missing_data_file(monkeypatch, tmp_path) -> None:
    """When a required data file is missing, CLI exits with a clear message."""
    import folionym.cli as cli

    monkeypatch.setattr(cli, "setup_logging", lambda **k: None)

    def _raise_file_not_found(*args, **kwargs):
        raise FileNotFoundError(
            "Data file 'heuristic_scores.json' not found. "
            "Looked in: /nonexistent and /packaged. "
            "Set FOLIONYM_DATA_DIR or run from the project root."
        )

    monkeypatch.setattr(cli, "rename_pdfs_in_directory", _raise_file_not_found)

    with pytest.raises(SystemExit) as excinfo:
        cli.main(make_cli_main_args(tmp_path))

    assert excinfo.value.code == 1


def test_cli_exits_on_broken_json_in_data_file_integration(monkeypatch, tmp_path) -> None:
    """Broken JSON in data dir + PDF with content: CLI exits with clear message."""
    import folionym.cli as cli
    import folionym.loaders as loaders

    monkeypatch.setattr(cli, "setup_logging", lambda **k: None)
    monkeypatch.setenv("FOLIONYM_DATA_DIR", str(tmp_path))

    (tmp_path / "meta_stopwords.json").write_text('{"stopwords": []}', encoding="utf-8")
    (tmp_path / "heuristic_scores.json").write_text("{ invalid }", encoding="utf-8")

    pdf_dir = tmp_path / "pdfs"
    pdf_dir.mkdir()

    try:
        import fitz

        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((72, 72), "Sample document text for renaming.")
        doc.save(str(pdf_dir / "dummy.pdf"))
        doc.close()
    except _PDF_TEST_SETUP_EXCEPTIONS:
        pytest.skip("PyMuPDF (fitz) required for integration test")

    loaders.stopwords_cached.cache_clear()
    loaders.heuristic_scorer_cached.cache_clear()

    with pytest.raises(SystemExit) as excinfo:
        cli.main(make_cli_main_args(pdf_dir))

    assert excinfo.value.code == 1


def test_cli_non_interactive_uses_defaults_without_hanging(monkeypatch, tmp_path) -> None:
    """With no TTY, CLI uses defaults and does not prompt (suitable for CI/cron)."""
    import folionym.cli as cli

    monkeypatch.setattr(cli, "setup_logging", lambda **k: None)
    monkeypatch.setattr(cli, "_is_interactive", lambda: False)

    captured = _capture_cli_rename(monkeypatch, cli)

    # Omit --language, --case, --project, --version; only --dir.
    cli.main(["--dir", str(tmp_path)])

    assert captured.get("directory") == str(tmp_path)
    config = captured["config"]
    assert config.output.naming.language == "de"
    assert config.output.naming.desired_case == "kebabCase"
    assert config.output.naming.project == ""
    assert config.output.naming.version == ""


def _patch_cli_llm_metadata(monkeypatch) -> None:
    import folionym.filename_llm_metadata as filename_llm_metadata
    import folionym.filename_metadata as filename_metadata

    monkeypatch.setattr(
        filename_llm_metadata,
        "get_document_summary",
        lambda *args, **kwargs: "Mocked summary for testing",
    )
    monkeypatch.setattr(
        filename_llm_metadata,
        "get_document_keywords",
        lambda *args, **kwargs: ["mock", "keywords"],
    )
    monkeypatch.setattr(filename_metadata, "get_document_category", lambda *args, **kwargs: "document")
    monkeypatch.setattr(
        filename_llm_metadata,
        "get_final_summary_tokens",
        lambda *args, **kwargs: ["mocked", "tokens"],
    )


def _write_sample_pdf_or_skip(path) -> None:
    try:
        import fitz

        doc = fitz.open()
        try:
            page = doc.new_page()
            page.insert_text((72, 72), "Sample text so PDF is not empty.")
            doc.save(str(path))
        finally:
            doc.close()
    except _PDF_TEST_SETUP_EXCEPTIONS:
        pytest.skip("PyMuPDF (fitz) required for this test")


def test_cli_renames_with_mocked_llm_no_network(monkeypatch, tmp_path) -> None:
    """Rename flow without real LLM when get_document_* are mocked (CI-safe)."""
    import folionym.cli as cli

    monkeypatch.setattr(cli, "setup_logging", lambda **k: None)
    monkeypatch.setattr(cli, "_is_interactive", lambda: False)
    _patch_cli_llm_metadata(monkeypatch)
    _write_sample_pdf_or_skip(tmp_path / "sample.pdf")

    cli.main(make_cli_main_args(tmp_path))

    pdfs = list(tmp_path.glob("*.pdf"))
    assert len(pdfs) == 1
    # Renamed file should no longer be named sample.pdf (content-based name)
    assert pdfs[0].name != "sample.pdf" or "mock" in pdfs[0].name.lower()


class TestResolveOptionFreePrompt:
    """Tests for the free_prompt branch in _resolve_option."""

    def test_resolve_option_valid_choice(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When input() returns a non-empty string, it is accepted as the value."""
        monkeypatch.setattr(cli_mod, "_is_interactive", lambda: True)
        monkeypatch.setattr("builtins.input", lambda _prompt: "my-project")

        args = _ns(project=None)
        result = _resolve_option(args, _free_project_request(""))

        assert result == "my-project"

    def test_resolve_option_empty_input_returns_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When input() returns empty string, the default is used."""
        monkeypatch.setattr(cli_mod, "_is_interactive", lambda: True)
        monkeypatch.setattr("builtins.input", lambda _prompt: "")

        args = _ns(project=None)
        result = _resolve_option(args, _free_project_request("fallback-default"))

        assert result == "fallback-default"

    def test_resolve_option_eof_exits(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When input() raises EOFError, SystemExit(1) is raised."""
        monkeypatch.setattr(cli_mod, "_is_interactive", lambda: True)

        def _raise_eof(_prompt: str) -> str:
            raise EOFError

        monkeypatch.setattr("builtins.input", _raise_eof)

        args = _ns(project=None)
        with pytest.raises(SystemExit) as exc_info:
            _resolve_option(args, _free_project_request(""))

        assert exc_info.value.code == 1

    def test_resolve_option_keyboard_interrupt(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When input() raises KeyboardInterrupt, SystemExit(130) is raised."""
        monkeypatch.setattr(cli_mod, "_is_interactive", lambda: True)

        def _raise_ki(_prompt: str) -> str:
            raise KeyboardInterrupt

        monkeypatch.setattr("builtins.input", _raise_ki)

        args = _ns(project=None)
        with pytest.raises(SystemExit) as exc_info:
            _resolve_option(args, _free_project_request(""))

        assert exc_info.value.code == 130


def _free_project_request(default: str) -> OptionResolutionRequest:
    return OptionResolutionRequest(
        attr="project",
        file_defaults={},
        file_key="project",
        default=default,
        prompt=OptionPromptSpec(free_prompt="Project name (optional): "),
    )
