from __future__ import annotations

import pytest


def test_cli_rejects_empty_dir(monkeypatch) -> None:
    import ai_pdf_renamer.cli as cli

    monkeypatch.setattr(cli, "setup_logging", lambda **k: None)

    with pytest.raises(SystemExit) as excinfo:
        cli.main(
            [
                "--dir",
                "",
                "--language",
                "de",
                "--case",
                "kebabCase",
                "--project",
                "",
                "--version",
                "",
            ]
        )

    assert "non-empty" in str(excinfo.value).lower()


def test_cli_exits_on_missing_directory(monkeypatch, tmp_path) -> None:
    import ai_pdf_renamer.cli as cli

    monkeypatch.setattr(cli, "setup_logging", lambda **k: None)

    missing = tmp_path / "missing"
    with pytest.raises(SystemExit) as excinfo:
        cli.main(
            [
                "--dir",
                str(missing),
                "--language",
                "de",
                "--case",
                "kebabCase",
                "--project",
                "",
                "--version",
                "",
            ]
        )

    assert "Directory does not exist" in str(excinfo.value)


def test_cli_reprompts_on_invalid_choices(monkeypatch, tmp_path) -> None:
    import builtins

    import ai_pdf_renamer.cli as cli

    monkeypatch.setattr(cli, "setup_logging", lambda **k: None)
    monkeypatch.setattr(cli, "_is_interactive", lambda: True)

    inputs = iter(["fr", "en", "badcase", "snakecase"])
    monkeypatch.setattr(builtins, "input", lambda _prompt: next(inputs))

    captured: dict[str, object] = {}

    def _fake_rename(directory, *, config, files_override=None):
        captured["directory"] = directory
        captured["config"] = config

    monkeypatch.setattr(cli, "rename_pdfs_in_directory", _fake_rename)

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
    assert config.language == "en"
    assert config.desired_case == "snakeCase"


def test_cli_exits_on_invalid_json_in_data_file(monkeypatch, tmp_path) -> None:
    import json

    import ai_pdf_renamer.cli as cli

    monkeypatch.setattr(cli, "setup_logging", lambda **k: None)

    def _raise_json_error(*args, **kwargs):
        raise json.JSONDecodeError("Expecting value", doc="", pos=0)

    monkeypatch.setattr(cli, "rename_pdfs_in_directory", _raise_json_error)

    with pytest.raises(SystemExit) as excinfo:
        cli.main(
            [
                "--dir",
                str(tmp_path),
                "--language",
                "de",
                "--case",
                "kebabCase",
                "--project",
                "",
                "--version",
                "",
            ]
        )

    assert "Invalid JSON" in str(excinfo.value) or "JSON" in str(excinfo.value)


def test_cli_exits_on_missing_data_file(monkeypatch, tmp_path) -> None:
    """When a required data file is missing, CLI exits with a clear message."""
    import ai_pdf_renamer.cli as cli

    monkeypatch.setattr(cli, "setup_logging", lambda **k: None)

    def _raise_file_not_found(*args, **kwargs):
        raise FileNotFoundError(
            "Data file 'heuristic_scores.json' not found. "
            "Looked in: /nonexistent and /packaged. "
            "Set AI_PDF_RENAMER_DATA_DIR or run from the project root."
        )

    monkeypatch.setattr(cli, "rename_pdfs_in_directory", _raise_file_not_found)

    with pytest.raises(SystemExit) as excinfo:
        cli.main(
            [
                "--dir",
                str(tmp_path),
                "--language",
                "de",
                "--case",
                "kebabCase",
                "--project",
                "",
                "--version",
                "",
            ]
        )

    msg = str(excinfo.value)
    assert "not found" in msg or "Data file" in msg


def test_cli_exits_on_broken_json_in_data_file_integration(monkeypatch, tmp_path) -> None:
    """Broken JSON in data dir + PDF with content: CLI exits with clear message."""
    import ai_pdf_renamer.cli as cli
    import ai_pdf_renamer.renamer as renamer

    monkeypatch.setattr(cli, "setup_logging", lambda **k: None)
    monkeypatch.setenv("AI_PDF_RENAMER_DATA_DIR", str(tmp_path))

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
    except Exception:
        pytest.skip("PyMuPDF (fitz) required for integration test")

    renamer._stopwords_cached.cache_clear()
    renamer._heuristic_scorer_cached.cache_clear()

    with pytest.raises(SystemExit) as excinfo:
        cli.main(
            [
                "--dir",
                str(pdf_dir),
                "--language",
                "de",
                "--case",
                "kebabCase",
                "--project",
                "",
                "--version",
                "",
            ]
        )

    msg = str(excinfo.value)
    assert "Invalid JSON" in msg or "heuristic_scores" in msg or "JSON" in msg


def test_cli_non_interactive_uses_defaults_without_hanging(monkeypatch, tmp_path) -> None:
    """With no TTY, CLI uses defaults and does not prompt (suitable for CI/cron)."""
    import ai_pdf_renamer.cli as cli

    monkeypatch.setattr(cli, "setup_logging", lambda **k: None)
    monkeypatch.setattr(cli, "_is_interactive", lambda: False)

    captured: dict[str, object] = {}

    def _fake_rename(directory, *, config, files_override=None):
        captured["directory"] = directory
        captured["config"] = config

    monkeypatch.setattr(cli, "rename_pdfs_in_directory", _fake_rename)

    # Omit --language, --case, --project, --version; only --dir.
    cli.main(["--dir", str(tmp_path)])

    assert captured.get("directory") == str(tmp_path)
    config = captured["config"]
    assert config.language == "de"
    assert config.desired_case == "kebabCase"
    assert config.project == ""
    assert config.version == ""


def test_cli_renames_with_mocked_llm_no_network(monkeypatch, tmp_path) -> None:
    """Rename flow without real LLM when get_document_* are mocked (CI-safe)."""
    import ai_pdf_renamer.cli as cli
    import ai_pdf_renamer.renamer as renamer

    monkeypatch.setattr(cli, "setup_logging", lambda **k: None)
    monkeypatch.setattr(cli, "_is_interactive", lambda: False)

    import ai_pdf_renamer.filename as filename_mod

    monkeypatch.setattr(
        filename_mod,
        "get_document_summary",
        lambda *args, **kwargs: "Mocked summary for testing",
    )
    monkeypatch.setattr(
        filename_mod,
        "get_document_keywords",
        lambda *args, **kwargs: ["mock", "keywords"],
    )
    monkeypatch.setattr(
        filename_mod,
        "get_document_category",
        lambda *args, **kwargs: "document",
    )
    monkeypatch.setattr(
        filename_mod,
        "get_final_summary_tokens",
        lambda *args, **kwargs: ["mocked", "tokens"],
    )

    try:
        import fitz

        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((72, 72), "Sample text so PDF is not empty.")
        doc.save(str(tmp_path / "sample.pdf"))
        doc.close()
    except Exception:
        pytest.skip("PyMuPDF (fitz) required for this test")

    cli.main(
        [
            "--dir",
            str(tmp_path),
            "--language",
            "de",
            "--case",
            "kebabCase",
            "--project",
            "",
            "--version",
            "",
        ]
    )

    pdfs = list(tmp_path.glob("*.pdf"))
    assert len(pdfs) == 1
    # Renamed file should no longer be named sample.pdf (content-based name)
    assert pdfs[0].name != "sample.pdf" or "mock" in pdfs[0].name.lower()
