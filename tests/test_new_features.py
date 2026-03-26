from __future__ import annotations

import argparse
import json
import re
import threading
from pathlib import Path

import pytest


def test_parser_accepts_post_rename_hook_flag() -> None:
    from ai_pdf_renamer.cli_parser import build_parser

    parser = build_parser()
    args = parser.parse_args(["--dir", ".", "--post-rename-hook", "echo hi"])
    assert args.post_rename_hook == "echo hi"


def test_parser_max_tokens_help_uses_runtime_default() -> None:
    from ai_pdf_renamer.cli_parser import build_parser
    from ai_pdf_renamer.pdf_extract import DEFAULT_MAX_CONTENT_TOKENS

    help_text = build_parser().format_help()
    assert str(DEFAULT_MAX_CONTENT_TOKENS) in help_text


def test_doctor_mode_invokes_preflight(monkeypatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    import ai_pdf_renamer.cli as cli

    monkeypatch.setattr(cli, "setup_logging", lambda **k: None)

    called: dict[str, bool] = {"ran": False}

    def fake_doctor(args) -> int:
        called["ran"] = True
        print("doctor-ok")
        return 0

    monkeypatch.setattr(cli, "run_doctor_checks", fake_doctor)

    with pytest.raises(SystemExit) as excinfo:
        cli.main(["--doctor", "--dir", str(tmp_path)])

    assert excinfo.value.code == 0
    assert called["ran"] is True
    out = capsys.readouterr().out
    assert "doctor-ok" in out


def test_summary_json_written(monkeypatch, tmp_path: Path) -> None:
    import ai_pdf_renamer.renamer as renamer

    pdf_path = tmp_path / "doc.pdf"
    pdf_path.write_bytes(b"x")

    monkeypatch.setattr(renamer, "_produce_rename_results", lambda *a, **k: [(pdf_path, "new-name", {}, None)])
    monkeypatch.setattr(
        renamer,
        "apply_single_rename",
        lambda *a, **k: (True, pdf_path.with_name("new-name.pdf")),
    )

    summary_path = tmp_path / "summary.json"
    cfg = renamer.RenamerConfig(dry_run=True, summary_json_path=summary_path)
    renamer.rename_pdfs_in_directory(tmp_path, config=cfg)

    data = json.loads(summary_path.read_text(encoding="utf-8"))
    assert data["processed"] == 1
    assert data["renamed"] == 1
    assert data["skipped"] == 0
    assert data["failed"] == 0


def test_rules_allowed_categories_are_passed_to_llm(monkeypatch) -> None:
    import ai_pdf_renamer.filename as filename_mod
    from ai_pdf_renamer.config import RenamerConfig
    from ai_pdf_renamer.heuristics import HeuristicRule, HeuristicScorer
    from ai_pdf_renamer.rules import ProcessingRules
    from ai_pdf_renamer.text_utils import Stopwords

    scorer = HeuristicScorer(
        rules=[
            HeuristicRule(
                pattern=re.compile("invoice"),
                category="invoice",
                score=2.0,
            )
        ]
    )

    captured: dict[str, object] = {}

    monkeypatch.setattr(filename_mod, "get_document_summary", lambda *a, **k: "summary")
    monkeypatch.setattr(filename_mod, "get_document_keywords", lambda *a, **k: ["kw"])

    def fake_get_document_category(*args, **kwargs):
        captured["allowed_categories"] = kwargs.get("allowed_categories")
        return "invoice"

    monkeypatch.setattr(filename_mod, "get_document_category", fake_get_document_category)
    monkeypatch.setattr(filename_mod, "get_final_summary_tokens", lambda *a, **k: ["kw"])

    rules = ProcessingRules(
        skip_llm_if_heuristic_category=[],
        force_category_by_pattern=[],
        skip_files_by_pattern=[],
        allowed_categories=["invoice", "receipt"],
    )

    name, _meta = filename_mod.generate_filename(
        "Invoice 2024-01-09",
        config=RenamerConfig(use_llm=True, use_single_llm_call=False),
        llm_client=object(),
        heuristic_scorer=scorer,
        stopwords=Stopwords(words=set()),
        rules=rules,
    )

    assert name
    assert captured["allowed_categories"] == ["invoice", "receipt"]


def test_collect_pdf_files_max_depth_counts_directory_levels(tmp_path: Path) -> None:
    from ai_pdf_renamer.renamer_files import collect_pdf_files

    root = tmp_path
    (root / "root.pdf").write_text("x", encoding="utf-8")
    (root / "level1").mkdir()
    (root / "level1" / "one.pdf").write_text("x", encoding="utf-8")
    (root / "level1" / "level2").mkdir()
    (root / "level1" / "level2" / "two.pdf").write_text("x", encoding="utf-8")

    files_depth_1 = collect_pdf_files(root, recursive=True, max_depth=1)
    rel_1 = sorted(p.relative_to(root).as_posix() for p in files_depth_1)
    assert rel_1 == ["level1/one.pdf", "root.pdf"]

    files_depth_2 = collect_pdf_files(root, recursive=True, max_depth=2)
    rel_2 = sorted(p.relative_to(root).as_posix() for p in files_depth_2)
    assert rel_2 == ["level1/level2/two.pdf", "level1/one.pdf", "root.pdf"]


def test_summary_json_written_for_empty_directory(tmp_path: Path) -> None:
    import ai_pdf_renamer.renamer as renamer

    summary_path = tmp_path / "summary-empty.json"
    cfg = renamer.RenamerConfig(dry_run=True, summary_json_path=summary_path)
    renamer.rename_pdfs_in_directory(tmp_path, config=cfg)

    data = json.loads(summary_path.read_text(encoding="utf-8"))
    assert data["processed"] == 0
    assert data["renamed"] == 0
    assert data["skipped"] == 0
    assert data["failed"] == 0


def test_summary_processed_count_uses_processed_items_not_discovered(monkeypatch, tmp_path: Path) -> None:
    import ai_pdf_renamer.renamer as renamer

    pdf_path = tmp_path / "doc.pdf"
    pdf_path.write_bytes(b"x")

    monkeypatch.setattr(renamer, "_produce_rename_results", lambda *a, **k: [(pdf_path, "new-name", {}, None)])

    stop_event = threading.Event()
    stop_event.set()
    summary_path = tmp_path / "summary-stopped.json"
    cfg = renamer.RenamerConfig(dry_run=True, summary_json_path=summary_path, stop_event=stop_event)
    renamer.rename_pdfs_in_directory(tmp_path, config=cfg)

    data = json.loads(summary_path.read_text(encoding="utf-8"))
    assert data["processed"] == 0
    assert data["renamed"] == 0


def test_parallel_processing_respects_stop_before_submitting(monkeypatch, tmp_path: Path) -> None:
    import ai_pdf_renamer.renamer as renamer

    calls: dict[str, int] = {"count": 0}

    def fake_process(*args, **kwargs):
        calls["count"] += 1
        return (args[0], None, None, None)

    monkeypatch.setattr(renamer, "_process_one_file", fake_process)

    files = []
    for i in range(5):
        p = tmp_path / f"f{i}.pdf"
        p.write_bytes(b"x")
        files.append(p)

    stop_event = threading.Event()
    stop_event.set()
    cfg = renamer.RenamerConfig(workers=4, stop_event=stop_event)
    out = renamer._produce_rename_results(files, cfg, rules=None)

    assert out == []
    assert calls["count"] == 0


def test_post_rename_hook_runs_without_shell(monkeypatch, tmp_path: Path) -> None:
    import ai_pdf_renamer.renamer as renamer

    called: dict[str, object] = {}

    def fake_run(*args, **kwargs):
        called["args"] = args
        called["kwargs"] = kwargs
        return None

    monkeypatch.setattr(renamer.subprocess, "run", fake_run)

    old_path = tmp_path / "old.pdf"
    new_path = tmp_path / "new.pdf"
    renamer._run_post_rename_hook("echo hello", old_path, new_path, {"category": "invoice"})

    assert called["args"][0] == ["echo", "hello"]
    assert called["kwargs"]["shell"] is False


def test_post_rename_hook_shell_features_use_shell_wrapper(monkeypatch, tmp_path: Path) -> None:
    import os

    import ai_pdf_renamer.renamer as renamer

    called: dict[str, object] = {}

    def fake_run(*args, **kwargs):
        called["args"] = args
        called["kwargs"] = kwargs
        return None

    monkeypatch.setattr(renamer.subprocess, "run", fake_run)

    old_path = tmp_path / "old.pdf"
    new_path = tmp_path / "new.pdf"
    renamer._run_post_rename_hook("echo hello | cat", old_path, new_path, {"category": "invoice"})

    argv = called["args"][0]
    if os.name == "nt":
        assert len(argv) >= 3
        assert argv[1] == "/c"
        assert argv[2] == "echo hello | cat"
    else:
        assert len(argv) >= 3
        assert argv[1] == "-c"
        assert argv[2] == "echo hello | cat"
    assert called["kwargs"]["shell"] is False


def test_post_rename_hook_supports_http_endpoint(monkeypatch, tmp_path: Path) -> None:
    import ai_pdf_renamer.renamer as renamer

    calls: dict[str, object] = {}

    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

    class FakeSession:
        def __init__(self) -> None:
            self.trust_env = True

        def __enter__(self) -> FakeSession:
            calls["trust_env_before_post"] = self.trust_env
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            return None

        def post(self, url, json, timeout):
            calls["url"] = url
            calls["payload"] = json
            calls["timeout"] = timeout
            calls["trust_env_at_post"] = self.trust_env
            return FakeResponse()

    monkeypatch.setattr(renamer.requests, "Session", FakeSession)

    old_path = tmp_path / "old.pdf"
    new_path = tmp_path / "new.pdf"
    renamer._run_post_rename_hook("https://example.invalid/hook", old_path, new_path, {"x": 1})

    assert calls["url"] == "https://example.invalid/hook"
    assert calls["payload"]["old_path"] == str(old_path)
    assert calls["payload"]["new_path"] == str(new_path)
    assert calls["trust_env_at_post"] is False


def test_doctor_checks_fail_on_invalid_llm_probe_response(monkeypatch, tmp_path: Path) -> None:
    import ai_pdf_renamer.cli as cli

    # Prepare required data files.
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    for name in ("heuristic_scores.json", "meta_stopwords.json"):
        (data_dir / name).write_text("{}", encoding="utf-8")

    monkeypatch.setattr(cli, "data_path", lambda filename: data_dir / filename)
    monkeypatch.setattr(cli.importlib.util, "find_spec", lambda _name: object())

    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, str]:
            return {"unexpected": "shape"}

    class FakeSession:
        def __init__(self) -> None:
            self.trust_env = True

        def __enter__(self) -> FakeSession:
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            return None

        def post(self, url, json, timeout):
            return FakeResponse()

    monkeypatch.setattr(cli.requests, "Session", FakeSession)

    args = argparse.Namespace(
        llm_base_url="http://127.0.0.1:11434/v1/completions",
        llm_model="qwen3:8b",
        llm_timeout_s=1.0,
        use_llm=True,
    )
    code = cli.run_doctor_checks(args)
    assert code == 1


def test_doctor_checks_fail_on_invalid_data_file_json(monkeypatch, tmp_path: Path) -> None:
    import ai_pdf_renamer.cli as cli

    good = tmp_path / "heuristic_scores.json"
    bad = tmp_path / "meta_stopwords.json"
    good.write_text("{}", encoding="utf-8")
    bad.write_text("{ invalid", encoding="utf-8")

    def fake_data_path(name: str) -> Path:
        return good if name == "heuristic_scores.json" else bad

    monkeypatch.setattr(cli, "data_path", fake_data_path)
    monkeypatch.setattr(cli.importlib.util, "find_spec", lambda _name: None)

    args = argparse.Namespace(use_llm=False)
    code = cli.run_doctor_checks(args)
    assert code == 1


def test_llm_backend_concurrent_calls(monkeypatch) -> None:
    """HttpLLMBackend.complete() works correctly from multiple threads sharing one instance."""
    from ai_pdf_renamer.llm_backend import HttpLLMBackend

    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, object]:
            return {"choices": [{"message": {"content": "ok"}}]}

    results: dict[str, str] = {}

    backend = HttpLLMBackend(base_url="http://example.invalid/v1/completions", model="x", timeout_s=1.0, use_chat=True)
    monkeypatch.setattr(backend._session, "post", lambda *a, **kw: FakeResponse())

    assert backend.complete("ping") == "ok"
    results["main"] = backend.complete("ping")

    def worker() -> None:
        results["thread"] = backend.complete("pong")

    t = threading.Thread(target=worker)
    t.start()
    t.join()

    assert results["main"] == "ok"
    assert results["thread"] == "ok"


def test_build_config_parses_string_boolean_values() -> None:
    from ai_pdf_renamer.config_resolver import build_config

    cfg = build_config(
        {
            "dry_run": "true",
            "use_llm": "false",
            "use_ocr": "0",
            "recursive": "1",
        }
    )
    assert cfg.dry_run is True
    assert cfg.use_llm is False
    assert cfg.use_ocr is False
    assert cfg.recursive is True


def test_load_config_file_non_object_returns_empty_dict(tmp_path: Path) -> None:
    import ai_pdf_renamer.cli as cli

    json_path = tmp_path / "cfg.json"
    json_path.write_text('["not", "an", "object"]', encoding="utf-8")

    loaded = cli._load_config_file(json_path)
    assert loaded == {}
