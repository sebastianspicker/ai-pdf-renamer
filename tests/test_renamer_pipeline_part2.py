"""Tests for renamer.py pipeline helper functions.

Covers _sanitize_csv_cell, _write_json_or_csv, _write_summary_json,
and _write_rename_outputs without requiring full pipeline orchestration.
"""

from __future__ import annotations

import contextlib
import json
from concurrent.futures import Future
from pathlib import Path
from typing import Any

import pytest

import ai_pdf_renamer.renamer as renamer
from tests.conftest import make_config as _cfg
from tests.conftest import patch_renamer_process_result
from tests.conftest import write_dummy_pdf as _write_dummy_pdf


def _stop_after_sleep_cycles(monkeypatch: pytest.MonkeyPatch, cycles: int) -> None:
    sleep_count = 0

    def fake_sleep(seconds: float) -> None:
        nonlocal sleep_count
        sleep_count += 1
        if sleep_count >= cycles:
            raise KeyboardInterrupt(f"stop after {cycles} sleep cycle(s)")

    monkeypatch.setattr("time.sleep", fake_sleep)


def _patch_watch_collect_and_rename(
    monkeypatch: pytest.MonkeyPatch,
    scans: list[list[Path]],
    original: Path,
    renamed_output: Path,
) -> list[list[Path]]:
    scan_iter = iter(scans)
    rename_calls: list[list[Path]] = []

    def fake_rename(
        directory: Any,
        *,
        config: Any,
        files_override: list[Path] | None = None,
        rules_override: Any = None,
    ) -> set[Path]:
        assert files_override is not None
        rename_calls.append(files_override)
        return {renamed_output} if files_override == [original] else set()

    monkeypatch.setattr(renamer, "collect_pdf_files", lambda *args, **kwargs: next(scan_iter))
    monkeypatch.setattr(renamer, "rename_pdfs_in_directory", fake_rename)
    return rename_calls


def _export_metadata_result(pdf: Path) -> list[tuple[Path, str, dict[str, object], None]]:
    return [
        (
            pdf,
            "20260101-exported-doc",
            {
                "category": "report",
                "summary": "quarterly report",
                "keywords": "finance, q1",
                "category_source": "heuristic",
                "llm_failed": False,
                "used_vision_fallback": False,
                "invoice_id": "",
                "amount": "",
                "company": "Acme",
            },
            None,
        )
    ]


def _apply_with_success_callback(
    file_path: Path,
    base: str,
    **kwargs: Any,
) -> tuple[bool, Path]:
    target = file_path.with_name(base + file_path.suffix)
    on_success = kwargs.get("on_success")
    if on_success is not None:
        on_success(file_path, target, base)
    return (True, target)


class TestProcessContentToResult:
    def test_process_content_to_result_success(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """generate_filename returns a valid tuple; result includes used_vision_fallback."""
        pdf = _write_dummy_pdf(tmp_path / "doc.pdf")
        cfg = _cfg()

        monkeypatch.setattr(
            renamer,
            "_extract_pdf_content",
            lambda path, config: ("some invoice content", False),
        )
        monkeypatch.setattr(
            renamer,
            "generate_filename",
            lambda content, request: (
                "20260101-invoice-acme",
                {"category": "invoice"},
            ),
        )
        monkeypatch.setattr(renamer, "get_pdf_metadata", lambda path: None)

        new_base, meta, exc = renamer.suggest_rename_for_file(pdf, cfg)

        assert new_base == "20260101-invoice-acme"
        assert meta is not None
        assert meta["category"] == "invoice"
        assert meta["used_vision_fallback"] is False
        assert exc is None

    def test_process_content_to_result_error(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """When generate_filename raises ValueError, error is captured in result tuple."""
        pdf = tmp_path / "broken.pdf"
        pdf.write_bytes(b"%PDF-1.4 dummy")
        cfg = _cfg()

        def _raise_value_error(*args: Any, **kwargs: Any) -> Any:
            raise ValueError("bad content format")

        monkeypatch.setattr(renamer, "generate_filename", _raise_value_error)
        monkeypatch.setattr(renamer, "get_pdf_metadata", lambda path: None)
        monkeypatch.setattr(renamer, "_extract_pdf_content", lambda path, config: ("content", False))

        new_base, meta, exc = renamer.suggest_rename_for_file(pdf, cfg)

        assert new_base is None
        assert meta is None
        assert isinstance(exc, ValueError)
        assert "bad content format" in str(exc)


class TestWatchLoop:
    def test_watch_loop_processes_new_files(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """Watch loop detects a new PDF and calls rename_pdfs_in_directory for it."""
        pdf = tmp_path / "new-doc.pdf"
        pdf.write_bytes(b"%PDF-1.4 dummy")
        cfg = _cfg(dry_run=True)

        rename_calls: list[dict[str, Any]] = []

        def fake_rename(directory: Any, *, config: Any, files_override: Any = None, rules_override: Any = None) -> None:
            rename_calls.append({"dir": directory, "files": files_override})

        monkeypatch.setattr(renamer, "rename_pdfs_in_directory", fake_rename)
        monkeypatch.setattr(
            renamer,
            "collect_pdf_files",
            lambda *args, **kwargs: [pdf],
        )

        sleep_count = 0

        def fake_sleep(seconds: float) -> None:
            nonlocal sleep_count
            sleep_count += 1
            if sleep_count >= 1:
                raise KeyboardInterrupt("stop watch loop")

        monkeypatch.setattr("time.sleep", fake_sleep)

        with contextlib.suppress(KeyboardInterrupt):
            renamer.run_watch_loop(tmp_path, config=cfg, interval_seconds=0.01)

        assert len(rename_calls) >= 1, "rename_pdfs_in_directory should have been called at least once"
        assert rename_calls[0]["files"] == [pdf]

    def test_watch_loop_skips_unchanged_files(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """Watch loop does NOT re-process a file if mtime has not changed between iterations."""
        pdf = tmp_path / "stable.pdf"
        pdf.write_bytes(b"%PDF-1.4 dummy")
        cfg = _cfg(dry_run=True)

        rename_calls: list[dict[str, Any]] = []

        def fake_rename(directory: Any, *, config: Any, files_override: Any = None, rules_override: Any = None) -> None:
            rename_calls.append({"dir": directory, "files": files_override})

        monkeypatch.setattr(renamer, "rename_pdfs_in_directory", fake_rename)

        # collect_pdf_files returns the same file every iteration
        monkeypatch.setattr(
            renamer,
            "collect_pdf_files",
            lambda *args, **kwargs: [pdf],
        )

        iteration = 0

        def fake_sleep(seconds: float) -> None:
            nonlocal iteration
            iteration += 1
            if iteration >= 2:
                raise KeyboardInterrupt("stop after 2 sleep cycles")

        monkeypatch.setattr("time.sleep", fake_sleep)

        with contextlib.suppress(KeyboardInterrupt):
            renamer.run_watch_loop(tmp_path, config=cfg, interval_seconds=0.01)

        # File is only processed on the first iteration since mtime does not change.
        assert len(rename_calls) == 1, (
            f"Expected exactly 1 rename call (file unchanged on 2nd iteration), got {len(rename_calls)}"
        )

    def test_watch_loop_processes_new_file_seen_during_post_scan(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """A legitimate new PDF discovered after a pass must be processed on the next iteration."""
        original = _write_dummy_pdf(tmp_path / "incoming.pdf", b"%PDF-1.4 original")
        renamed_output = _write_dummy_pdf(tmp_path / "20260101-incoming.pdf", b"%PDF-1.4 renamed")
        newcomer = _write_dummy_pdf(tmp_path / "arrived-later.pdf", b"%PDF-1.4 newcomer")
        cfg = _cfg(dry_run=False)

        rename_calls = _patch_watch_collect_and_rename(
            monkeypatch,
            [
                [original],
                [renamed_output, newcomer],
                [renamed_output, newcomer],
                [renamed_output, newcomer],
            ],
            original,
            renamed_output,
        )
        _stop_after_sleep_cycles(monkeypatch, 2)

        with contextlib.suppress(KeyboardInterrupt):
            renamer.run_watch_loop(tmp_path, config=cfg, interval_seconds=0.01)

        assert rename_calls == [[original], [newcomer]]

    def test_watch_loop_reloads_rules_each_iteration(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        cfg = _cfg(dry_run=True, rules_file=str(tmp_path / "rules.json"))

        rules_seen: list[object] = []
        rule_a = object()
        rule_b = object()
        rule_iter = iter([rule_a, rule_b])

        monkeypatch.setattr(renamer, "load_processing_rules", lambda *args, **kwargs: next(rule_iter))
        monkeypatch.setattr(
            renamer,
            "collect_pdf_files",
            lambda *args, **kwargs: rules_seen.append(kwargs["rules"]) or [],
        )

        sleep_count = 0

        def fake_sleep(seconds: float) -> None:
            nonlocal sleep_count
            sleep_count += 1
            if sleep_count >= 2:
                raise KeyboardInterrupt("stop after second watch cycle")

        monkeypatch.setattr("time.sleep", fake_sleep)

        with contextlib.suppress(KeyboardInterrupt):
            renamer.run_watch_loop(tmp_path, config=cfg, interval_seconds=0.01)

        assert rules_seen == [rule_a, rule_b]

    def test_watch_loop_passes_iteration_rules_snapshot_to_rename(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        pdf = tmp_path / "incoming.pdf"
        pdf.write_bytes(b"%PDF-1.4 dummy")
        cfg = _cfg(dry_run=True, rules_file=str(tmp_path / "rules.json"))
        rule_snapshot = object()
        rename_rules: list[object | None] = []

        monkeypatch.setattr(renamer, "load_processing_rules", lambda *args, **kwargs: rule_snapshot)
        monkeypatch.setattr(renamer, "collect_pdf_files", lambda *args, **kwargs: [pdf])

        def fake_rename(
            directory: Any,
            *,
            config: Any,
            files_override: list[Path] | None = None,
            rules_override: Any = None,
        ) -> set[Path]:
            rename_rules.append(rules_override)
            raise KeyboardInterrupt("stop after first rename")

        monkeypatch.setattr(renamer, "rename_pdfs_in_directory", fake_rename)

        with contextlib.suppress(KeyboardInterrupt):
            renamer.run_watch_loop(tmp_path, config=cfg, interval_seconds=0.01)

        assert rename_rules == [rule_snapshot]


class TestRenamePipelineEdgeCases:
    def test_produce_rename_results_reports_progress(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        pdf_a = tmp_path / "a.pdf"
        pdf_b = tmp_path / "b.pdf"
        pdf_a.write_bytes(b"%PDF-1.4 a")
        pdf_b.write_bytes(b"%PDF-1.4 b")
        cfg = _cfg(workers=1)

        monkeypatch.setattr(renamer, "_extract_pdf_content", lambda path, config: ("text", False))
        patch_renamer_process_result(
            monkeypatch,
            renamer,
            lambda file_path: file_path.stem,
            meta={"category": "invoice"},
        )

        seen: list[tuple[int, int, str]] = []

        renamer.produce_rename_results(
            [pdf_a, pdf_b],
            cfg,
            progress_callback=lambda current, total, file_path: seen.append((current, total, file_path.name)),
        )

        assert seen == [(1, 2, "a.pdf"), (2, 2, "b.pdf")]

    def test_produce_rename_results_reports_progress_with_workers(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        pdf_a = tmp_path / "a.pdf"
        pdf_b = tmp_path / "b.pdf"
        pdf_a.write_bytes(b"%PDF-1.4 a")
        pdf_b.write_bytes(b"%PDF-1.4 b")
        cfg = _cfg(workers=2)

        monkeypatch.setattr(renamer, "_extract_pdf_content", lambda path, config: ("text", False))
        patch_renamer_process_result(
            monkeypatch,
            renamer,
            lambda file_path: file_path.stem,
            meta={"category": "invoice"},
        )

        seen: list[tuple[int, int, str]] = []

        renamer.produce_rename_results(
            [pdf_a, pdf_b],
            cfg,
            progress_callback=lambda current, total, file_path: seen.append((current, total, file_path.name)),
        )

        assert {(total, filename) for _, total, filename in seen} == {(2, "a.pdf"), (2, "b.pdf")}

    def test_produce_results_parallel_keeps_inflight_queue_bounded(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        files = []
        for i in range(6):
            p = tmp_path / f"doc{i}.pdf"
            p.write_bytes(b"%PDF-1.4 dummy")
            files.append(p)

        cfg = _cfg(workers=2)
        pending_results: dict[
            Future[tuple[Path, str | None, dict[str, object] | None, BaseException | None]],
            tuple[Path, str | None, dict[str, object] | None, BaseException | None],
        ] = {}
        max_pending = 0

        class FakeExecutor:
            def __init__(self, max_workers: int) -> None:
                assert max_workers == 2

            def submit(
                self, fn: Any, file_path: Path, config: Any, rules: Any
            ) -> Future[tuple[Path, str | None, dict[str, object] | None, BaseException | None]]:
                nonlocal max_pending
                future: Future[tuple[Path, str | None, dict[str, object] | None, BaseException | None]] = Future()
                pending_results[future] = (file_path, f"parallel-{file_path.stem}", {"worker": "pool"}, None)
                max_pending = max(max_pending, sum(1 for pending in pending_results if not pending.done()))
                return future

            def shutdown(self, wait: bool = True, cancel_futures: bool = False) -> None:
                return None

        def fake_wait(
            futures: set[Future[tuple[Path, str | None, dict[str, object] | None, BaseException | None]]],
            return_when: Any,
        ) -> tuple[
            set[Future[tuple[Path, str | None, dict[str, object] | None, BaseException | None]]],
            set[Future[tuple[Path, str | None, dict[str, object] | None, BaseException | None]]],
        ]:
            future = next(future for future in futures if not future.done())
            future.set_result(pending_results[future])
            return ({future}, futures - {future})

        monkeypatch.setattr(renamer, "ThreadPoolExecutor", FakeExecutor)
        monkeypatch.setattr(renamer, "wait", fake_wait)

        results = renamer.produce_rename_results(files, cfg, rules=None)

        assert len(results) == len(files)
        assert max_pending <= cfg.workers

    def test_rename_pdfs_files_override(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """files_override is passed to _collect_pdf_files and those files are processed."""
        pdf = tmp_path / "override-target.pdf"
        pdf.write_bytes(b"%PDF-1.4 dummy")
        cfg = _cfg(dry_run=True)

        collected_overrides: list[list[Path] | None] = []
        original_collect = renamer.collect_pdf_files

        def spy_collect(*args: Any, **kwargs: Any) -> list[Path]:
            collected_overrides.append(kwargs.get("files_override"))
            return original_collect(*args, **kwargs)

        monkeypatch.setattr(renamer, "collect_pdf_files", spy_collect)
        monkeypatch.setattr(
            renamer,
            "produce_rename_results",
            lambda files, config, rules=None, progress_callback=None: [
                (pdf, "20260101-overridden-file", {"category": "test"}, None)
            ],
        )

        renamer.rename_pdfs_in_directory(
            tmp_path,
            config=cfg,
            files_override=[pdf],
        )

        assert len(collected_overrides) == 1
        assert collected_overrides[0] == [pdf]

    def test_rename_pdfs_with_export_metadata(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """When config.export_metadata_path is set, the export file is written after processing."""
        pdf = _write_dummy_pdf(tmp_path / "exportable.pdf")
        export_path = tmp_path / "export.json"
        cfg = _cfg(dry_run=True, export_metadata_path=str(export_path))

        monkeypatch.setattr(
            renamer,
            "collect_pdf_files",
            lambda *args, **kwargs: [pdf],
        )
        monkeypatch.setattr(
            renamer,
            "produce_rename_results",
            lambda files, config, rules=None, progress_callback=None: _export_metadata_result(pdf),
        )

        # In dry_run mode, apply_single_rename returns True without calling on_success.
        # We mock it to invoke on_success so the export path is exercised.
        monkeypatch.setattr(renamer, "apply_single_rename", _apply_with_success_callback)

        renamer.rename_pdfs_in_directory(tmp_path, config=cfg)

        assert export_path.exists(), "Export metadata file should have been written"
        data = json.loads(export_path.read_text(encoding="utf-8"))
        assert len(data) == 1
        row = data[0]
        assert row["category"] == "report"
        assert row["summary"] == "quarterly report"
        assert row["company"] == "Acme"
        assert row["new_name"] == "20260101-exported-doc.pdf"
