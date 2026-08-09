"""Tests for renamer.py pipeline helper functions.

Covers _sanitize_csv_cell, _write_json_or_csv, _write_summary_json,
and _write_rename_outputs without requiring full pipeline orchestration.
"""

from __future__ import annotations

import contextlib
import json
import signal
from concurrent.futures import Future
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import pytest

import folionym.renamer as renamer
import folionym.renamer_watch as renamer_watch
from tests.conftest import make_config as _cfg
from tests.conftest import write_dummy_pdf as _write_dummy_pdf
from tests.helpers import patch_renamer_process_result


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

    monkeypatch.setattr(renamer, "_collect_pdf_files", lambda *args, **kwargs: next(scan_iter))
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


def _apply_with_success_callback(file_path: Path, base: str, options: Any) -> tuple[bool, Path]:
    target = file_path.with_name(base + file_path.suffix)
    on_success = options.on_success
    if on_success is not None:
        on_success(file_path, target, base)
    return (True, target)


def _run_progress_scenario(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, workers: int) -> list[tuple[int, int, str]]:
    """Run two deterministic results and record each progress callback."""
    files = [_write_dummy_pdf(tmp_path / f"{name}.pdf", f"%PDF-1.4 {name}".encode()) for name in ("a", "b")]
    monkeypatch.setattr(renamer, "_extract_pdf_content", lambda _path, _config: ("text", False))
    patch_renamer_process_result(monkeypatch, renamer, lambda file_path: file_path.stem, meta={"category": "invoice"})
    seen: list[tuple[int, int, str]] = []
    renamer.produce_rename_results(
        files,
        _cfg(workers=workers),
        progress_callback=lambda current, total, file_path: seen.append((current, total, file_path.name)),
    )
    return seen


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
            "_generate_filename",
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

        monkeypatch.setattr(renamer, "_generate_filename", _raise_value_error)
        monkeypatch.setattr(renamer, "get_pdf_metadata", lambda path: None)
        monkeypatch.setattr(renamer, "_extract_pdf_content", lambda path, config: ("content", False))

        new_base, meta, exc = renamer.suggest_rename_for_file(pdf, cfg)

        assert new_base is None
        assert meta is None
        assert isinstance(exc, ValueError)
        assert "bad content format" in str(exc)


class TestWatchLoop:
    def test_watch_loop_recovers_once_then_stops_without_extra_sleep(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """A recoverable scan failure keeps state, pauses once, and respects a later stop signal."""
        logger = Mock()
        stop_handlers: list[Any] = []
        restored_handlers: list[tuple[bool, Any, Any]] = []
        states: list[Any] = []
        sleep_calls: list[float] = []
        recoverable_error = OSError("temporary scan failure")

        def install_handler(handler: Any) -> tuple[bool, Any, Any]:
            stop_handlers.append(handler)
            return (True, signal.SIG_DFL, signal.SIG_IGN)

        def restore_handlers(is_main_thread: bool, sigterm: Any, sigint: Any) -> None:
            restored_handlers.append((is_main_thread, sigterm, sigint))

        def fail_then_stop(*args: Any, **kwargs: Any) -> renamer_watch.WatchLoopState:
            state = args[2]
            states.append(state)
            if len(states) == 1:
                raise recoverable_error
            stop_handlers[0](signal.SIGTERM, None)
            return state

        deps = renamer_watch.WatchLoopDependencies(
            collect_pdf_files_fn=lambda *args, **kwargs: [],
            load_processing_rules_fn=lambda *args, **kwargs: None,
            rename_pdfs_in_directory_fn=lambda *args, **kwargs: set(),
            logger=logger,
        )
        monkeypatch.setattr(renamer_watch, "_install_watch_signal_handlers", install_handler)
        monkeypatch.setattr(renamer_watch, "_restore_watch_signal_handlers", restore_handlers)
        monkeypatch.setattr(renamer_watch, "_run_watch_iteration", fail_then_stop)
        monkeypatch.setattr(renamer_watch.time, "sleep", sleep_calls.append)

        renamer_watch.run_watch_loop_impl(tmp_path, config=_cfg(dry_run=True), interval_seconds=0.5, deps=deps)

        assert states[0] is states[1]
        assert sleep_calls == [0.5]
        logger.exception.assert_called_once_with("Watch iteration failed: %s", recoverable_error)
        assert restored_handlers == [(True, signal.SIG_DFL, signal.SIG_IGN)]

    def test_watch_loop_processes_new_files(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """Watch loop detects a new PDF and calls rename_pdfs_in_directory for it."""
        pdf = tmp_path / "new-doc.pdf"
        pdf.write_bytes(b"%PDF-1.4 dummy")
        cfg = _cfg(dry_run=True)

        rename_calls = _patch_watch_collect_and_rename(monkeypatch, [[pdf]], pdf, tmp_path / "renamed.pdf")

        sleep_count = 0

        def fake_sleep(seconds: float) -> None:
            nonlocal sleep_count
            sleep_count += 1
            if sleep_count >= 1:
                raise KeyboardInterrupt("stop watch loop")

        monkeypatch.setattr("time.sleep", fake_sleep)

        with contextlib.suppress(KeyboardInterrupt):
            renamer.run_watch_loop(tmp_path, config=cfg, interval_seconds=0.01)

        assert rename_calls == [[pdf]]

    def test_watch_loop_skips_unchanged_files(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """Watch loop does NOT re-process a file if mtime has not changed between iterations."""
        pdf = tmp_path / "stable.pdf"
        pdf.write_bytes(b"%PDF-1.4 dummy")
        cfg = _cfg(dry_run=True)

        rename_calls = _patch_watch_collect_and_rename(monkeypatch, [[pdf], [pdf]], pdf, tmp_path / "renamed.pdf")

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
            "_collect_pdf_files",
            lambda _path, options: rules_seen.append(options.rules) or [],
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
        monkeypatch.setattr(renamer, "_collect_pdf_files", lambda *args, **kwargs: [pdf])

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
        seen = _run_progress_scenario(monkeypatch, tmp_path, workers=1)

        assert seen == [(1, 2, "a.pdf"), (2, 2, "b.pdf")]

    def test_produce_rename_results_reports_progress_with_workers(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        seen = _run_progress_scenario(monkeypatch, tmp_path, workers=2)

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
        assert max_pending <= cfg.output.traversal.workers

    def test_rename_pdfs_files_override(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """files_override is passed to _collect_pdf_files and those files are processed."""
        pdf = tmp_path / "override-target.pdf"
        pdf.write_bytes(b"%PDF-1.4 dummy")
        cfg = _cfg(dry_run=True)

        collected_overrides: list[list[Path] | None] = []
        original_collect = renamer._collect_pdf_files

        def spy_collect(*args: Any, **kwargs: Any) -> list[Path]:
            collected_overrides.append(args[1].files_override)
            return original_collect(*args, **kwargs)

        monkeypatch.setattr(renamer, "_collect_pdf_files", spy_collect)
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
            "_collect_pdf_files",
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
