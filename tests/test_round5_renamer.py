"""Round 5 coverage tests for renamer.py.

Covers _process_one_file, suggest_rename_for_file, _produce_rename_results,
rename_pdfs_in_directory (stop_event early exit), and _write_summary_json
with failure details.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

import pytest

import ai_pdf_renamer.renamer as renamer
from ai_pdf_renamer.config import RenamerConfig


def _cfg(**overrides: Any) -> RenamerConfig:
    """Build a RenamerConfig with sensible test defaults."""
    defaults: dict[str, Any] = {
        "use_llm": False,
        "use_single_llm_call": False,
    }
    defaults.update(overrides)
    return RenamerConfig(**defaults)


# ---------------------------------------------------------------------------
# _process_one_file
# ---------------------------------------------------------------------------


class TestProcessOneFile:
    def test_process_one_file_success(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """Extraction returns content; processing returns a valid result."""
        pdf = tmp_path / "invoice.pdf"
        pdf.write_bytes(b"%PDF-1.4 dummy")
        cfg = _cfg()

        monkeypatch.setattr(
            renamer,
            "_extract_pdf_content",
            lambda path, config: ("sample text about an invoice", False),
        )
        monkeypatch.setattr(
            renamer,
            "_process_content_to_result",
            lambda file_path, content, config, rules=None, used_vision=False: (
                file_path,
                "20260101-invoice-sample",
                {"category": "invoice"},
                None,
            ),
        )

        result = renamer._process_one_file(pdf, cfg)
        path_out, new_base, meta, exc = result

        assert path_out == pdf
        assert new_base == "20260101-invoice-sample"
        assert meta == {"category": "invoice"}
        assert exc is None

    def test_process_one_file_extraction_error(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """When _extract_pdf_content raises, error is captured in the result tuple."""
        pdf = tmp_path / "broken.pdf"
        pdf.write_bytes(b"%PDF-1.4 dummy")
        cfg = _cfg()

        monkeypatch.setattr(
            renamer,
            "_extract_pdf_content",
            lambda path, config: (_ for _ in ()).throw(RuntimeError("extraction failed")),
        )

        result = renamer._process_one_file(pdf, cfg)
        path_out, new_base, meta, exc = result

        assert path_out == pdf
        assert new_base is None
        assert meta is None
        assert isinstance(exc, RuntimeError)
        assert "extraction failed" in str(exc)

    def test_process_one_file_processing_error(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """Extraction succeeds but _process_content_to_result raises."""
        pdf = tmp_path / "bad.pdf"
        pdf.write_bytes(b"%PDF-1.4 dummy")
        cfg = _cfg()

        monkeypatch.setattr(
            renamer,
            "_extract_pdf_content",
            lambda path, config: ("some valid content here", False),
        )

        def _raise_on_process(*args: Any, **kwargs: Any) -> Any:
            raise ValueError("filename generation blew up")

        monkeypatch.setattr(renamer, "_process_content_to_result", _raise_on_process)

        result = renamer._process_one_file(pdf, cfg)
        path_out, new_base, meta, exc = result

        assert path_out == pdf
        assert new_base is None
        assert meta is None
        assert isinstance(exc, ValueError)
        assert "filename generation blew up" in str(exc)

    def test_process_one_file_empty_content(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """When extracted content is empty/whitespace, returns (path, None, None, None) skip."""
        pdf = tmp_path / "empty.pdf"
        pdf.write_bytes(b"%PDF-1.4 dummy")
        cfg = _cfg()

        monkeypatch.setattr(
            renamer,
            "_extract_pdf_content",
            lambda path, config: ("   ", False),
        )

        result = renamer._process_one_file(pdf, cfg)
        path_out, new_base, meta, exc = result

        assert path_out == pdf
        assert new_base is None
        assert meta is None
        assert exc is None

    def test_process_one_file_stop_requested(self, tmp_path: Path) -> None:
        """When stop_event is set, returns early without extracting."""
        pdf = tmp_path / "doc.pdf"
        pdf.write_bytes(b"%PDF-1.4 dummy")
        stop_event = threading.Event()
        stop_event.set()
        cfg = _cfg(stop_event=stop_event)

        result = renamer._process_one_file(pdf, cfg)
        path_out, new_base, meta, exc = result

        assert path_out == pdf
        assert new_base is None
        assert meta is None
        assert exc is None


# ---------------------------------------------------------------------------
# suggest_rename_for_file
# ---------------------------------------------------------------------------


class TestSuggestRenameForFile:
    def test_suggest_rename_success(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """Full pipeline: extraction + filename generation returns suggestion."""
        pdf = tmp_path / "contract.pdf"
        pdf.write_bytes(b"%PDF-1.4 dummy")
        cfg = _cfg()

        monkeypatch.setattr(
            renamer,
            "_extract_pdf_content",
            lambda path, config: ("Contract for services dated 2025-06-15", False),
        )
        monkeypatch.setattr(
            renamer,
            "_process_content_to_result",
            lambda file_path, content, config, rules=None, used_vision=False: (
                file_path,
                "20250615-contract-services",
                {"category": "contract", "summary": "services contract"},
                None,
            ),
        )
        monkeypatch.setattr(renamer, "load_processing_rules", lambda path: None)

        new_base, meta, exc = renamer.suggest_rename_for_file(pdf, cfg)

        assert new_base == "20250615-contract-services"
        assert meta is not None
        assert meta["category"] == "contract"
        assert exc is None

    def test_suggest_rename_empty_content(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """When extracted content is empty, returns (None, None, None)."""
        pdf = tmp_path / "blank.pdf"
        pdf.write_bytes(b"%PDF-1.4 dummy")
        cfg = _cfg()

        monkeypatch.setattr(
            renamer,
            "_extract_pdf_content",
            lambda path, config: ("", False),
        )
        monkeypatch.setattr(renamer, "load_processing_rules", lambda path: None)

        new_base, meta, exc = renamer.suggest_rename_for_file(pdf, cfg)

        assert new_base is None
        assert meta is None
        assert exc is None

    def test_suggest_rename_error(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """When _extract_pdf_content raises, error is captured."""
        pdf = tmp_path / "missing.pdf"
        pdf.write_bytes(b"%PDF-1.4 dummy")
        cfg = _cfg()

        def _raise(*args: Any, **kwargs: Any) -> Any:
            raise OSError("cannot read PDF")

        monkeypatch.setattr(renamer, "_extract_pdf_content", _raise)
        monkeypatch.setattr(renamer, "load_processing_rules", lambda path: None)

        new_base, meta, exc = renamer.suggest_rename_for_file(pdf, cfg)

        assert new_base is None
        assert meta is None
        assert isinstance(exc, OSError)
        assert "cannot read PDF" in str(exc)

    def test_suggest_rename_process_error(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """When _process_content_to_result returns an error, it propagates via (None, None, exc)."""
        pdf = tmp_path / "doc.pdf"
        pdf.write_bytes(b"%PDF-1.4 dummy")
        cfg = _cfg()

        monkeypatch.setattr(
            renamer,
            "_extract_pdf_content",
            lambda path, config: ("valid content", False),
        )
        inner_exc = RuntimeError("generate failed")
        monkeypatch.setattr(
            renamer,
            "_process_content_to_result",
            lambda file_path, content, config, rules=None, used_vision=False: (
                file_path,
                None,
                None,
                inner_exc,
            ),
        )
        monkeypatch.setattr(renamer, "load_processing_rules", lambda path: None)

        new_base, meta, exc = renamer.suggest_rename_for_file(pdf, cfg)

        assert new_base is None
        assert meta is None
        assert exc is inner_exc


# ---------------------------------------------------------------------------
# _produce_rename_results
# ---------------------------------------------------------------------------


class TestProduceRenameResults:
    def test_produce_results_sequential(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """workers=1: all files processed sequentially via prefetch path."""
        files = []
        for i in range(3):
            p = tmp_path / f"file{i}.pdf"
            p.write_bytes(b"%PDF-1.4 dummy")
            files.append(p)

        cfg = _cfg(workers=1)

        call_log: list[Path] = []

        def fake_extract(path: Path, config: RenamerConfig) -> tuple[str, bool]:
            call_log.append(path)
            return (f"content of {path.name}", False)

        monkeypatch.setattr(renamer, "_extract_pdf_content", fake_extract)
        monkeypatch.setattr(
            renamer,
            "_process_content_to_result",
            lambda file_path, content, config, rules=None, used_vision=False: (
                file_path,
                f"renamed-{file_path.stem}",
                {"category": "test"},
                None,
            ),
        )

        results = renamer._produce_rename_results(files, cfg, rules=None)

        assert len(results) == 3
        for i, (path_out, new_base, meta, exc) in enumerate(results):
            assert path_out == files[i]
            assert new_base == f"renamed-{files[i].stem}"
            assert meta == {"category": "test"}
            assert exc is None

    def test_produce_results_parallel(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """workers=2: results collected from ThreadPoolExecutor."""
        files = []
        for i in range(4):
            p = tmp_path / f"doc{i}.pdf"
            p.write_bytes(b"%PDF-1.4 dummy")
            files.append(p)

        cfg = _cfg(workers=2)

        def fake_process(
            file_path: Path, config: RenamerConfig, rules: Any = None
        ) -> tuple[Path, str | None, dict[str, object] | None, BaseException | None]:
            return (file_path, f"parallel-{file_path.stem}", {"worker": "pool"}, None)

        monkeypatch.setattr(renamer, "_process_one_file", fake_process)

        results = renamer._produce_rename_results(files, cfg, rules=None)

        assert len(results) == 4
        returned_paths = {r[0] for r in results}
        assert returned_paths == set(files)
        for _path_out, new_base, meta, exc in results:
            assert new_base is not None
            assert new_base.startswith("parallel-")
            assert meta == {"worker": "pool"}
            assert exc is None

    def test_produce_results_parallel_with_exception(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """workers=2: when _process_one_file raises inside the future, error is captured."""
        files = []
        for i in range(2):
            p = tmp_path / f"err{i}.pdf"
            p.write_bytes(b"%PDF-1.4 dummy")
            files.append(p)

        cfg = _cfg(workers=2)

        def fake_process(
            file_path: Path, config: RenamerConfig, rules: Any = None
        ) -> tuple[Path, str | None, dict[str, object] | None, BaseException | None]:
            raise RuntimeError(f"worker error for {file_path.name}")

        monkeypatch.setattr(renamer, "_process_one_file", fake_process)

        results = renamer._produce_rename_results(files, cfg, rules=None)

        assert len(results) == 2
        for _path_out, new_base, meta, exc in results:
            assert new_base is None
            assert meta is None
            assert isinstance(exc, RuntimeError)
            assert "worker error" in str(exc)

    def test_produce_results_sequential_extraction_error(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """workers=1: extraction error for one file does not stop other files."""
        files = []
        for i in range(3):
            p = tmp_path / f"mix{i}.pdf"
            p.write_bytes(b"%PDF-1.4 dummy")
            files.append(p)

        cfg = _cfg(workers=1)

        def fake_extract(path: Path, config: RenamerConfig) -> tuple[str, bool]:
            if path.name == "mix1.pdf":
                raise RuntimeError("corrupt PDF")
            return ("good content", False)

        monkeypatch.setattr(renamer, "_extract_pdf_content", fake_extract)
        monkeypatch.setattr(
            renamer,
            "_process_content_to_result",
            lambda file_path, content, config, rules=None, used_vision=False: (
                file_path,
                f"ok-{file_path.stem}",
                {},
                None,
            ),
        )

        results = renamer._produce_rename_results(files, cfg, rules=None)

        assert len(results) == 3
        # mix0 and mix2 succeed, mix1 fails
        assert results[0][1] == "ok-mix0"
        assert results[0][3] is None
        assert results[1][1] is None
        assert isinstance(results[1][3], RuntimeError)
        assert results[2][1] == "ok-mix2"
        assert results[2][3] is None


# ---------------------------------------------------------------------------
# rename_pdfs_in_directory — stop_event early exit
# ---------------------------------------------------------------------------


class TestRenamePdfsStopEvent:
    def test_rename_pdfs_stop_event_skips_processing(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """When stop_event is already set, processing loop exits before any renames."""
        pdf = tmp_path / "doc.pdf"
        pdf.write_bytes(b"%PDF-1.4 dummy")

        stop_event = threading.Event()
        stop_event.set()

        summary_path = tmp_path / "summary.json"
        cfg = _cfg(
            dry_run=True,
            summary_json_path=str(summary_path),
            stop_event=stop_event,
        )

        # _produce_rename_results returns one result, but the apply loop should
        # break immediately because stop_event is set.
        monkeypatch.setattr(
            renamer,
            "_produce_rename_results",
            lambda *a, **k: [(pdf, "new-name", {}, None)],
        )

        renamer.rename_pdfs_in_directory(tmp_path, config=cfg)

        data = json.loads(summary_path.read_text(encoding="utf-8"))
        # Stop event was set, so processed should be 0 (loop breaks immediately).
        assert data["processed"] == 0
        assert data["renamed"] == 0


# ---------------------------------------------------------------------------
# _write_summary_json — with failure details
# ---------------------------------------------------------------------------


class TestWriteSummaryJson:
    def test_write_summary_json_with_failures(self, tmp_path: Path) -> None:
        """Verify JSON structure includes failure_details list."""
        summary_path = tmp_path / "summary.json"
        failures = [
            {"file": "/tmp/bad1.pdf", "error": "corrupt header"},
            {"file": "/tmp/bad2.pdf", "error": "timeout contacting LLM"},
        ]

        renamer._write_summary_json(
            summary_path,
            directory=tmp_path,
            processed=5,
            renamed=2,
            skipped=1,
            failed=2,
            dry_run=False,
            failures=failures,
        )

        data = json.loads(summary_path.read_text(encoding="utf-8"))

        assert data["processed"] == 5
        assert data["renamed"] == 2
        assert data["skipped"] == 1
        assert data["failed"] == 2
        assert data["dry_run"] is False
        assert data["directory"] == str(tmp_path)
        assert len(data["failures"]) == 2
        assert data["failures"][0]["file"] == "/tmp/bad1.pdf"
        assert data["failures"][0]["error"] == "corrupt header"
        assert data["failures"][1]["file"] == "/tmp/bad2.pdf"
        assert data["failures"][1]["error"] == "timeout contacting LLM"

    def test_write_summary_json_none_path(self) -> None:
        """When summary_path is None, no file is written (no-op)."""
        # Should not raise.
        renamer._write_summary_json(
            None,
            directory=Path("/tmp"),
            processed=0,
            renamed=0,
            skipped=0,
            failed=0,
            dry_run=True,
            failures=[],
        )

    def test_write_summary_json_creates_parent_dirs(self, tmp_path: Path) -> None:
        """Parent directories are created if they don't exist."""
        summary_path = tmp_path / "nested" / "deep" / "summary.json"

        renamer._write_summary_json(
            summary_path,
            directory=tmp_path,
            processed=1,
            renamed=1,
            skipped=0,
            failed=0,
            dry_run=True,
            failures=[],
        )

        assert summary_path.exists()
        data = json.loads(summary_path.read_text(encoding="utf-8"))
        assert data["processed"] == 1
        assert data["dry_run"] is True
