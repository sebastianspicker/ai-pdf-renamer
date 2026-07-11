from __future__ import annotations

import argparse
import re
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

from ai_pdf_renamer.config import RenamerConfig

if TYPE_CHECKING:
    from unittest.mock import MagicMock

    from ai_pdf_renamer.heuristics import HeuristicScorer
    from ai_pdf_renamer.text_utils import Stopwords
    from ai_pdf_renamer.tui import AIRenamerTUI


REFERENCE_TODAY = date(2026, 4, 8)


def make_config(**overrides: Any) -> RenamerConfig:
    """Build a RenamerConfig with sensible test defaults and optional overrides."""
    defaults: dict[str, Any] = {
        "use_llm": False,
        "use_single_llm_call": False,
        "dry_run": False,
    }
    defaults.update(overrides)
    return RenamerConfig(**defaults)


def make_fake_pdf(tmp_path: Path, name: str = "test.pdf", mtime: float | None = None) -> Path:
    """Create a minimal PDF in tmp_path and optionally set its mtime."""
    pdf = tmp_path / name
    pdf.write_bytes(b"%PDF-1.0\n1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n%%EOF\n")
    if mtime is not None:
        import os

        os.utime(pdf, (mtime, mtime))
    return pdf


def write_dummy_pdf(path: Path, payload: bytes = b"%PDF-1.4 dummy") -> Path:
    """Write a minimal dummy PDF payload and return the path."""
    path.write_bytes(payload)
    return path


def make_pdf_to_text_sequence(*responses: str) -> Callable[..., str]:
    """Build a pdf_to_text stub that returns responses in order, then repeats the last one."""
    pending = iter(responses)
    fallback = responses[-1] if responses else ""

    def fake_pdf_to_text(*_args: Any, **_kwargs: Any) -> str:
        return next(pending, fallback)

    return fake_pdf_to_text


def make_renamer_output_config(**overrides: object) -> MagicMock:
    """Build a MagicMock that behaves like RenamerConfig output settings."""
    from unittest.mock import MagicMock

    cfg = MagicMock()
    cfg.export_metadata_path = overrides.get("export_metadata_path")
    cfg.plan_file_path = overrides.get("plan_file_path")
    cfg.summary_json_path = overrides.get("summary_json_path")
    cfg.dry_run = overrides.get("dry_run", False)
    return cfg


def make_summary_data(
    directory: Path,
    **overrides: object,
) -> object:
    """Build summary data for rename summary writer tests."""
    from ai_pdf_renamer.renamer_output import RenameSummaryData

    values: dict[str, object] = {
        "directory": directory,
        "processed": 0,
        "renamed": 0,
        "skipped": 0,
        "failed": 0,
        "dry_run": True,
        "failures": [],
    }
    values.update(overrides)
    return RenameSummaryData(
        directory=values["directory"],
        processed=values["processed"],
        renamed=values["renamed"],
        skipped=values["skipped"],
        failed=values["failed"],
        dry_run=values["dry_run"],
        failures=values["failures"],
    )


def make_tui_app(settings: dict[str, object] | None = None) -> AIRenamerTUI:
    """Create an AIRenamerTUI with patched CSS and optional pre-loaded settings."""
    from unittest.mock import patch

    from ai_pdf_renamer.tui import AIRenamerTUI

    with patch("ai_pdf_renamer.tui._load_settings", return_value=settings or {}):
        app = AIRenamerTUI()
    app.CSS = AIRenamerTUI.CSS.replace("flex-wrap: wrap;", "")  # type: ignore[assignment]
    return app


def make_cli_namespace(**overrides: Any) -> argparse.Namespace:
    """Create an argparse.Namespace with sensible defaults for CLI tests."""
    defaults: dict[str, Any] = {
        "dirs": None,
        "dirs_from_file": None,
        "single_file": None,
        "manual_file": None,
        "doctor": False,
        "watch": False,
        "watch_interval": 60,
        "language": "de",
        "desired_case": "kebabCase",
        "project": "",
        "version": "",
        "config": None,
    }
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def make_cli_main_args(*dirs: object, watch: bool = False) -> list[str]:
    """Build the common CLI main argument list used by failure-path tests."""
    args = ["--watch"] if watch else []
    args.extend(
        [
            "--dir",
            *(str(path) for path in dirs),
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
    return args


def make_heuristic_scorer(
    categories: list[tuple[str, str, float]] | None = None,
) -> HeuristicScorer:
    """Build a HeuristicScorer from (regex, category, score) triples."""
    from ai_pdf_renamer.heuristics import HeuristicRule, HeuristicScorer

    if categories is None:
        categories = [
            (r"invoice", "invoice", 10.0),
            (r"contract", "contract", 5.0),
        ]
    rules = [
        HeuristicRule(pattern=re.compile(regex, re.IGNORECASE), category=category, score=score)
        for regex, category, score in categories
    ]
    return HeuristicScorer(rules=rules)


def make_llm_client() -> MagicMock:
    """Return a MagicMock that satisfies LLMClient protocol."""
    from unittest.mock import MagicMock

    client = MagicMock()
    client.model = "test-model"
    client.base_url = "http://localhost:8080"
    client.complete.return_value = '{"summary": "test"}'
    client.complete_vision.return_value = '{"summary": "test"}'
    return client


def empty_stopwords() -> Stopwords:
    """Return an empty Stopwords instance for filename tests."""
    from ai_pdf_renamer.text_utils import Stopwords

    return Stopwords(words=set())


def rename_pdf(src: Path, base: str, **overrides: Any) -> tuple[bool, Path]:
    """Call apply_single_rename with common test defaults."""
    from ai_pdf_renamer.rename_ops import apply_single_rename

    defaults: dict[str, Any] = {
        "plan_file_path": None,
        "plan_entries": [],
        "dry_run": False,
        "backup_dir": None,
        "on_success": None,
        "max_filename_chars": None,
    }
    defaults.update(overrides)
    return apply_single_rename(src, base, **defaults)


def make_fitz_doc(
    page: object | None = None,
    *,
    page_count: int = 1,
    is_encrypted: bool = False,
    item_access: bool = False,
    close: bool = False,
) -> MagicMock:
    """Build a minimal fitz-like document mock for PDF extraction tests."""
    from unittest.mock import MagicMock

    mock_doc = MagicMock()
    mock_doc.is_encrypted = is_encrypted
    mock_doc.page_count = page_count
    if page is not None:
        mock_doc.load_page.return_value = page
        if item_access:
            mock_doc.__getitem__ = MagicMock(return_value=page)
    if close:
        mock_doc.close = MagicMock()
    return mock_doc


def install_fitz_mock(monkeypatch: Any, mock_doc: object) -> MagicMock:
    """Install a fitz module mock that opens to the supplied document."""
    import sys
    from unittest.mock import MagicMock

    mock_fitz = MagicMock()
    mock_fitz.open.return_value = mock_doc
    monkeypatch.setitem(sys.modules, "fitz", mock_fitz)
    return mock_fitz


@contextmanager
def patch_pdf_metadata_save_context(
    metadata_tmp_pdf: Path,
) -> Iterator[tuple[MagicMock, MagicMock, MagicMock, MagicMock]]:
    """Patch fitz/tempfile/os calls used by _write_pdf_title_metadata tests."""
    import sys
    from unittest.mock import MagicMock, patch

    mock_doc = MagicMock()
    mock_fitz = MagicMock()
    mock_fitz.open.return_value = mock_doc
    mock_tempfile = MagicMock()
    mock_tempfile.mkstemp.return_value = (99, str(metadata_tmp_pdf))

    with (
        patch.dict(sys.modules, {"fitz": mock_fitz, "tempfile": mock_tempfile}),
        patch("ai_pdf_renamer.renamer.os.close") as mock_os_close,
        patch("ai_pdf_renamer.renamer.os.replace") as mock_os_replace,
    ):
        yield mock_doc, mock_fitz, mock_os_close, mock_os_replace


def make_hook_paths(tmp_path: Path) -> tuple[Path, Path]:
    """Create old/new PDF paths for post-rename hook tests."""
    old = tmp_path / "old.pdf"
    new = tmp_path / "new.pdf"
    old.touch()
    new.touch()
    return old, new


def make_http_hook_session(*, post_side_effect: BaseException | None = None) -> MagicMock:
    """Build a context-manager requests.Session mock for HTTP hook tests."""
    from unittest.mock import MagicMock

    mock_session = MagicMock()
    mock_session.__enter__ = MagicMock(return_value=mock_session)
    mock_session.__exit__ = MagicMock(return_value=False)
    if post_side_effect is None:
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_session.post.return_value = mock_response
    else:
        mock_session.post.side_effect = post_side_effect
    return mock_session


def patch_renamer_process_result(
    monkeypatch: Any,
    renamer_module: object,
    new_base: object,
    *,
    meta: dict[str, object] | None = None,
    error: BaseException | None = None,
) -> None:
    """Patch renamer._process_content_to_result for pipeline tests."""

    def fake_process(
        file_path: Path,
        content: str,
        config: object,
        rules: object | None = None,
        used_vision: bool = False,
    ) -> tuple[Path, str | None, dict[str, object] | None, BaseException | None]:
        base = new_base(file_path) if callable(new_base) else new_base
        return (file_path, base, meta, error)  # type: ignore[return-value]

    monkeypatch.setattr(renamer_module, "_process_content_to_result", fake_process)


@pytest.fixture
def default_config() -> RenamerConfig:
    """A default RenamerConfig with LLM disabled for fast tests."""
    return make_config()


@pytest.fixture
def tmp_pdf(tmp_path: Path) -> Path:
    """Create a minimal syntactically valid PDF file in a temp directory."""
    pdf = tmp_path / "test.pdf"
    # Minimal valid PDF with proper xref table and trailer
    pdf.write_bytes(
        b"%PDF-1.4\n"
        b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
        b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n"
        b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] >>\nendobj\n"
        b"xref\n0 4\n"
        b"0000000000 65535 f \n"
        b"0000000009 00000 n \n"
        b"0000000058 00000 n \n"
        b"0000000115 00000 n \n"
        b"trailer\n<< /Size 4 /Root 1 0 R >>\n"
        b"startxref\n190\n%%EOF\n"
    )
    return pdf
