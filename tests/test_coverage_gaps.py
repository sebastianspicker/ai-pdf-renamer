"""Tests to fill coverage gaps across multiple modules."""

from __future__ import annotations

import contextlib
import json
import logging
from pathlib import Path

from ai_pdf_renamer.config import RenamerConfig
from ai_pdf_renamer.config_resolver import build_config
from ai_pdf_renamer.logging_utils import StructuredLogFormatter, setup_logging
from ai_pdf_renamer.rename_ops import sanitize_filename_base, sanitize_filename_from_llm

# ---------------------------------------------------------------------------
# StructuredLogFormatter
# ---------------------------------------------------------------------------


class TestStructuredLogFormatter:
    def test_basic_format(self) -> None:
        fmt = StructuredLogFormatter()
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="", lineno=0, msg="hello", args=(), exc_info=None
        )
        result = fmt.format(record)
        data = json.loads(result)
        assert data["level"] == "INFO"
        assert data["message"] == "hello"
        assert "timestamp" in data

    def test_logger_name_included_for_non_root(self) -> None:
        fmt = StructuredLogFormatter()
        record = logging.LogRecord(
            name="myapp.module", level=logging.WARNING, pathname="", lineno=0, msg="warn", args=(), exc_info=None
        )
        result = fmt.format(record)
        data = json.loads(result)
        assert data["logger"] == "myapp.module"

    def test_root_logger_name_excluded(self) -> None:
        fmt = StructuredLogFormatter()
        record = logging.LogRecord(
            name="root", level=logging.INFO, pathname="", lineno=0, msg="msg", args=(), exc_info=None
        )
        result = fmt.format(record)
        data = json.loads(result)
        assert "logger" not in data

    def test_exception_info_included(self) -> None:
        fmt = StructuredLogFormatter()
        try:
            raise ValueError("test error")
        except ValueError:
            import sys

            exc_info = sys.exc_info()
        record = logging.LogRecord(
            name="test", level=logging.ERROR, pathname="", lineno=0, msg="error", args=(), exc_info=exc_info
        )
        result = fmt.format(record)
        data = json.loads(result)
        assert "exception" in data
        assert "ValueError" in data["exception"]

    def test_format_with_args(self) -> None:
        fmt = StructuredLogFormatter()
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="", lineno=0, msg="count: %d", args=(42,), exc_info=None
        )
        result = fmt.format(record)
        data = json.loads(result)
        assert data["message"] == "count: 42"


class TestSetupLogging:
    def test_structured_logging(self, tmp_path: Path, monkeypatch: object) -> None:
        import os

        monkeypatch.setattr(os, "environ", {**os.environ, "AI_PDF_RENAMER_STRUCTURED_LOGS": "1"})  # type: ignore[attr-defined]
        root = logging.getLogger()
        old_handlers = list(root.handlers)
        old_level = root.level
        try:
            for h in list(root.handlers):
                root.removeHandler(h)
                with contextlib.suppress(Exception):
                    h.close()
            setup_logging(log_file=tmp_path / "test.log", level=logging.DEBUG)
            # Should have handlers
            assert len(root.handlers) >= 1
        finally:
            for h in list(root.handlers):
                root.removeHandler(h)
                with contextlib.suppress(Exception):
                    h.close()
            for h in old_handlers:
                root.addHandler(h)
            root.setLevel(old_level)


# ---------------------------------------------------------------------------
# sanitize_filename_base edge cases
# ---------------------------------------------------------------------------


class TestSanitizeFilenameBase:
    def test_empty_string(self) -> None:
        assert sanitize_filename_base("") == "unnamed"

    def test_whitespace_only(self) -> None:
        assert sanitize_filename_base("   ") == "unnamed"

    def test_windows_reserved_con(self) -> None:
        assert sanitize_filename_base("CON") == "CON_"

    def test_windows_reserved_prn(self) -> None:
        assert sanitize_filename_base("PRN") == "PRN_"

    def test_windows_reserved_aux(self) -> None:
        assert sanitize_filename_base("AUX") == "AUX_"

    def test_windows_reserved_nul(self) -> None:
        assert sanitize_filename_base("NUL") == "NUL_"

    def test_windows_reserved_com1(self) -> None:
        assert sanitize_filename_base("COM1") == "COM1_"

    def test_windows_reserved_lpt1(self) -> None:
        assert sanitize_filename_base("LPT1") == "LPT1_"

    def test_windows_reserved_case_insensitive(self) -> None:
        assert sanitize_filename_base("con") == "con_"

    def test_control_chars_stripped(self) -> None:
        result = sanitize_filename_base("hello\x00world")
        assert "\x00" not in result
        assert result == "helloworld"

    def test_path_separators_stripped(self) -> None:
        result = sanitize_filename_base("hello/world")
        assert "/" not in result

    def test_normal_name_unchanged(self) -> None:
        assert sanitize_filename_base("my-document") == "my-document"


# ---------------------------------------------------------------------------
# sanitize_filename_from_llm edge cases
# ---------------------------------------------------------------------------


class TestSanitizeFilenameFromLlm:
    def test_empty_input(self) -> None:
        assert sanitize_filename_from_llm("") == "document"

    def test_none_input(self) -> None:
        assert sanitize_filename_from_llm(None) == "document"  # type: ignore[arg-type]

    def test_non_string_input(self) -> None:
        assert sanitize_filename_from_llm(123) == "document"  # type: ignore[arg-type]

    def test_strips_pdf_extension(self) -> None:
        assert sanitize_filename_from_llm("my-doc.pdf") == "my-doc"
        assert sanitize_filename_from_llm("my-doc.PDF") == "my-doc"

    def test_replaces_special_chars(self) -> None:
        result = sanitize_filename_from_llm('test:file*name?"<>|end')
        assert all(c not in result for c in ':*?"<>|')

    def test_truncates_long_name(self) -> None:
        long_name = "x" * 200
        result = sanitize_filename_from_llm(long_name)
        assert len(result) <= 120

    def test_strips_leading_dots_underscores(self) -> None:
        result = sanitize_filename_from_llm("...__test")
        assert not result.startswith(".")
        assert not result.startswith("_")

    def test_spaces_to_underscores(self) -> None:
        result = sanitize_filename_from_llm("hello world test")
        assert " " not in result
        assert "_" in result


# ---------------------------------------------------------------------------
# build_config (config_resolver)
# ---------------------------------------------------------------------------


class TestBuildConfig:
    def test_empty_dict_returns_defaults(self) -> None:
        config = build_config({})
        assert isinstance(config, RenamerConfig)
        assert config.language == "de"
        assert config.desired_case == "kebabCase"

    def test_language_override(self) -> None:
        config = build_config({"language": "en"})
        assert config.language == "en"

    def test_bool_parsing(self) -> None:
        config = build_config({"dry_run": True})
        assert config.dry_run is True

    def test_none_values_use_defaults(self) -> None:
        config = build_config({"language": None})
        assert config.language == "de"

    def test_preset_apple_silicon(self) -> None:
        config = build_config({"llm_preset": "apple-silicon"})
        assert config.llm_preset == "apple-silicon"

    def test_require_https_flag(self) -> None:
        config = build_config({"require_https": True})
        assert config.require_https is True

    def test_workers_default(self) -> None:
        config = build_config({})
        assert config.workers == 1
