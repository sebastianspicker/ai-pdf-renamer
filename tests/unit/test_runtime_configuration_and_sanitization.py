"""Exercise logging, configuration, heuristic, and filename edge cases."""

from __future__ import annotations

import contextlib
import json
import logging
import re
from pathlib import Path

import pytest

from folionym.config import RenamerConfig
from folionym.config_resolver import (
    _optional_float,
    _positive_int_or_none,
    build_config,
)
from folionym.heuristic_scoring import ScoringOptions
from folionym.heuristics import (
    ConflictResolutionOptions,
    HeuristicRule,
    _score_text,
    load_heuristic_rules,
)
from folionym.heuristics import (
    _combine_resolve_conflict as _resolve_conflict,
)
from folionym.logging_utils import StructuredLogFormatter, setup_logging
from folionym.rename_ops import sanitize_filename_base, sanitize_filename_from_llm


def _combine_resolve_conflict(cat_llm: str, cat_heuristic: str, **values: object) -> str:
    return _resolve_conflict(cat_llm, cat_heuristic, ConflictResolutionOptions(**values))


# ---------------------------------------------------------------------------
# StructuredLogFormatter
# ---------------------------------------------------------------------------


def _value_error_exc_info() -> tuple[type[ValueError], ValueError, object]:
    try:
        raise ValueError("test error")
    except ValueError as exc:
        return type(exc), exc, exc.__traceback__


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
        record = logging.LogRecord(
            name="test",
            level=logging.ERROR,
            pathname="",
            lineno=0,
            msg="error",
            args=(),
            exc_info=_value_error_exc_info(),
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

        monkeypatch.setattr(os, "environ", {**os.environ, "FOLIONYM_STRUCTURED_LOGS": "1"})  # type: ignore[attr-defined]
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
        assert config.output.naming.language == "de"
        assert config.output.naming.desired_case == "kebabCase"

    def test_language_override(self) -> None:
        config = build_config({"language": "en"})
        assert config.output.naming.language == "en"

    def test_bool_parsing(self) -> None:
        config = build_config({"dry_run": True})
        assert config.output.mode.dry_run is True

    def test_none_values_use_defaults(self) -> None:
        config = build_config({"language": None})
        assert config.output.naming.language == "de"

    def test_preset_apple_silicon(self) -> None:
        config = build_config({"llm_preset": "apple-silicon"})
        assert config.llm.backend.llm_preset == "apple-silicon"

    def test_require_https_flag(self) -> None:
        config = build_config({"require_https": True})
        assert config.llm.runtime.require_https is True

    def test_workers_default(self) -> None:
        config = build_config({})
        assert config.output.traversal.workers == 1


class TestOptionalFloat:
    def test_optional_float_invalid(self) -> None:
        """_optional_float('abc') returns None for non-numeric strings."""
        assert _optional_float("abc") is None

    def test_optional_float_none(self) -> None:
        """_optional_float(None) returns None."""
        assert _optional_float(None) is None

    def test_optional_float_empty_string(self) -> None:
        """_optional_float('') returns None."""
        assert _optional_float("") is None

    def test_optional_float_valid(self) -> None:
        """_optional_float('3.14') returns 3.14."""
        assert _optional_float("3.14") == pytest.approx(3.14)


class TestPositiveIntOrNone:
    def test_positive_int_or_none_negative(self) -> None:
        """_positive_int_or_none(-5) returns None for negative values."""
        assert _positive_int_or_none(-5) is None

    def test_positive_int_or_none_zero(self) -> None:
        """_positive_int_or_none(0) returns None for zero."""
        assert _positive_int_or_none(0) is None

    def test_positive_int_or_none_positive(self) -> None:
        """_positive_int_or_none(10) returns 10."""
        assert _positive_int_or_none(10) == 10


class TestBuildConfigPresets:
    def test_build_config_high_confidence_preset(self) -> None:
        """preset='high-confidence-heuristic' sets skip_llm thresholds."""
        cfg = build_config({"preset": "high-confidence-heuristic"}, env={})
        assert cfg.heuristic.skip.skip_llm_category_if_heuristic_score_ge == pytest.approx(0.5)
        assert cfg.heuristic.skip.skip_llm_category_if_heuristic_gap_ge == pytest.approx(0.3)

    def test_build_config_high_confidence_preset_no_overwrite(self) -> None:
        """preset='high-confidence-heuristic' does NOT overwrite user-set values."""
        cfg = build_config(
            {
                "preset": "high-confidence-heuristic",
                "skip_llm_category_if_heuristic_score_ge": 0.9,
                "skip_llm_category_if_heuristic_gap_ge": 0.8,
            },
            env={},
        )
        assert cfg.heuristic.skip.skip_llm_category_if_heuristic_score_ge == pytest.approx(0.9)
        assert cfg.heuristic.skip.skip_llm_category_if_heuristic_gap_ge == pytest.approx(0.8)

    def test_build_config_no_heuristic_override(self) -> None:
        """no_heuristic_override=True clears override scores to None."""
        cfg = build_config({"no_heuristic_override": True}, env={})
        assert cfg.heuristic.skip.heuristic_override_min_score is None
        assert cfg.heuristic.skip.heuristic_override_min_gap is None

    def test_build_config_prefer_heuristic(self) -> None:
        """prefer_heuristic=True overrides prefer_llm_category to False."""
        cfg = build_config({"prefer_heuristic": True}, env={})
        assert cfg.heuristic.category.prefer_llm_category is False

    def test_build_config_prefer_heuristic_false_default(self) -> None:
        """Without prefer_heuristic, prefer_llm_category defaults to True."""
        cfg = build_config({}, env={})
        assert cfg.heuristic.category.prefer_llm_category is True

    def test_build_config_manual_mode(self) -> None:
        """manual_mode=True forces interactive=True."""
        cfg = build_config({"manual_mode": True}, env={})
        assert cfg.output.mode.manual_mode is True
        assert cfg.output.mode.interactive is True

    def test_build_config_scanned_preset(self) -> None:
        """preset='scanned' enables vision fallback and simple naming mode."""
        cfg = build_config({"preset": "scanned"}, env={})
        assert cfg.llm.vision.use_vision_fallback is True
        assert cfg.llm.runtime.simple_naming_mode is True

    def test_build_config_fast_preset(self) -> None:
        cfg = build_config({"preset": "fast"}, env={})
        assert cfg.llm.runtime.use_llm is False
        assert cfg.heuristic.scoring.min_heuristic_score == 0.6
        assert cfg.heuristic.scoring.min_heuristic_score_gap == 0.25

    def test_build_config_accurate_preset(self) -> None:
        cfg = build_config({"preset": "accurate"}, env={})
        assert cfg.llm.runtime.use_llm is True
        assert cfg.llm.runtime.use_single_llm_call is False
        assert cfg.heuristic.scoring.min_heuristic_score == 0.1
        assert cfg.heuristic.scoring.min_heuristic_score_gap == 0.0

    def test_build_config_normalizes_validated_enum_values(self) -> None:
        cfg = build_config({"date_locale": "DMY", "category_display": "Specific"}, env={})
        assert cfg.output.naming.date_locale == "dmy"
        assert cfg.heuristic.category.category_display == "specific"

    def test_build_config_batch_preset_enables_cache_and_workers(self) -> None:
        cfg = build_config({"preset": "batch"}, env={})
        assert cfg.llm.content.use_cache is True
        assert cfg.output.traversal.workers == 4

    def test_build_config_cache_dir_from_env(self) -> None:
        cfg = build_config({}, env={"FOLIONYM_CACHE_DIR": "/tmp/ai-pdf-cache"})
        assert str(cfg.llm.content.cache_dir) == "/tmp/ai-pdf-cache"

    def test_build_config_default_heuristic_override(self) -> None:
        """Without no_heuristic_override, default override scores are set."""
        cfg = build_config({}, env={})
        assert cfg.heuristic.skip.heuristic_override_min_score == pytest.approx(0.55)
        assert cfg.heuristic.skip.heuristic_override_min_gap == pytest.approx(0.3)

    def test_build_config_reports_multiple_validation_errors(self) -> None:
        """Config validation reports multiple invalid enum values together."""
        with pytest.raises(ValueError) as excinfo:
            build_config(
                {
                    "desired_case": "invalid-case",
                    "date_locale": "ymd",
                    "category_display": "invalid-display",
                },
                env={},
            )
        message = str(excinfo.value)
        assert "desired_case" in message
        assert "date_locale" in message
        assert "category_display" in message


# ---------------------------------------------------------------------------
# 2. heuristics.py tests
# ---------------------------------------------------------------------------


class TestLoadRulesEdgeCases:
    def test_load_rules_invalid_regex(self, tmp_path: Path) -> None:
        """Rule with invalid regex pattern fails fast."""
        data = {
            "patterns": [
                {"regex": "[invalid(", "category": "bad", "score": 1.0},
                {"regex": "good", "category": "good", "score": 2.0},
            ]
        }
        rules_file = tmp_path / "rules.json"
        rules_file.write_text(json.dumps(data))
        with pytest.raises(ValueError, match="Invalid regex"):
            load_heuristic_rules(rules_file)

    def test_load_rules_missing_regex_key(self, tmp_path: Path) -> None:
        """Rule entry without 'regex' key is skipped."""
        data = {
            "patterns": [
                {"category": "orphan", "score": 1.0},
                {"regex": "present", "category": "found", "score": 3.0},
            ]
        }
        rules_file = tmp_path / "rules.json"
        rules_file.write_text(json.dumps(data))
        rules = load_heuristic_rules(rules_file)
        assert len(rules) == 1
        assert rules[0].category == "found"

    def test_load_rules_non_numeric_score(self, tmp_path: Path) -> None:
        """Non-numeric score string defaults to 0.0."""
        data = {
            "patterns": [
                {"regex": "test", "category": "cat", "score": "not_a_number"},
            ]
        }
        rules_file = tmp_path / "rules.json"
        rules_file.write_text(json.dumps(data))
        rules = load_heuristic_rules(rules_file)
        assert len(rules) == 1
        assert rules[0].score == pytest.approx(0.0)

    def test_load_rules_none_score(self, tmp_path: Path) -> None:
        """None score defaults to 0.0."""
        data = {
            "patterns": [
                {"regex": "test", "category": "cat", "score": None},
            ]
        }
        rules_file = tmp_path / "rules.json"
        rules_file.write_text(json.dumps(data))
        rules = load_heuristic_rules(rules_file)
        assert len(rules) == 1
        assert rules[0].score == pytest.approx(0.0)


class TestMaxScorePerCategory:
    @pytest.mark.parametrize(
        ("maximum", "expected"),
        [(5.0, 5.0), (None, 8.0)],
        ids=["caps", "does-not-cap"],
    )
    def test_max_score_per_category(self, maximum: float | None, expected: float) -> None:
        """The optional category cap bounds otherwise additive rule scores."""
        rules = [
            HeuristicRule(pattern=re.compile("invoice"), category="invoice", score=4.0),
            HeuristicRule(pattern=re.compile("INVOICE"), category="invoice", score=4.0),
        ]
        scores = _score_text(
            "invoice INVOICE",
            rules,
            None,
            options=ScoringOptions(max_score_per_category=maximum),
        )
        assert scores["invoice"] == pytest.approx(expected)

    def test_negative_regex_skips_false_positive_match(self, tmp_path: Path) -> None:
        """A negative_regex suppresses local false positives such as 'invoice address'."""
        data = {
            "patterns": [
                {
                    "regex": r"(?i)invoice",
                    "negative_regex": r"invoice\s+address",
                    "category": "invoice",
                    "score": 4.0,
                }
            ]
        }
        rules_file = tmp_path / "rules.json"
        rules_file.write_text(json.dumps(data))
        rules = load_heuristic_rules(rules_file)

        false_positive = _score_text("Please update the invoice address for returns.", rules, None)
        valid = _score_text("Invoice INV-2025-001 total due.", rules, None)

        assert false_positive == {}
        assert valid["invoice"] == pytest.approx(4.0)


class TestCombineResolveConflictKeywordOverlap:
    def test_keyword_overlap_llm_wins(self) -> None:
        """use_keyword_overlap=True with LLM category tokens overlapping more with context."""
        # context has "insurance" which overlaps with LLM category "insurance"
        result = _combine_resolve_conflict(
            "insurance",
            "invoice",
            prefer_llm=False,
            context_for_overlap="this is about insurance policy coverage",
            use_keyword_overlap=True,
            heuristic_score=0.1,
            heuristic_score_weight=0.0,
        )
        assert result == "insurance"

    def test_keyword_overlap_heuristic_wins(self) -> None:
        """use_keyword_overlap=True with heuristic category tokens overlapping more."""
        result = _combine_resolve_conflict(
            "insurance",
            "invoice",
            prefer_llm=False,
            context_for_overlap="this is an invoice for payment services",
            use_keyword_overlap=True,
            heuristic_score=0.1,
            heuristic_score_weight=0.0,
        )
        assert result == "invoice"

    def test_keyword_overlap_tie_returns_heuristic(self) -> None:
        """On keyword overlap tie, heuristic wins."""
        result = _combine_resolve_conflict(
            "alpha",
            "beta",
            prefer_llm=True,
            context_for_overlap="neither alpha nor beta appears here gamma delta",
            use_keyword_overlap=True,
            heuristic_score=None,
            heuristic_score_weight=0.0,
        )
        # Both have 0 overlap -> tie -> heuristic wins
        assert result == "beta"

    def test_keyword_overlap_with_score_weight_bonus(self) -> None:
        """Heuristic score weight bonus can tip the overlap in favor of heuristic."""
        # Both categories have 0 overlap with context, but heuristic_score_weight
        # gives heuristic a bonus, so heuristic wins even with equal overlap.
        result = _combine_resolve_conflict(
            "alpha",
            "beta",
            prefer_llm=True,
            context_for_overlap="unrelated context",
            use_keyword_overlap=True,
            heuristic_score=5.0,
            heuristic_score_weight=0.5,
        )
        # overlap_llm=0, overlap_heur_weighted=0+2.5=2.5 -> heuristic wins
        assert result == "beta"
