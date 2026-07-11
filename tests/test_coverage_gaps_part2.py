"""Additional coverage-gap tests for PDF extraction, metadata, and config helpers."""

from __future__ import annotations

import base64
from datetime import date
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from ai_pdf_renamer.config import LLMConfig, RenamerConfig, build_config_from_flat_dict
from ai_pdf_renamer.pdf_extract import (
    _ocr_language_code,
    _parse_pdf_date,
    _shrink_to_token_limit,
    _token_count,
    pdf_first_page_to_image_base64,
)
from ai_pdf_renamer.renamer import _write_pdf_title_metadata
from tests.conftest import make_fitz_doc, patch_pdf_metadata_save_context

# ---------------------------------------------------------------------------
# pdf_extract.py tests
# ---------------------------------------------------------------------------


class TestShrinkToTokenLimit:
    def test_shrink_to_token_limit_multi_iteration(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Text needs >1 shrink iteration when density estimate is off."""
        import ai_pdf_renamer.pdf_extract as pmod

        text = "word " * 2000  # 10000 chars

        # First call: initial count (over limit) -> 500 tokens
        # After density jump, text is shortened. We want the fine-tuning loop
        # to iterate multiple times. We fake _token_count so the initial jump
        # leaves the text still over limit, requiring iterations.
        call_count = 0

        def fake_token_count(t: str) -> int:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # Initial check: report way over limit
                return 500
            if call_count == 2:
                # After density jump: still over limit to force loop iterations
                return 120
            if call_count == 3:
                # Second fine-tuning iteration: still slightly over
                return 105
            # Third+ iteration: under limit
            return 95

        monkeypatch.setattr(pmod, "_token_count", fake_token_count)
        result = _shrink_to_token_limit(text, max_tokens=100)
        assert len(result) < len(text)
        # At least 3 calls to _token_count (initial + density jump check + loop iterations)
        assert call_count >= 3


class TestVisionFormatFallbacks:
    def test_vision_tobytes_jpeg_fails_png_fallback(self) -> None:
        """When tobytes(jpeg) raises TypeError, falls back to PNG."""
        mock_pix = MagicMock()
        mock_pix.tobytes = MagicMock(side_effect=[TypeError("jpeg not supported"), b"PNG_DATA"])

        mock_page = MagicMock()
        mock_page.get_pixmap.return_value = mock_pix

        mock_doc = make_fitz_doc(mock_page, close=True)

        with patch("ai_pdf_renamer.pdf_extract.fitz", create=True) as mock_fitz:
            mock_fitz.open.return_value = mock_doc
            with patch.dict("sys.modules", {"fitz": mock_fitz}):
                result = pdf_first_page_to_image_base64("/fake/path.pdf")

        assert result is not None

        assert base64.b64decode(result) == b"PNG_DATA"

    def test_vision_getimagedata_fallback(self) -> None:
        """When tobytes is missing, getImageData is used."""
        mock_pix = MagicMock(spec=[])  # no attributes by default
        mock_pix.getImageData = MagicMock(return_value=b"JPEG_VIA_OLD_API")
        # Remove tobytes and getPNGData
        assert not hasattr(mock_pix, "tobytes")

        mock_page = MagicMock()
        mock_page.get_pixmap.return_value = mock_pix

        mock_doc = make_fitz_doc(mock_page, close=True)

        with patch("ai_pdf_renamer.pdf_extract.fitz", create=True) as mock_fitz:
            mock_fitz.open.return_value = mock_doc
            with patch.dict("sys.modules", {"fitz": mock_fitz}):
                result = pdf_first_page_to_image_base64("/fake/path.pdf")

        assert result is not None

        assert base64.b64decode(result) == b"JPEG_VIA_OLD_API"


class TestOcrLanguageCode:
    def test_ocr_language_code_env_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """AI_PDF_RENAMER_OCR_LANG env var overrides default language mapping."""
        monkeypatch.setenv("AI_PDF_RENAMER_OCR_LANG", "fra")
        assert _ocr_language_code("de") == "fra"
        assert _ocr_language_code("en") == "fra"

    def test_ocr_language_code_default_de(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Default German maps to 'deu'."""
        monkeypatch.delenv("AI_PDF_RENAMER_OCR_LANG", raising=False)
        assert _ocr_language_code("de") == "deu"

    def test_ocr_language_code_en(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """English maps to 'eng'."""
        monkeypatch.delenv("AI_PDF_RENAMER_OCR_LANG", raising=False)
        assert _ocr_language_code("en") == "eng"


class TestParsePdfDate:
    def test_parse_pdf_date_boundary_non_leap_year(self) -> None:
        """'D:20250229' (Feb 29, non-leap year 2025) returns None."""
        result = _parse_pdf_date("D:20250229")
        assert result is None

    def test_parse_pdf_date_valid(self) -> None:
        """Valid PDF date parses correctly."""
        result = _parse_pdf_date("D:20240229")
        assert result == date(2024, 2, 29)

    def test_parse_pdf_date_none_input(self) -> None:
        """None input returns None."""
        assert _parse_pdf_date(None) is None

    def test_parse_pdf_date_no_prefix(self) -> None:
        """String without D: prefix returns None."""
        assert _parse_pdf_date("20250101") is None

    def test_parse_pdf_date_month_13(self) -> None:
        """Invalid month 13 returns None."""
        assert _parse_pdf_date("D:20251301") is None


# --- Merged from test_round_6_targeted.py ---


def test_shrink_to_token_limit_optimized():
    # Large string: 1000 'a ' (2000 chars), approx 500 tokens (fallback 1 token per 4 chars = 500)
    text = "a " * 1000
    max_tokens = 100

    # Verify initial count
    assert _token_count(text) > max_tokens

    shrunk = _shrink_to_token_limit(text, max_tokens=max_tokens)

    assert _token_count(shrunk) <= max_tokens
    assert len(shrunk) < len(text)
    # Jump optimization should ensure it doesn't take many iterations
    # We can't easily count iterations without patching, but we verify result.


def test_write_pdf_metadata_atomic_save(tmp_path: Path) -> None:
    pdf_path = tmp_path / "fake.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 dummy")
    title = "New Title"

    tmp_pdf = tmp_path / "tmp.pdf"
    tmp_pdf.write_bytes(b"%PDF-1.4 saved content")  # Create tmp file so stat works

    with patch_pdf_metadata_save_context(tmp_pdf) as (mock_doc, _mock_fitz, mock_os_close, mock_os_replace):
        _write_pdf_title_metadata(pdf_path, title)

    mock_doc.set_metadata.assert_called_once_with({"title": title})
    # Verify save uses non-incremental mode with encryption kept
    _args, kwargs = mock_doc.save.call_args
    assert kwargs["incremental"] is False
    assert "encryption" in kwargs
    mock_doc.close.assert_called_once()
    mock_os_close.assert_called_once_with(99)
    mock_os_replace.assert_called_once()


# ---------------------------------------------------------------------------
# RenamerConfig — uncovered branches
# ---------------------------------------------------------------------------


class TestRenamerConfigValidation:
    def test_unknown_kwarg_raises_type_error(self) -> None:
        """Unknown flat kwarg raises TypeError with the offending name."""
        with pytest.raises(TypeError, match="unexpected keyword argument"):
            RenamerConfig(nonexistent_field="oops")

    def test_invalid_desired_case_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="Invalid --case"):
            RenamerConfig(desired_case="PascalCase")

    def test_invalid_date_locale_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="Invalid --date-format"):
            RenamerConfig(date_locale="iso")

    def test_invalid_category_display_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="Invalid --category-display"):
            RenamerConfig(category_display="unknown_display")


class TestRenamerConfigMerge:
    def test_flat_kwargs_merge_with_sub_config(self) -> None:
        """Flat kwargs override values in an explicitly provided sub-config."""
        base_llm = LLMConfig(use_llm=False)
        cfg = RenamerConfig(llm=base_llm, use_llm=True)
        assert cfg.use_llm is True

    def test_sub_config_only_no_flat_overrides(self) -> None:
        """Sub-config without flat overrides is used as-is."""
        llm = LLMConfig(use_llm=False)
        cfg = RenamerConfig(llm=llm)
        assert cfg.use_llm is False


class TestRenamerConfigDunder:
    def test_repr_contains_sub_configs(self) -> None:
        cfg = RenamerConfig()
        r = repr(cfg)
        assert "RenamerConfig(" in r
        assert "llm=" in r
        assert "heuristic=" in r
        assert "extraction=" in r
        assert "output=" in r

    def test_eq_same_defaults(self) -> None:
        assert RenamerConfig() == RenamerConfig()

    def test_eq_different_value(self) -> None:
        assert RenamerConfig(use_llm=True) != RenamerConfig(use_llm=False)

    def test_eq_non_renamer_config(self) -> None:
        assert RenamerConfig() != "not-a-config"
        assert RenamerConfig() != 42

    def test_hash_consistent(self) -> None:
        cfg = RenamerConfig()
        assert hash(cfg) == hash(cfg)

    def test_hash_equal_objects_same_hash(self) -> None:
        assert hash(RenamerConfig()) == hash(RenamerConfig())

    def test_setattr_immutable(self) -> None:
        cfg = RenamerConfig()
        with pytest.raises(AttributeError):
            cfg.language = "en"  # type: ignore[misc]

    def test_delattr_immutable(self) -> None:
        cfg = RenamerConfig()
        with pytest.raises(AttributeError):
            del cfg.language  # type: ignore[misc]

    def test_getattr_unknown_raises_attribute_error(self) -> None:
        cfg = RenamerConfig()
        with pytest.raises(AttributeError):
            _ = cfg.this_does_not_exist  # type: ignore[attr-defined]


class TestBuildConfigFromFlatDict:
    def test_known_fields_are_applied(self) -> None:
        cfg = build_config_from_flat_dict({"language": "en", "dry_run": True})
        assert cfg.language == "en"
        assert cfg.dry_run is True

    def test_unknown_fields_are_silently_ignored(self) -> None:
        """Unknown keys are filtered out before building."""
        cfg = build_config_from_flat_dict({"language": "en", "unknown_garbage": "x"})
        assert cfg.language == "en"

    def test_empty_dict_returns_defaults(self) -> None:
        cfg = build_config_from_flat_dict({})
        assert isinstance(cfg, RenamerConfig)
        assert cfg.language == "de"
