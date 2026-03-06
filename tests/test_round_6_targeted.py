from pathlib import Path
from unittest.mock import MagicMock, patch

from ai_pdf_renamer.pdf_extract import _shrink_to_token_limit, _token_count
from ai_pdf_renamer.renamer import _write_pdf_title_metadata


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


def test_write_pdf_metadata_fallback():
    pdf_path = Path("fake.pdf")
    title = "New Title"

    with patch("fitz.open") as mock_open:
        mock_doc = MagicMock()
        mock_open.return_value = mock_doc

        # Simulate incremental save failure
        mock_doc.save_incremental.side_effect = Exception("Incremental fail")

        _write_pdf_title_metadata(pdf_path, title)

        # Verify fallback to full save was called
        mock_doc.save_incremental.assert_called_once()
        args, kwargs = mock_doc.save.call_args
        assert args[0] == pdf_path
        assert kwargs["incremental"] is False
        # PDF_ENCRYPT_KEEP is usually 0
        assert "encryption" in kwargs


