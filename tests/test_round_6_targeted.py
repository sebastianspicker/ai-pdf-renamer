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


def test_write_pdf_metadata_atomic_save(tmp_path):
    pdf_path = tmp_path / "fake.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 dummy")
    title = "New Title"

    mock_doc = MagicMock()
    mock_fitz = MagicMock()
    mock_fitz.open.return_value = mock_doc

    tmp_pdf = tmp_path / "tmp.pdf"
    tmp_pdf.write_bytes(b"%PDF-1.4 saved content")  # Create tmp file so stat works

    mock_tempfile = MagicMock()
    mock_tempfile.mkstemp.return_value = (99, str(tmp_pdf))

    with (
        patch.dict("sys.modules", {"fitz": mock_fitz, "tempfile": mock_tempfile}),
        patch("ai_pdf_renamer.renamer.os.close") as mock_os_close,
        patch("ai_pdf_renamer.renamer.os.replace") as mock_os_replace,
    ):
        _write_pdf_title_metadata(pdf_path, title)

    mock_doc.set_metadata.assert_called_once_with({"title": title})
    # Verify save uses non-incremental mode with encryption kept
    _args, kwargs = mock_doc.save.call_args
    assert kwargs["incremental"] is False
    assert "encryption" in kwargs
    mock_doc.close.assert_called_once()
    mock_os_close.assert_called_once_with(99)
    mock_os_replace.assert_called_once()
