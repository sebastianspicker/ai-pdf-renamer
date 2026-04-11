from __future__ import annotations

from pathlib import Path

from ai_pdf_renamer.cache import ResponseCache


def test_response_cache_persists_to_disk(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"
    cache = ResponseCache(cache_dir=cache_dir)
    cache.set("analysis:test-key", '{"summary":"cached"}')

    reloaded = ResponseCache(cache_dir=cache_dir)
    assert reloaded.get("analysis:test-key") == '{"summary":"cached"}'


def test_response_cache_file_key_uses_prefix_and_size(tmp_path: Path) -> None:
    pdf_path = tmp_path / "doc.pdf"
    pdf_path.write_bytes(b"A" * 128 + b"B" * 128)

    key_before = ResponseCache.build_file_key(pdf_path, prefix_bytes=64)

    pdf_path.write_bytes(b"C" + b"A" * 127 + b"B" * 128)
    key_after = ResponseCache.build_file_key(pdf_path, prefix_bytes=64)

    assert key_before != key_after

    pdf_path.write_bytes(b"C" + b"A" * 127 + b"B" * 129)
    key_after_size = ResponseCache.build_file_key(pdf_path, prefix_bytes=64)

    assert key_before != key_after_size
    assert key_after != key_after_size
