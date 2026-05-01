from __future__ import annotations

import os
import stat
from pathlib import Path

import pytest

from ai_pdf_renamer.cache import ResponseCache


def _permission_bits(path: Path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


def test_response_cache_persists_to_disk(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"
    cache = ResponseCache(cache_dir=cache_dir)
    cache.set("analysis:test-key", '{"summary":"cached"}')

    reloaded = ResponseCache(cache_dir=cache_dir)
    assert reloaded.get("analysis:test-key") == '{"summary":"cached"}'


def test_response_cache_persistent_paths_are_owner_only_on_posix(tmp_path: Path) -> None:
    if os.name != "posix":
        pytest.skip("POSIX permission bits are not portable on this platform")

    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    cache_dir.chmod(0o755)
    existing_cache_file = cache_dir / "existing-key.json"
    existing_cache_file.write_text("{}", encoding="utf-8")
    existing_cache_file.chmod(0o644)

    cache = ResponseCache(cache_dir=cache_dir)
    cache.set("existing-key", '{"summary":"document-derived"}')
    cache.set("new-key", '{"keywords":["private"]}')

    assert _permission_bits(cache_dir) == 0o700
    assert _permission_bits(existing_cache_file) == 0o600
    assert _permission_bits(cache_dir / "new-key.json") == 0o600


def test_response_cache_file_key_changes_when_tail_changes_with_same_size(tmp_path: Path) -> None:
    pdf_path = tmp_path / "doc.pdf"
    pdf_path.write_bytes(b"A" * 128 + b"B" * 128)

    key_before = ResponseCache.build_file_key(pdf_path, prefix_bytes=64)

    pdf_path.write_bytes(b"A" * 128 + b"C" * 128)
    key_after_tail_change = ResponseCache.build_file_key(pdf_path, prefix_bytes=64)

    assert key_before != key_after_tail_change

    pdf_path.write_bytes(b"C" + b"A" * 127 + b"C" * 128)
    key_after_prefix_change = ResponseCache.build_file_key(pdf_path, prefix_bytes=64)

    assert key_before != key_after_prefix_change
    assert key_after_tail_change != key_after_prefix_change

    pdf_path.write_bytes(b"A" * 128 + b"C" * 129)
    key_after_size = ResponseCache.build_file_key(pdf_path, prefix_bytes=64)

    assert key_before != key_after_size
    assert key_after_tail_change != key_after_size


def test_response_cache_file_key_is_stable_for_unchanged_file(tmp_path: Path) -> None:
    pdf_path = tmp_path / "doc.pdf"
    pdf_path.write_bytes(b"A" * 128 + b"B" * 128)

    key_before = ResponseCache.build_file_key(pdf_path, prefix_bytes=64)
    key_after = ResponseCache.build_file_key(pdf_path, prefix_bytes=64)

    assert key_before == key_after
