"""Verify in-memory and private on-disk response-cache behavior."""

from __future__ import annotations

import os
import stat
from pathlib import Path

import pytest

import folionym.cache as cache_mod
from folionym.cache import ResponseCache


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
    existing_key = "a" * 64
    new_key = "b" * 64
    existing_cache_file = cache_dir / f"{existing_key}.json"
    existing_cache_file.write_text("{}", encoding="utf-8")
    existing_cache_file.chmod(0o644)

    cache = ResponseCache(cache_dir=cache_dir)
    cache.set(existing_key, '{"summary":"document-derived"}')
    cache.set(new_key, '{"keywords":["private"]}')

    assert _permission_bits(cache_dir) == 0o700
    assert _permission_bits(existing_cache_file) == 0o600
    assert _permission_bits(cache_dir / f"{new_key}.json") == 0o600


@pytest.mark.parametrize("key", ["../escaped", "/absolute/path", "..\\escaped", "invalid\0key"])
def test_response_cache_maps_arbitrary_keys_to_safe_disk_names(tmp_path: Path, key: str) -> None:
    cache_dir = tmp_path / "cache"
    cache = ResponseCache(cache_dir=cache_dir)

    cache.set(key, '{"summary":"private"}')

    disk_files = list(cache_dir.glob("*.json"))
    assert len(disk_files) == 1
    assert disk_files[0].parent == cache_dir
    assert disk_files[0].stem != key
    assert ResponseCache(cache_dir=cache_dir).get(key) == '{"summary":"private"}'


def test_response_cache_does_not_read_traversal_key_outside_cache_root(tmp_path: Path) -> None:
    outside_file = tmp_path / "injected.json"
    outside_file.write_text('{"value":"attacker-controlled"}', encoding="utf-8")

    cache = ResponseCache(cache_dir=tmp_path / "cache")

    assert cache.get("../injected") is None


def test_response_cache_disables_disk_cache_when_permissions_cannot_be_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache_dir = tmp_path / "cache"
    cache = ResponseCache(cache_dir=cache_dir)

    def deny_chmod(path: Path, mode: int) -> None:
        raise cache_mod.CachePermissionError(path, mode, RuntimeError("chmod unavailable"))

    monkeypatch.setattr(cache_mod, "_set_owner_only_permissions", deny_chmod)

    cache.set("unsafe-key", '{"summary":"private"}')

    assert cache.get("unsafe-key") == '{"summary":"private"}'
    assert cache.cache_dir is None
    assert not (cache_dir / "unsafe-key.json").exists()


def test_response_cache_file_key_changes_when_any_content_changes_with_same_size(tmp_path: Path) -> None:
    pdf_path = tmp_path / "doc.pdf"
    pdf_path.write_bytes(b"A" * 128 + b"B" * 128)

    key_before = ResponseCache.build_file_key(pdf_path)

    pdf_path.write_bytes(b"A" * 128 + b"C" * 128)
    key_after_tail_change = ResponseCache.build_file_key(pdf_path)

    assert key_before != key_after_tail_change

    pdf_path.write_bytes(b"C" + b"A" * 127 + b"C" * 128)
    key_after_prefix_change = ResponseCache.build_file_key(pdf_path)

    assert key_before != key_after_prefix_change
    assert key_after_tail_change != key_after_prefix_change

    pdf_path.write_bytes(b"A" * 128 + b"D" + b"B" * 127)
    key_after_middle_change = ResponseCache.build_file_key(pdf_path)

    assert key_before != key_after_middle_change

    pdf_path.write_bytes(b"A" * 128 + b"C" * 129)
    key_after_size = ResponseCache.build_file_key(pdf_path)

    assert key_before != key_after_size
    assert key_after_tail_change != key_after_size


def test_response_cache_file_key_is_stable_for_unchanged_file(tmp_path: Path) -> None:
    pdf_path = tmp_path / "doc.pdf"
    pdf_path.write_bytes(b"A" * 128 + b"B" * 128)

    key_before = ResponseCache.build_file_key(pdf_path)
    key_after = ResponseCache.build_file_key(pdf_path)

    assert key_before == key_after


def test_response_cache_file_key_rejects_file_changed_during_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pdf_path = tmp_path / "doc.pdf"
    pdf_path.write_bytes(b"A" * 256)
    real_identity = cache_mod._file_identity
    calls = 0

    def changed_identity(status: os.stat_result) -> tuple[int, int, int, int]:
        nonlocal calls
        calls += 1
        identity = real_identity(status)
        if calls == 3:
            return (*identity[:3], identity[3] + 1)
        return identity

    monkeypatch.setattr(cache_mod, "_file_identity", changed_identity)

    with pytest.raises(OSError, match="File changed while building cache key"):
        ResponseCache.build_file_key(pdf_path)
