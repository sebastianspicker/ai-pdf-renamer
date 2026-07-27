"""Response caching for repeated LLM work.

Cache entries always live in memory for the current process. When ``cache_dir``
is set, entries are also persisted on disk as individual JSON files.
"""

from __future__ import annotations

import errno
import hashlib
import json
import logging
import os
import threading
from pathlib import Path

from .private_io import atomic_write_private_text

_HASH_CHUNK_BYTES = 1024 * 1024
_OWNER_ONLY_DIR_MODE = 0o700
_OWNER_ONLY_FILE_MODE = 0o600
logger = logging.getLogger(__name__)
_shared_caches: dict[str, ResponseCache] = {}
_shared_caches_lock = threading.Lock()


def default_cache_dir() -> Path:
    """Return the default persistent cache directory."""
    return Path.home() / ".cache" / "folionym"


class CachePermissionError(RuntimeError):
    """Raised when a persistent cache path cannot be restricted to the owner."""

    def __init__(self, path: Path, mode: int, original: BaseException) -> None:
        """Record the path and requested mode for a cache permission failure."""
        self.path = path
        self.mode = mode
        super().__init__(f"Could not set owner-only permissions {mode:o} on {path}: {original}")


def _set_owner_only_permissions(path: Path, mode: int) -> None:
    """Restrict local document-derived cache data to the current owner."""
    try:
        path.chmod(mode)
    except (OSError, NotImplementedError) as exc:
        raise CachePermissionError(path, mode, exc) from exc


def _write_private_text(path: Path, text: str) -> None:
    """Write a cache entry atomically with owner-only permissions."""
    # fmt: off
    try:
        atomic_write_private_text(
            path,
            text,
            before_write=lambda temp_path: _set_owner_only_permissions(temp_path, _OWNER_ONLY_FILE_MODE),
        )
    except (OSError, CachePermissionError):
        raise
    # fmt: on


def _file_identity(status: os.stat_result) -> tuple[int, int, int, int]:
    """Return fields used to detect path or file changes during hashing."""
    return (status.st_dev, status.st_ino, status.st_size, status.st_mtime_ns)


class ResponseCache:
    """Cache string responses in memory and optionally on disk."""

    def __init__(self, cache_dir: str | Path | None = None) -> None:
        """Initialize an in-memory cache and enable private disk persistence when possible."""
        requested_cache_dir = Path(cache_dir).expanduser() if cache_dir else None
        self.cache_dir = None
        if requested_cache_dir is not None:
            try:
                requested_cache_dir.mkdir(parents=True, mode=_OWNER_ONLY_DIR_MODE, exist_ok=True)
                _set_owner_only_permissions(requested_cache_dir, _OWNER_ONLY_DIR_MODE)
            except (OSError, CachePermissionError) as exc:
                logger.warning("Persistent LLM cache disabled: %s", exc)
            else:
                self.cache_dir = requested_cache_dir
        self._memory: dict[str, str] = {}
        self._lock = threading.Lock()

    @staticmethod
    def build_file_key(path: str | Path) -> str:
        """Hash the complete file, failing if its path or opened file changes during the read."""
        file_path = Path(path)
        before = file_path.stat()
        digest = hashlib.sha256()
        with file_path.open("rb") as handle:
            opened = os.fstat(handle.fileno())
            if _file_identity(opened) != _file_identity(before):
                raise OSError(errno.EAGAIN, f"File changed before cache hashing began: {file_path}")
            while chunk := handle.read(_HASH_CHUNK_BYTES):
                digest.update(chunk)
            after_read = os.fstat(handle.fileno())
        after_path = file_path.stat()
        expected_identity = _file_identity(before)
        if _file_identity(after_read) != expected_identity or _file_identity(after_path) != expected_identity:
            raise OSError(errno.EAGAIN, f"File changed while building cache key: {file_path}")
        return digest.hexdigest()

    @staticmethod
    def derive_response_key(
        file_key: str,
        *,
        operation: str,
        model: str = "",
        language: str = "",
        extra: str = "",
    ) -> str:
        """Derive a response key from the base file fingerprint plus request metadata."""
        payload = "\0".join([file_key, operation, model, language, extra])
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _disk_path(self, key: str) -> Path | None:
        """Return the sanitized cache-entry path, or None when disk caching is disabled."""
        with self._lock:
            cache_dir = self.cache_dir
        if cache_dir is None:
            return None
        if len(key) == 64 and all(character in "0123456789abcdef" for character in key):
            disk_key = key
        else:
            disk_key = hashlib.sha256(key.encode("utf-8", errors="surrogatepass")).hexdigest()
        return cache_dir / f"{disk_key}.json"

    def get(self, key: str) -> str | None:
        """Return a memory or valid on-disk cached string; treat unreadable entries as misses."""
        with self._lock:
            cached = self._memory.get(key)
        if cached is not None:
            return cached
        disk_path = self._disk_path(key)
        if disk_path is None or not disk_path.exists():
            return None
        # fmt: off
        try:
            payload = json.loads(disk_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        # fmt: on
        value = payload.get("value")
        if not isinstance(value, str):
            return None
        with self._lock:
            self._memory[key] = value
        return value

    def set(self, key: str, value: str) -> None:
        """Cache a response in memory and persist it privately when disk caching is enabled."""
        with self._lock:
            self._memory[key] = value
        disk_path = self._disk_path(key)
        if disk_path is None:
            return
        payload = json.dumps({"value": value}, ensure_ascii=False)
        try:
            _write_private_text(disk_path, payload)
        except (OSError, CachePermissionError) as exc:
            logger.warning("Persistent LLM cache disabled: %s", exc)
            with self._lock:
                self.cache_dir = None


def get_shared_response_cache(cache_dir: str | Path | None = None) -> ResponseCache:
    """Return a shared cache instance for the current process."""
    key = "<memory>" if cache_dir is None else str(Path(cache_dir).expanduser())
    with _shared_caches_lock:
        cache = _shared_caches.get(key)
        if cache is None:
            cache = ResponseCache(cache_dir=cache_dir)
            _shared_caches[key] = cache
        return cache
