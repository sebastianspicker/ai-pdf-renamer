"""Response caching for repeated LLM work.

Cache entries always live in memory for the current process. When ``cache_dir``
is set, entries are also persisted on disk as individual JSON files.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
from contextlib import suppress
from pathlib import Path

_DEFAULT_PREFIX_BYTES = 65_536
_OWNER_ONLY_DIR_MODE = 0o700
_OWNER_ONLY_FILE_MODE = 0o600
logger = logging.getLogger(__name__)
_shared_caches: dict[str, ResponseCache] = {}
_shared_caches_lock = threading.Lock()


def default_cache_dir() -> Path:
    """Return the default persistent cache directory."""
    return Path.home() / ".cache" / "ai-pdf-renamer"


class CachePermissionError(RuntimeError):
    """Raised when a persistent cache path cannot be restricted to the owner."""

    def __init__(self, path: Path, mode: int, original: BaseException) -> None:
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
    try:
        if os.name == "posix":
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, _OWNER_ONLY_FILE_MODE)
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(text)
        else:
            path.write_text(text, encoding="utf-8")
        _set_owner_only_permissions(path, _OWNER_ONLY_FILE_MODE)
    except (OSError, CachePermissionError):
        with suppress(OSError):
            path.unlink()
        raise


class ResponseCache:
    """Cache string responses in memory and optionally on disk."""

    def __init__(self, cache_dir: str | Path | None = None) -> None:
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
    def build_file_key(path: str | Path, *, prefix_bytes: int = _DEFAULT_PREFIX_BYTES) -> str:
        """Build a stable key from file size plus leading and trailing bytes."""
        file_path = Path(path)
        stat = file_path.stat()
        with file_path.open("rb") as handle:
            prefix = handle.read(prefix_bytes)
            tail = b""
            if stat.st_size > len(prefix):
                tail_bytes = min(prefix_bytes, stat.st_size)
                handle.seek(stat.st_size - tail_bytes)
                tail = handle.read(tail_bytes)
        digest = hashlib.sha256()
        digest.update(str(stat.st_size).encode("ascii"))
        digest.update(b"\0")
        digest.update(prefix)
        digest.update(b"\0")
        digest.update(tail)
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
        if self.cache_dir is None:
            return None
        return self.cache_dir / f"{key}.json"

    def get(self, key: str) -> str | None:
        with self._lock:
            cached = self._memory.get(key)
        if cached is not None:
            return cached
        disk_path = self._disk_path(key)
        if disk_path is None or not disk_path.exists():
            return None
        try:
            payload = json.loads(disk_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        value = payload.get("value")
        if not isinstance(value, str):
            return None
        with self._lock:
            self._memory[key] = value
        return value

    def set(self, key: str, value: str) -> None:
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
