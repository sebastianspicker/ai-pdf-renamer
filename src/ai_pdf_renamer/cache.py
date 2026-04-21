"""Response caching for repeated LLM work.

Cache entries always live in memory for the current process. When ``cache_dir``
is set, entries are also persisted on disk as individual JSON files.
"""

from __future__ import annotations

import hashlib
import json
import threading
from pathlib import Path

_DEFAULT_PREFIX_BYTES = 65_536
_shared_caches: dict[str, ResponseCache] = {}
_shared_caches_lock = threading.Lock()


def default_cache_dir() -> Path:
    """Return the default persistent cache directory."""
    return Path.home() / ".cache" / "ai-pdf-renamer"


class ResponseCache:
    """Cache string responses in memory and optionally on disk."""

    def __init__(self, cache_dir: str | Path | None = None) -> None:
        self.cache_dir = Path(cache_dir).expanduser() if cache_dir else None
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
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
        disk_path.write_text(payload, encoding="utf-8")


def get_shared_response_cache(cache_dir: str | Path | None = None) -> ResponseCache:
    """Return a shared cache instance for the current process."""
    key = "<memory>" if cache_dir is None else str(Path(cache_dir).expanduser())
    with _shared_caches_lock:
        cache = _shared_caches.get(key)
        if cache is None:
            cache = ResponseCache(cache_dir=cache_dir)
            _shared_caches[key] = cache
        return cache
