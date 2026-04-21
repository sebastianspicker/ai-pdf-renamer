from __future__ import annotations

import contextlib
from pathlib import Path


def _lookup_override_category(file_path: Path, override_map: dict[str, str] | None) -> str | None:
    if not override_map:
        return None

    def _normalize_path(value: str) -> str:
        return value.replace("\\", "/")

    direct_keys = [file_path.name, str(file_path), file_path.as_posix()]
    with contextlib.suppress(OSError):
        resolved = file_path.resolve()
        direct_keys.extend([str(resolved), resolved.as_posix()])

    for key in direct_keys:
        category = override_map.get(key)
        if category:
            return category

    normalized_candidates = [_normalize_path(key) for key in direct_keys]
    for raw_key, category in override_map.items():
        normalized_key = _normalize_path(raw_key)
        if "/" not in normalized_key:
            continue
        if any(
            candidate == normalized_key or candidate.endswith(f"/{normalized_key}")
            for candidate in normalized_candidates
        ):
            return category
    return None
