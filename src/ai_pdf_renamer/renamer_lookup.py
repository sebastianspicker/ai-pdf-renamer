from __future__ import annotations

import contextlib
from pathlib import Path


def _normalize_path(value: str) -> str:
    return value.replace("\\", "/")


def _direct_override_keys(file_path: Path) -> list[str]:
    direct_keys = [file_path.name, str(file_path), file_path.as_posix()]
    with contextlib.suppress(OSError):
        resolved = file_path.resolve()
        direct_keys.extend([str(resolved), resolved.as_posix()])
    return direct_keys


def _lookup_direct_override(direct_keys: list[str], override_map: dict[str, str]) -> str | None:
    for key in direct_keys:
        category = override_map.get(key)
        if category:
            return category
    return None


def _matches_path_override(normalized_key: str, normalized_candidates: list[str]) -> bool:
    if "/" not in normalized_key:
        return False
    return any(
        candidate == normalized_key or candidate.endswith(f"/{normalized_key}") for candidate in normalized_candidates
    )


def _lookup_path_override(direct_keys: list[str], override_map: dict[str, str]) -> str | None:
    normalized_candidates = [_normalize_path(key) for key in direct_keys]
    for raw_key, category in override_map.items():
        if _matches_path_override(_normalize_path(raw_key), normalized_candidates):
            return category
    return None


def _lookup_override_category(file_path: Path, override_map: dict[str, str] | None) -> str | None:
    if not override_map:
        return None
    direct_keys = _direct_override_keys(file_path)
    return _lookup_direct_override(direct_keys, override_map) or _lookup_path_override(direct_keys, override_map)
