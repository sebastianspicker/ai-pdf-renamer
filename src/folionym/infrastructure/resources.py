"""Packaged data lookup and JSON loading for Folionym runtime resources."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Literal

DataFileName = Literal[
    "heuristic_scores.json",
    "heuristic_scores_de.json",
    "heuristic_scores_en.json",
    "meta_stopwords.json",
    "category_aliases.json",
]

# Runtime-supported data files that may be loaded through data_path().
DATA_FILES: frozenset[str] = frozenset(
    {
        "heuristic_scores.json",
        "heuristic_scores_de.json",
        "heuristic_scores_en.json",
        "meta_stopwords.json",
        "category_aliases.json",
    }
)


def default_cache_dir() -> Path:
    """Return the default private directory for persistent LLM responses."""
    return Path.home() / ".cache" / "folionym"


def load_json_data(path: str | Path) -> Any:
    """Read one JSON data file with consistent path-aware errors."""
    path_obj = Path(path)
    try:
        text = path_obj.read_text(encoding="utf-8")
    except OSError as exc:
        raise ValueError(f"Could not read data file at {path_obj.absolute()}: {exc!s}") from exc
    try:
        data: Any = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in data file at {path_obj.absolute()}. {exc!s}") from exc
    return data


def _discover_repo_root(start: Path | None = None) -> Path | None:
    """Return repo root containing pyproject.toml, or None when not found."""
    if start is None:
        start = Path(__file__).resolve()
    for parent in [start, *start.parents]:
        if (parent / "pyproject.toml").exists():
            return parent
    return None


def project_root(start: Path | None = None) -> Path:
    """Best-effort editable-run project root, falling back to the current directory."""
    discovered = _discover_repo_root(start)
    if discovered is not None:
        return discovered
    return Path.cwd()


def data_dir() -> Path:
    """Return the data directory from an override, repository root, or package data."""
    override = (os.getenv("FOLIONYM_DATA_DIR") or "").strip()
    if override:
        return Path(override).expanduser().resolve()
    repo_root = _discover_repo_root(Path(__file__).resolve())
    if repo_root is not None:
        return repo_root.resolve()
    # Installed-package fallback: use bundled data, not the process CWD.
    return (Path(__file__).resolve().parent.parent / "data").resolve()


def package_data_path(filename: str) -> Path:
    """Return the path to a data file within the installed package's data directory."""
    return Path(__file__).resolve().parent.parent / "data" / filename


def category_aliases_path() -> Path:
    """Path to category_aliases.json: data_dir first (if present), else package data."""
    override = data_dir() / "category_aliases.json"
    if override.exists() and override.is_file():
        return override
    return package_data_path("category_aliases.json")


def data_path(filename: DataFileName) -> Path:
    """Resolve a data filename through override, repo root, then package data."""
    if filename not in DATA_FILES:
        raise ValueError(f"Unsupported data file: {filename}")
    candidate = data_dir() / filename
    if candidate.exists() and candidate.is_file():
        return candidate

    packaged = package_data_path(filename)
    if packaged.exists() and packaged.is_file():
        return packaged

    raise FileNotFoundError(
        f"Data file {filename!r} not found. Looked in: {candidate} and {packaged}. "
        "Set FOLIONYM_DATA_DIR to a directory containing the JSON files, "
        "or run from the project root."
    )
