from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

DataFileName = Literal[
    "heuristic_patterns.json",
    "heuristic_scores.json",
    "heuristic_scores_de.json",
    "heuristic_scores_en.json",
    "meta_stopwords.json",
    "category_aliases.json",
]

# P3: Include all data files that may be loaded
DATA_FILES: frozenset[str] = frozenset(
    {
        "heuristic_patterns.json",
        "heuristic_scores.json",
        "heuristic_scores_de.json",
        "heuristic_scores_en.json",
        "meta_stopwords.json",
        "category_aliases.json",
    }
)


def _discover_repo_root(start: Path | None = None) -> Path | None:
    """Return repo root containing pyproject.toml, or None when not found."""
    if start is None:
        start = Path(__file__).resolve()
    for parent in [start, *start.parents]:
        if (parent / "pyproject.toml").exists():
            return parent
    return None


def project_root(start: Path | None = None) -> Path:
    """
    Best-effort discovery of the repo root for editable/dev runs.
    Falls back to CWD if nothing is found.
    """
    discovered = _discover_repo_root(start)
    if discovered is not None:
        return discovered
    return Path.cwd()


def data_dir() -> Path:
    """Return the resolved data directory, preferring AI_PDF_RENAMER_DATA_DIR env override, then repo root."""
    override = (os.getenv("AI_PDF_RENAMER_DATA_DIR") or "").strip()
    if override:
        return Path(override).expanduser().resolve()
    repo_root = _discover_repo_root(Path(__file__).resolve())
    if repo_root is not None:
        return repo_root.resolve()
    # Installed-package fallback: use bundled data, not the process CWD.
    return (Path(__file__).resolve().parent / "data").resolve()


def package_data_path(filename: str) -> Path:
    """Return the path to a data file within the installed package's data directory."""
    return Path(__file__).resolve().parent / "data" / filename


def category_aliases_path() -> Path:
    """Path to category_aliases.json: data_dir first (if present), else package data."""
    override = data_dir() / "category_aliases.json"
    if override.exists() and override.is_file():
        return override
    return package_data_path("category_aliases.json")


def data_path(filename: DataFileName) -> Path:
    """Resolve a data filename to its full path, checking env override, repo root, then package data."""
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
        "Set AI_PDF_RENAMER_DATA_DIR to a directory containing the JSON files, "
        "or run from the project root."
    )
