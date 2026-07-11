"""CLI config-file and override-map loading helpers."""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import cast

from rich.console import Console

logger = logging.getLogger(__name__)
_console = Console(stderr=True)


class ConfigLoadError(ValueError):
    """Raised when an explicitly requested config file cannot be loaded safely."""


def _fail_config_load(path: str | Path, message: str, *details: str, raise_on_error: bool = False) -> dict[str, object]:
    _console.print(message)
    for detail in details:
        _console.print(detail)
    if raise_on_error:
        raise ConfigLoadError(str(path))
    return {}


def _load_json_config_file(path: Path, raw: str, *, raise_on_error: bool) -> dict[str, object]:
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        return _fail_config_load(
            path,
            f"[red]Invalid JSON in config file[/red] {path}",
            f"[dim]  Line {exc.lineno}, col {exc.colno}: {exc.msg}[/dim]",
            raise_on_error=raise_on_error,
        )
    if not isinstance(data, dict):
        dtype = type(data).__name__
        return _fail_config_load(
            path,
            f"[yellow]Config file must contain a JSON object (got {dtype}):[/yellow] {path}",
            raise_on_error=raise_on_error,
        )
    return data


def _load_yaml_config_file(path: Path, raw: str, *, raise_on_error: bool) -> dict[str, object]:
    try:
        import yaml
    except ImportError:
        return _fail_config_load(
            path,
            "[red]Cannot parse YAML config:[/red] PyYAML not installed.",
            "[dim]  Install with: pip install pyyaml[/dim]",
            raise_on_error=raise_on_error,
        )

    yaml_error = cast(type[BaseException], getattr(yaml, "YAMLError", ValueError))
    try:
        data = yaml.safe_load(raw)
    except (yaml_error, ValueError) as exc:
        return _fail_config_load(
            path,
            f"[red]Invalid YAML in config file[/red] {path}: {exc}",
            raise_on_error=raise_on_error,
        )
    if not isinstance(data, dict):
        dtype = type(data).__name__
        return _fail_config_load(
            path,
            f"[yellow]Config file must contain a YAML mapping (got {dtype}):[/yellow] {path}",
            raise_on_error=raise_on_error,
        )
    return data


def _load_config_file(path: str | Path, *, raise_on_error: bool = False) -> dict[str, object]:
    """Load JSON or YAML config file. Returns a dict unless raise_on_error is enabled."""
    p = Path(path)
    if not p.exists():
        return _fail_config_load(
            path,
            f"[yellow]Config file not found:[/yellow] {p}",
            "[dim]Create one with JSON or YAML format, or omit --config.[/dim]",
            raise_on_error=raise_on_error,
        )
    try:
        raw = p.read_text(encoding="utf-8")
    except OSError as exc:
        return _fail_config_load(path, f"[red]Cannot read config file[/red] {p}: {exc}", raise_on_error=raise_on_error)
    suf = p.suffix.lower()
    if suf == ".json":
        return _load_json_config_file(p, raw, raise_on_error=raise_on_error)
    if suf in (".yaml", ".yml"):
        return _load_yaml_config_file(p, raw, raise_on_error=raise_on_error)
    return _fail_config_load(
        path,
        f"[yellow]Unsupported config file format:[/yellow] {p.suffix}",
        "[dim]Supported formats: .json, .yaml, .yml[/dim]",
        raise_on_error=raise_on_error,
    )


def load_config_file(path: str | Path, *, raise_on_error: bool = False) -> dict[str, object]:
    """Load a JSON or YAML CLI config file."""
    return _load_config_file(path, raise_on_error=raise_on_error)


def _load_override_category_map(path: str | Path) -> dict[str, str]:
    """Load CSV with columns filename,category (or path,category). Returns dict filename -> category."""
    result: dict[str, str] = {}
    p = Path(path)
    if not p.exists():
        return result
    try:
        with open(p, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                _add_override_category_row(result, row)
    except OSError as exc:
        logger.warning("Could not read override-category file %s: %s. Proceeding with no overrides.", p, exc)
    return result


def _add_override_category_row(result: dict[str, str], row: dict[str, str | None]) -> None:
    name = (row.get("filename") or row.get("path") or row.get("file") or "").strip()
    category = (row.get("category") or "").strip()
    if name and category:
        result[name] = category
