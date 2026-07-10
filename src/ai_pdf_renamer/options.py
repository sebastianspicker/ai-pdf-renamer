"""Shared helpers for immutable dataclass option objects."""

from __future__ import annotations

from dataclasses import fields, replace
from typing import Any, TypeVar, cast

_OptionsT = TypeVar("_OptionsT")


def merge_options(options: _OptionsT, overrides: dict[str, object]) -> _OptionsT:
    """Return a copy of a dataclass options object with validated overrides."""
    allowed = {field.name for field in fields(cast(Any, options))}
    unknown = sorted(set(overrides) - allowed)
    if unknown:
        raise TypeError(f"Unexpected option(s): {', '.join(unknown)}")
    return cast(_OptionsT, replace(cast(Any, options), **overrides))
