"""Behavioral contracts for distribution verification helpers."""

from __future__ import annotations

import importlib
import sys
from collections.abc import Callable
from pathlib import Path
from typing import NoReturn

import pytest

from scripts import repository_hygiene

# The executable script imports its sibling directly so ``python scripts/...`` works.
sys.modules.setdefault("repository_hygiene", repository_hygiene)
verify_distributions = importlib.import_module("scripts.verify_distributions")


def _exits_zero() -> NoReturn:
    raise SystemExit(0)


def _exits_nonzero() -> NoReturn:
    raise SystemExit(2)


@pytest.mark.parametrize("entry_point", [lambda: None, _exits_zero], ids=["returns", "exits-zero"])
def test_assert_help_succeeds_and_restores_argv(
    entry_point: Callable[[], None], monkeypatch: pytest.MonkeyPatch
) -> None:
    original_argv = ["pytest", "tests/unit/test_verify_distributions.py"]
    monkeypatch.setattr(verify_distributions.sys, "argv", original_argv)

    verify_distributions._assert_help(entry_point, "folionym")

    assert verify_distributions.sys.argv is original_argv


def test_assert_help_calls_entry_point_with_only_command_and_help(monkeypatch: pytest.MonkeyPatch) -> None:
    original_argv = ["pytest"]
    seen_argv: list[list[str]] = []
    monkeypatch.setattr(verify_distributions.sys, "argv", original_argv)

    def entry_point() -> None:
        seen_argv.append(verify_distributions.sys.argv)

    verify_distributions._assert_help(entry_point, "folionym-undo")

    assert seen_argv == [["folionym-undo", "--help"]]
    assert verify_distributions.sys.argv is original_argv


def test_assert_help_reports_nonzero_exit_and_restores_argv(monkeypatch: pytest.MonkeyPatch) -> None:
    original_argv = ["pytest"]
    monkeypatch.setattr(verify_distributions.sys, "argv", original_argv)

    with pytest.raises(AssertionError, match=r"folionym --help exited with 2"):
        verify_distributions._assert_help(_exits_nonzero, "folionym")

    assert verify_distributions.sys.argv is original_argv


def test_assert_help_propagates_other_exceptions_and_restores_argv(monkeypatch: pytest.MonkeyPatch) -> None:
    original_argv = ["pytest"]
    expected = RuntimeError("entry point failed")
    monkeypatch.setattr(verify_distributions.sys, "argv", original_argv)

    def entry_point() -> NoReturn:
        raise expected

    with pytest.raises(RuntimeError) as raised:
        verify_distributions._assert_help(entry_point, "folionym")

    assert raised.value is expected
    assert verify_distributions.sys.argv is original_argv


def _required_members() -> list[str]:
    return [
        *(f"folionym/data/{name}" for name in verify_distributions.DATA_FILES),
        "folionym/py.typed",
    ]


def test_verify_members_accepts_wheel_with_required_data_and_frontend() -> None:
    members = [*_required_members(), "folionym/web_dist/index.html"]

    verify_distributions._verify_members(Path("folionym-1.0.0-py3-none-any.whl"), members)


def test_verify_members_requires_wheel_frontend() -> None:
    with pytest.raises(AssertionError, match="packaged browser frontend is missing"):
        verify_distributions._verify_members(Path("folionym-1.0.0-py3-none-any.whl"), _required_members())
