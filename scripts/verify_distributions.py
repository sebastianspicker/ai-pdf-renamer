#!/usr/bin/env python3
"""Verify built distributions and smoke-test the installed wheel."""

from __future__ import annotations

import argparse
import importlib.metadata
import sys
import tarfile
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any

from repository_hygiene import forbidden_reason

DATA_FILES = {
    "category_aliases.json",
    "heuristic_patterns.json",
    "heuristic_scores.json",
    "llm_response_schema.json",
    "meta_stopwords.json",
}
ENTRY_POINTS = {
    "ai-pdf-renamer": "ai_pdf_renamer.cli:main",
    "ai-pdf-renamer-gui": "ai_pdf_renamer.tui:main",
    "ai-pdf-renamer-tui": "ai_pdf_renamer.tui:main",
    "ai-pdf-renamer-undo": "ai_pdf_renamer.undo_cli:main",
}


def _archive_names(path: Path) -> list[str]:
    if path.suffix == ".whl":
        with zipfile.ZipFile(path) as archive:
            return archive.namelist()
    with tarfile.open(path, mode="r:gz") as archive:
        return archive.getnames()


def _verify_safe_member(archive: Path, member: PurePosixPath) -> None:
    reason = forbidden_reason(str(member))
    if reason is not None:
        raise AssertionError(f"{archive.name} contains forbidden path ({reason}): {member}")


def _package_members(members: list[PurePosixPath], name: str) -> list[PurePosixPath]:
    return [member for member in members if member.name == name and "ai_pdf_renamer" in member.parts]


def _verify_unique_member(archive: Path, members: list[PurePosixPath], name: str) -> None:
    count = len(_package_members(members, name))
    if count != 1:
        raise AssertionError(f"{archive.name}: expected {name} exactly once, found {count}")


def _verify_members(path: Path, names: list[str]) -> None:
    members = [PurePosixPath(name) for name in names if not name.endswith("/")]
    for member in members:
        _verify_safe_member(path, member)
    for data_file in DATA_FILES:
        _verify_unique_member(path, members, data_file)
    _verify_unique_member(path, members, "py.typed")


def _installed_entry_points() -> dict[str, importlib.metadata.EntryPoint]:
    return {
        point.name: point
        for point in importlib.metadata.entry_points(group="console_scripts")
        if point.name in ENTRY_POINTS
    }


def _assert_help(entry_point: Any, command: str) -> None:
    original_argv = sys.argv
    sys.argv = [command, "--help"]
    try:
        try:
            entry_point()
        except SystemExit as exc:
            if exc.code != 0:
                raise AssertionError(f"{command} --help exited with {exc.code}") from exc
    finally:
        sys.argv = original_argv


def _verify_installed_entry_points() -> None:
    points = _installed_entry_points()
    values = {name: point.value for name, point in points.items()}
    if values != ENTRY_POINTS:
        raise AssertionError(f"unexpected console entry points: {values}")
    loaded = {name: point.load() for name, point in points.items()}
    if not all(callable(entry_point) for entry_point in loaded.values()):
        raise AssertionError("all console entry points must resolve to callables")
    if points["ai-pdf-renamer-gui"].value != points["ai-pdf-renamer-tui"].value:
        raise AssertionError("GUI and TUI commands must remain aliases")
    _assert_help(loaded["ai-pdf-renamer"], "ai-pdf-renamer")
    _assert_help(loaded["ai-pdf-renamer-undo"], "ai-pdf-renamer-undo")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dist", type=Path, nargs="?", default=Path("dist"))
    parser.add_argument("--installed-wheel", action="store_true", help="verify entry points in this environment")
    args = parser.parse_args(argv)

    wheels = sorted(args.dist.glob("*.whl"))
    sdists = sorted(args.dist.glob("*.tar.gz"))
    if len(wheels) != 1 or len(sdists) != 1:
        raise AssertionError(
            f"expected one wheel and one sdist, found {len(wheels)} wheel(s) and {len(sdists)} sdist(s)"
        )
    for archive in (*wheels, *sdists):
        _verify_members(archive, _archive_names(archive))
    if args.installed_wheel:
        _verify_installed_entry_points()
    print(f"Verified {wheels[0].name} and {sdists[0].name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
