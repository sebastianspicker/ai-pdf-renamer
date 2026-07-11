#!/usr/bin/env python3
"""Reject tracked repository paths that may contain private or generated data."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Iterable
from pathlib import PurePosixPath

_FORBIDDEN_COMPONENTS = {
    ".agents",
    ".claude",
    ".codacy",
    ".codegraph",
    ".codex",
    ".cursor",
    ".kilo",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".serena",
    "__pycache__",
}
_FORBIDDEN_DOC_LANES = {
    "agent",
    "archive",
    "deprecated",
    "ledgers",
    "reports",
    "status",
    "tmp",
}
_FORBIDDEN_SUFFIXES = {
    ".key",
    ".p12",
    ".pdf",
    ".pem",
    ".pfx",
    ".sarif",
}
_FORBIDDEN_BASENAMES = {
    ".DS_Store",
    "AGENTS.md",
    "credentials.json",
    "metadata.csv",
    "metadata.json",
    "plan.json",
    "rename.log",
    "secrets.yaml",
    "secrets.yml",
    "summary.json",
}


def _invalid_path(path: PurePosixPath) -> bool:
    return not path.parts or path.is_absolute() or ".." in path.parts


def _private_basename(path: PurePosixPath) -> bool:
    return path.name in _FORBIDDEN_BASENAMES


def _environment_file(path: PurePosixPath) -> bool:
    return path.name == ".env" or path.name.startswith(".env.")


def _credential_export(path: PurePosixPath) -> bool:
    return path.name.startswith("service-account") and path.name.endswith(".json")


def _rename_log(path: PurePosixPath) -> bool:
    return path.name.startswith("rename-") and path.name.endswith(".log")


def _private_file_type(path: PurePosixPath) -> bool:
    return path.suffix.lower() in _FORBIDDEN_SUFFIXES


def _tool_workspace(path: PurePosixPath) -> bool:
    return any(part in _FORBIDDEN_COMPONENTS for part in path.parts)


def _package_metadata(path: PurePosixPath) -> bool:
    return any(part.endswith(".egg-info") for part in path.parts)


def _build_output(path: PurePosixPath) -> bool:
    return any(part in {"build", "dist"} for part in path.parts)


def _private_docs(path: PurePosixPath) -> bool:
    return any(
        part == "docs" and index + 1 < len(path.parts) and path.parts[index + 1] in _FORBIDDEN_DOC_LANES
        for index, part in enumerate(path.parts)
    )


_FORBIDDEN_CHECKS: tuple[tuple[Callable[[PurePosixPath], bool], str], ...] = (
    (_invalid_path, "invalid repository path"),
    (_private_basename, "private or generated artifact"),
    (_environment_file, "environment file"),
    (_credential_export, "credential export"),
    (_rename_log, "rename log"),
    (_private_file_type, "private or generated file type"),
    (_tool_workspace, "agent or tool workspace"),
    (_package_metadata, "package build metadata"),
    (_build_output, "package build output"),
    (_private_docs, "private documentation lane"),
)


def forbidden_reason(path_text: str) -> str | None:
    """Return the policy reason for a forbidden repository-relative path."""
    path = PurePosixPath(path_text)
    return next((reason for check, reason in _FORBIDDEN_CHECKS if check(path)), None)


def find_forbidden(paths: Iterable[str]) -> list[tuple[str, str]]:
    """Return sorted forbidden paths and their policy reasons."""
    findings = []
    for path in paths:
        reason = forbidden_reason(path)
        if reason is not None:
            findings.append((path, reason))
    return sorted(findings)


def _stdin_paths() -> list[str]:
    return [item.decode("utf-8", errors="surrogateescape") for item in sys.stdin.buffer.read().split(b"\0") if item]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--null-stdin", action="store_true", help="read NUL-delimited repository paths from stdin")
    parser.add_argument("paths", nargs="*", help="explicit repository-relative paths")
    args = parser.parse_args(argv)

    if not args.paths and not args.null_stdin:
        parser.error("provide explicit paths or --null-stdin")
    paths = _stdin_paths() if args.null_stdin else args.paths
    findings = find_forbidden(paths)
    if not findings:
        return 0

    print("Forbidden private, local, or generated artifacts:", file=sys.stderr)
    for path, reason in findings:
        print(f"  {path}: {reason}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
