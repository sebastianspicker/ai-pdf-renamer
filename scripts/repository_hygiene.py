#!/usr/bin/env python3
"""Reject tracked repository paths that may contain private or generated data."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Iterable
from pathlib import PurePosixPath

_FORBIDDEN_COMPONENTS = {
    ".agents",
    ".ai",
    ".aider",
    ".claude",
    ".codacy",
    ".codegraph",
    ".codex",
    ".continue",
    ".cursor",
    ".kilo",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".serena",
    ".windsurf",
    "__pycache__",
}
_FORBIDDEN_ROOT_DIRECTORIES = {
    "chat-exports",
    "conversation-exports",
    "prompts.local",
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
    ".gguf",
    ".key",
    ".onnx",
    ".p12",
    ".pdf",
    ".pem",
    ".pfx",
    ".sarif",
    ".safetensors",
    ".sqlite",
    ".sqlite3",
}
_FORBIDDEN_BASENAMES = {
    ".DS_Store",
    "credentials.json",
    "metadata.csv",
    "metadata.json",
    "plan.json",
    "rename.log",
    "secrets.yaml",
    "secrets.yml",
    "summary.json",
}
_FORBIDDEN_AGENT_GUIDES = {
    ".cursorrules",
    "agent.md",
    "agents.md",
    "claude.md",
    "codex.md",
    "gemini.md",
}
_FORBIDDEN_ROOT_DOCS = {
    "agent_audit_report.md",
    "audit.md",
    "code_review.md",
    "decisions.md",
    "findings.md",
    "handoff.md",
    "log.md",
    "progress.md",
    "prompts.local.md",
    "refactor_plan.md",
    "release_status.md",
    "remediation_ledger.md",
    "remediation_plan.md",
    "remediation_status.md",
    "validation.md",
}


def _invalid_path(path: PurePosixPath) -> bool:
    """Return whether a path is empty, absolute, or contains parent traversal."""
    return not path.parts or path.is_absolute() or ".." in path.parts


def _private_basename(path: PurePosixPath) -> bool:
    """Return whether the basename is explicitly forbidden by repository policy."""
    return path.name in _FORBIDDEN_BASENAMES


def _agent_guide(path: PurePosixPath) -> bool:
    """Return whether the path is a development-assistant instruction file."""
    return path.name.lower() in _FORBIDDEN_AGENT_GUIDES


def _environment_file(path: PurePosixPath) -> bool:
    """Return whether the basename is .env or a private variant, not the public template."""
    return path.name == ".env" or (path.name.startswith(".env.") and path.name != ".env.example")


def _credential_export(path: PurePosixPath) -> bool:
    """Return whether the basename matches a service-account JSON credential export."""
    return path.name.startswith("service-account") and path.name.endswith(".json")


def _rename_log(path: PurePosixPath) -> bool:
    """Return whether the basename matches rename-*.log."""
    return path.name.startswith("rename-") and path.name.endswith(".log")


def _private_file_type(path: PurePosixPath) -> bool:
    """Return whether the suffix is a forbidden private or generated artifact type."""
    return path.suffix.lower() in _FORBIDDEN_SUFFIXES


def _local_workspace(path: PurePosixPath) -> bool:
    """Return whether any component names a local automation workspace or tool cache."""
    return any(part in _FORBIDDEN_COMPONENTS for part in path.parts)


def _local_root_directory(path: PurePosixPath) -> bool:
    """Return whether a path enters a local export or private prompt directory."""
    return bool(path.parts) and path.parts[0] in _FORBIDDEN_ROOT_DIRECTORIES


def _github_automation_instructions(path: PurePosixPath) -> bool:
    """Return whether a path contains local GitHub automation instructions."""
    return path.parts == (".github", "copilot-instructions.md") or (
        len(path.parts) >= 2 and path.parts[:2] == (".github", "prompts")
    )


def _package_metadata(path: PurePosixPath) -> bool:
    """Return whether any component is Python egg-info metadata."""
    return any(part.endswith(".egg-info") for part in path.parts)


def _build_output(path: PurePosixPath) -> bool:
    """Return whether any component is a build or dist directory."""
    return any(part in {"build", "dist"} for part in path.parts)


def _private_docs(path: PurePosixPath) -> bool:
    """Return whether the path enters a forbidden docs/<lane> subtree."""
    return any(
        part == "docs" and index + 1 < len(path.parts) and path.parts[index + 1] in _FORBIDDEN_DOC_LANES
        for index, part in enumerate(path.parts)
    )


def _private_root_document(path: PurePosixPath) -> bool:
    """Return whether the path is a forbidden root-level audit or status document."""
    normalized_name = path.name.lower().replace("-", "_")
    return len(path.parts) == 1 and normalized_name in _FORBIDDEN_ROOT_DOCS


_FORBIDDEN_CHECKS: tuple[tuple[Callable[[PurePosixPath], bool], str], ...] = (
    (_invalid_path, "invalid repository path"),
    (_private_basename, "private or generated artifact"),
    (_agent_guide, "development-assistant instructions"),
    (_environment_file, "environment file"),
    (_credential_export, "credential export"),
    (_rename_log, "rename log"),
    (_private_file_type, "private or generated file type"),
    (_local_workspace, "local automation workspace or tool cache"),
    (_local_root_directory, "local export or private prompt directory"),
    (_github_automation_instructions, "local GitHub automation instructions"),
    (_package_metadata, "package build metadata"),
    (_build_output, "package build output"),
    (_private_docs, "private documentation lane"),
    (_private_root_document, "private root documentation"),
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
    """Read NUL-delimited repository paths from stdin using surrogateescape."""
    return [item.decode("utf-8", errors="surrogateescape") for item in sys.stdin.buffer.read().split(b"\0") if item]


def main(argv: list[str] | None = None) -> int:
    """Check explicit or NUL-delimited paths and return 1 when forbidden artifacts are found."""
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
