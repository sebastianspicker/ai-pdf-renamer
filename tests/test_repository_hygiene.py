from __future__ import annotations

import pytest

from scripts.repository_hygiene import forbidden_reason


@pytest.mark.parametrize(
    "path",
    [
        ".env",
        ".env.example",
        "fixtures/source.pdf",
        "certificates/signing.key",
        "metadata.csv",
        "exports/metadata.json",
        "docs/archive/private-notes.md",
        "docs/agent/status.md",
        "src/__pycache__/module.pyc",
        ".pytest_cache/v/cache/nodeids",
        ".agents/worktrees/task/notes.md",
        ".codacy/results.json",
        ".codegraph/index.db",
        ".serena/project.yml",
        ".claude/settings.json",
        "analysis/results.sarif",
    ],
)
def test_forbidden_repository_paths(path: str) -> None:
    assert forbidden_reason(path) is not None


@pytest.mark.parametrize(
    "path",
    [
        "README.md",
        "docs/README.md",
        "src/ai_pdf_renamer/data/heuristic_scores.json",
        "tests/fixtures/sample.txt",
        "vendor/library/source.py",
        "third_party/licenses/NOTICE",
    ],
)
def test_public_repository_paths(path: str) -> None:
    assert forbidden_reason(path) is None
