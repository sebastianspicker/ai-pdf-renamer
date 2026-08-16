"""Define which paths are safe or forbidden in public distributions."""

from __future__ import annotations

import pytest

from scripts.repository_hygiene import forbidden_reason


@pytest.mark.parametrize(
    "path",
    [
        ".env",
        ".env.local",
        ".env.example.local",
        "fixtures/source.pdf",
        "models/local-model.gguf",
        "models/encoder.onnx",
        "models/weights.safetensors",
        "cache/responses.sqlite3",
        "certificates/signing.key",
        "metadata.csv",
        "exports/metadata.json",
        "docs/archive/private-notes.md",
        "docs/agent/status.md",
        "src/__pycache__/module.pyc",
        ".pytest_cache/v/cache/nodeids",
        ".agents/worktrees/task/notes.md",
        ".ai/session.json",
        ".aider/history.md",
        ".codacy/results.json",
        ".codegraph/index.db",
        ".continue/config.json",
        ".serena/project.yml",
        ".windsurf/rules.md",
        ".claude/settings.json",
        ".github/copilot-instructions.md",
        ".github/prompts/review.prompt.md",
        "PROMPTS.local.md",
        "prompts.local/release.md",
        "conversation-exports/session.json",
        "chat-exports/thread.md",
        "analysis/results.sarif",
        "RELEASE_STATUS.md",
        "release-status.md",
        "REMEDIATION_PLAN.md",
    ],
)
def test_forbidden_repository_paths(path: str) -> None:
    assert forbidden_reason(path) is not None


@pytest.mark.parametrize(
    "path",
    [
        "README.md",
        "RELEASING.md",
        ".env.example",
        ".github/release.yml",
        "docs/README.md",
        "docs/frontend.md",
        "docs/tui.md",
        "docs/screenshots/tui-preview.svg",
        "src/folionym/data/heuristic_scores.json",
        "tests/fixtures/sample.txt",
        "vendor/library/source.py",
        "third_party/licenses/NOTICE",
    ],
)
def test_public_repository_paths(path: str) -> None:
    assert forbidden_reason(path) is None
