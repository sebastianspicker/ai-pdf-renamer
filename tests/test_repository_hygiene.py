from __future__ import annotations

import importlib.util
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "repository_hygiene", Path(__file__).parents[1] / "scripts" / "repository_hygiene.py"
)
assert _SPEC and _SPEC.loader
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
forbidden_reason = _MODULE.forbidden_reason


def test_repository_hygiene_rejects_agent_guides() -> None:
    for filename in (
        ".cursorrules",
        "AGENT.md",
        "AGENTS.md",
        "CLAUDE.md",
        "CODEX.md",
        "GEMINI.md",
        "agent.md",
        "agents.md",
    ):
        assert forbidden_reason(filename) == "development-assistant instructions"


def test_repository_hygiene_rejects_nested_agent_guides() -> None:
    assert forbidden_reason("src/AGENTS.md") == "development-assistant instructions"
