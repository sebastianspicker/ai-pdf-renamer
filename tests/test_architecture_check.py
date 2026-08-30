from __future__ import annotations

import importlib.util
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "check_architecture", Path(__file__).parents[1] / "scripts" / "check_architecture.py"
)
assert _SPEC and _SPEC.loader
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
check_architecture = _MODULE.check_architecture


def _write(package_root: Path, relative_path: str, contents: str = "") -> None:
    path = package_root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(contents, encoding="utf-8")


def test_check_architecture_resolves_relative_layer_imports(tmp_path: Path) -> None:
    package_root = tmp_path / "folionym"
    _write(package_root, "naming/service.py", "from ..application import Proposal\n")

    diagnostics = check_architecture(package_root)

    assert diagnostics == [
        "src/folionym/naming/service.py:1: architecture violation: "
        "naming cannot import application (found 'folionym.application')"
    ]


def test_check_architecture_detects_concrete_llm_http_import(tmp_path: Path) -> None:
    package_root = tmp_path / "folionym"
    _write(package_root, "naming/service.py", "from folionym.llm import http\n")

    diagnostics = check_architecture(package_root)

    assert diagnostics == [
        "src/folionym/naming/service.py:1: architecture violation: "
        "naming cannot import llm.http (found 'folionym.llm.http')"
    ]


def test_check_architecture_rejects_framework_imports_from_naming(tmp_path: Path) -> None:
    package_root = tmp_path / "folionym"
    _write(package_root, "naming/service.py", "import requests\n")

    diagnostics = check_architecture(package_root)

    assert diagnostics == [
        "src/folionym/naming/service.py:1: architecture violation: naming cannot import requests (found 'requests')"
    ]


def test_check_architecture_rejects_model_infrastructure_and_root_debris(tmp_path: Path) -> None:
    package_root = tmp_path / "folionym"
    _write(package_root, "application/models.py", "from fastapi import FastAPI\n")
    _write(package_root, "temporary.py")

    diagnostics = check_architecture(package_root)

    assert diagnostics == [
        "src/folionym/temporary.py:1: architecture violation: "
        "production package root only permits public facades and metadata",
        "src/folionym/application/models.py:1: architecture violation: "
        "application/models.py cannot import FastAPI (found 'fastapi')",
    ]


def test_check_architecture_rejects_infrastructure_reverse_dependencies(tmp_path: Path) -> None:
    package_root = tmp_path / "folionym"
    _write(package_root, "infrastructure/files.py", "from ..settings import RenamerConfig\n")

    diagnostics = check_architecture(package_root)

    assert diagnostics == [
        "src/folionym/infrastructure/files.py:1: architecture violation: "
        "infrastructure cannot import settings (found 'folionym.settings')"
    ]


def test_check_architecture_rejects_internal_imports_through_public_facades(tmp_path: Path) -> None:
    package_root = tmp_path / "folionym"
    _write(package_root, "application/proposals.py", "from ..filename import generate_filename\n")

    diagnostics = check_architecture(package_root)

    assert diagnostics == [
        "src/folionym/application/proposals.py:1: architecture violation: "
        "application cannot import outward public facade filename (found 'folionym.filename')"
    ]
