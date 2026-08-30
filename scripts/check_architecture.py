"""Enforce Folionym's import-direction and package-layout boundaries."""

from __future__ import annotations

import argparse
import ast
from collections.abc import Iterable
from pathlib import Path

PROJECT_PACKAGE = "folionym"
ALLOWED_ROOT_MODULES = {"__init__.py", "config.py", "filename.py", "heuristics.py", "renamer.py"}
PUBLIC_FACADE_MODULES = {"config", "filename", "heuristics", "renamer"}
LAYER_RULES = {
    "naming": ("application", "extraction", "interfaces", "llm.http"),
    "llm": ("application", "extraction", "interfaces", "naming"),
    "extraction": ("application", "interfaces"),
    "settings": ("application", "extraction", "interfaces", "llm", "naming"),
    "rename_ops": ("application", "extraction", "interfaces", "llm", "naming", "settings"),
    "application": ("interfaces",),
    "infrastructure": ("application", "extraction", "interfaces", "llm", "naming", "rename_ops", "settings"),
    "interfaces": (),
}
MODEL_EXTERNAL_IMPORTS = {
    "requests": "requests",
    "fastapi": "FastAPI",
    "textual": "Textual",
    "fitz": "PyMuPDF",
    "ocrmypdf": "OCRmyPDF",
}
PURE_LAYER_EXTERNAL_IMPORTS = {"naming": MODEL_EXTERNAL_IMPORTS}


def _matches_prefix(module: str, prefix: str) -> bool:
    """Return whether *module* is exactly *prefix* or one of its children."""
    return module == prefix or module.startswith(f"{prefix}.")


def _module_name(path: Path, source_root: Path) -> str:
    relative = path.relative_to(source_root).with_suffix("")
    parts = [PROJECT_PACKAGE, *relative.parts]
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _package_name(path: Path, source_root: Path) -> str:
    module = _module_name(path, source_root)
    return module if path.name == "__init__.py" else module.rpartition(".")[0]


def _import_targets(node: ast.Import | ast.ImportFrom, package: str) -> Iterable[str]:
    if isinstance(node, ast.Import):
        yield from (alias.name for alias in node.names)
        return

    if node.level:
        package_parts = package.split(".")
        parent_parts = package_parts[: len(package_parts) - node.level + 1]
        if not parent_parts:
            return
        base = ".".join(parent_parts)
        if node.module:
            base = f"{base}.{node.module}"
    elif node.module:
        base = node.module
    else:
        return

    yield base
    yield from (f"{base}.{alias.name}" for alias in node.names)


def _layer_for(path: Path, source_root: Path) -> str | None:
    relative_parts = path.relative_to(source_root).parts
    return relative_parts[0] if relative_parts and relative_parts[0] in LAYER_RULES else None


def _format_path(path: Path, source_root: Path) -> str:
    return str(Path("src") / source_root.name / path.relative_to(source_root))


def _check_import(
    *,
    path: Path,
    source_root: Path,
    node: ast.Import | ast.ImportFrom,
    target: str,
) -> str | None:
    source_layer = _layer_for(path, source_root)
    if source_layer and (diagnostic := _layer_import_diagnostic(path, source_root, node, target, source_layer)):
        return diagnostic

    relative_path = path.relative_to(source_root)
    if relative_path in {Path("application/models.py"), Path("settings/models.py")}:
        for external_module, display_name in MODEL_EXTERNAL_IMPORTS.items():
            if _matches_prefix(target, external_module):
                return (
                    f"{_format_path(path, source_root)}:{node.lineno}: architecture violation: "
                    f"{relative_path.as_posix()} cannot import {display_name} (found '{target}')"
                )
    return None


def _layer_import_diagnostic(
    path: Path,
    source_root: Path,
    node: ast.Import | ast.ImportFrom,
    target: str,
    source_layer: str,
) -> str | None:
    """Return an import violation for a recognized internal layer."""
    prefix = f"{_format_path(path, source_root)}:{node.lineno}: architecture violation: {source_layer}"
    for facade in PUBLIC_FACADE_MODULES:
        if _matches_prefix(target, f"{PROJECT_PACKAGE}.{facade}"):
            return f"{prefix} cannot import outward public facade {facade} (found '{target}')"
    for forbidden_layer in LAYER_RULES[source_layer]:
        if _matches_prefix(target, f"{PROJECT_PACKAGE}.{forbidden_layer}"):
            return f"{prefix} cannot import {forbidden_layer} (found '{target}')"
    for external_module, display_name in PURE_LAYER_EXTERNAL_IMPORTS.get(source_layer, {}).items():
        if _matches_prefix(target, external_module):
            return f"{prefix} cannot import {display_name} (found '{target}')"
    return None


def check_architecture(source_root: Path) -> list[str]:
    """Return every package-layout or import-direction violation below *source_root*."""
    source_root = source_root.resolve()
    diagnostics: list[str] = []

    for path in sorted(source_root.glob("*.py")):
        if path.name not in ALLOWED_ROOT_MODULES:
            diagnostics.append(
                f"{_format_path(path, source_root)}:1: architecture violation: "
                "production package root only permits public facades and metadata"
            )

    for path in sorted(source_root.rglob("*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except SyntaxError as error:
            line = error.lineno or 1
            diagnostics.append(f"{_format_path(path, source_root)}:{line}: syntax error: {error.msg}")
            continue

        package = _package_name(path, source_root)
        for node in ast.walk(tree):
            if not isinstance(node, (ast.Import, ast.ImportFrom)):
                continue
            for target in _import_targets(node, package):
                diagnostic = _check_import(path=path, source_root=source_root, node=node, target=target)
                if diagnostic:
                    diagnostics.append(diagnostic)
                    break

    return diagnostics


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "src" / PROJECT_PACKAGE,
        help="Python package directory to inspect (default: src/folionym)",
    )
    args = parser.parse_args()
    diagnostics = check_architecture(args.source_root)
    if diagnostics:
        print("\n".join(diagnostics))
        return 1
    print(f"Architecture check passed: {args.source_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
