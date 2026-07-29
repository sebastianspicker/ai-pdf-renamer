"""Exercise the real CLI and undo workflow across dry-run and apply boundaries."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


def _clean_env(tmp_path: Path) -> dict[str, str]:
    env = {key: value for key, value in os.environ.items() if not key.startswith("FOLIONYM_")}
    env["PYTHONPATH"] = str(REPO_ROOT / "src")
    env["FOLIONYM_LOG_FILE"] = str(tmp_path / "runtime.log")
    env["FOLIONYM_CACHE_DIR"] = str(tmp_path / "cache")
    env["NO_COLOR"] = "1"
    return env


@dataclass(frozen=True)
class CliResult:
    returncode: int
    stdout: str
    stderr: str


@dataclass(frozen=True)
class ApplyOutputPaths:
    summary: Path
    metadata: Path
    rename_log: Path


def _run_cli(
    name: str,
    args: Sequence[str],
    *,
    cwd: Path,
    env: Mapping[str, str],
) -> CliResult:
    executable = shutil.which(name)
    assert executable is not None, f"Installed CLI entry point is missing: {name}"
    result = subprocess.run(
        [executable, *args],
        cwd=cwd,
        env=env,
        check=False,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
    )
    return CliResult(returncode=result.returncode, stdout=result.stdout, stderr=result.stderr)


def _write_pdf(path: Path, text: str) -> None:
    fitz = None
    try:
        import fitz
    except ImportError:
        pytest.skip("PyMuPDF is required for CLI E2E tests; install with the [pdf] extra.")

    if fitz is None:
        pytest.skip("PyMuPDF is required for CLI E2E tests; install with the [pdf] extra.")

    doc = fitz.open()
    try:
        page = doc.new_page()
        page.insert_text((72, 72), text, fontsize=11)
        doc.save(path)
    finally:
        doc.close()


def _write_invoice_fixture(pdf_dir: Path) -> Path:
    original = pdf_dir / "incoming.pdf"
    _write_pdf(
        original,
        "Invoice Date: 2024-03-14\n"
        "Invoice number: INV-2024-0314\n"
        "Total amount: 42.00 EUR\n"
        "This deterministic fixture exercises the local heuristic path.",
    )
    return original


def _run_dry_run_phase(tmp_path: Path, pdf_dir: Path, original: Path, env: dict[str, str]) -> str:
    plan_path = tmp_path / "plan.json"
    dry_summary_path = tmp_path / "dry-summary.json"
    dry_run = _run_cli(
        "folionym",
        [
            "--dir",
            str(pdf_dir),
            "--no-llm",
            "--no-cache",
            "--dry-run",
            "--plan-file",
            str(plan_path),
            "--summary-json",
            str(dry_summary_path),
        ],
        cwd=tmp_path,
        env=env,
    )
    assert dry_run.returncode == 0, dry_run.stderr
    assert original.exists()
    _assert_dry_run_outputs(pdf_dir, original, plan_path, dry_summary_path)
    return "20240314-invoice.pdf"


def _assert_dry_run_outputs(pdf_dir: Path, original: Path, plan_path: Path, dry_summary_path: Path) -> None:
    dry_summary = json.loads(dry_summary_path.read_text(encoding="utf-8"))
    assert dry_summary == {
        "directory": str(pdf_dir),
        "processed": 1,
        "renamed": 1,
        "skipped": 0,
        "failed": 0,
        "dry_run": True,
        "failures": [],
    }
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    assert len(plan) == 1
    assert plan[0]["old"] == str(original)
    assert Path(plan[0]["new"]).name == "20240314-invoice.pdf"


def _run_apply_phase(
    tmp_path: Path,
    pdf_dir: Path,
    original: Path,
    expected_name: str,
    env: dict[str, str],
) -> tuple[Path, Path]:
    apply_summary_path = tmp_path / "apply-summary.json"
    metadata_path = tmp_path / "metadata.json"
    rename_log_path = tmp_path / "rename.log"
    output_paths = ApplyOutputPaths(apply_summary_path, metadata_path, rename_log_path)
    apply = _run_cli(
        "folionym",
        [
            "--dir",
            str(pdf_dir),
            "--no-llm",
            "--no-cache",
            "--rename-log",
            str(rename_log_path),
            "--export-metadata",
            str(metadata_path),
            "--summary-json",
            str(apply_summary_path),
        ],
        cwd=tmp_path,
        env=env,
    )
    assert apply.returncode == 0, apply.stderr
    renamed = pdf_dir / expected_name
    _assert_apply_outputs(original, renamed, expected_name, output_paths)
    return (renamed, rename_log_path)


def _assert_apply_outputs(
    original: Path,
    renamed: Path,
    expected_name: str,
    output_paths: ApplyOutputPaths,
) -> None:
    assert renamed.exists()
    assert not original.exists()
    apply_summary = json.loads(output_paths.summary.read_text(encoding="utf-8"))
    assert apply_summary["processed"] == 1
    assert apply_summary["renamed"] == 1
    assert apply_summary["failed"] == 0
    assert apply_summary["dry_run"] is False
    metadata = json.loads(output_paths.metadata.read_text(encoding="utf-8"))
    assert metadata[0]["new_name"] == expected_name
    assert metadata[0]["category"] == "invoice"
    assert output_paths.rename_log.read_text(encoding="utf-8").strip() == f"{original}\t{renamed}"


def _run_undo_phase(tmp_path: Path, original: Path, renamed: Path, rename_log_path: Path, env: dict[str, str]) -> None:
    undo = _run_cli(
        "folionym-undo",
        ["--rename-log", str(rename_log_path)],
        cwd=tmp_path,
        env=env,
    )
    assert undo.returncode == 0, undo.stderr
    assert original.exists()
    assert not renamed.exists()
    assert "Reverted:" in undo.stdout


def test_cli_dry_run_apply_and_undo_round_trip(tmp_path: Path) -> None:
    pdf_dir = tmp_path / "pdfs"
    pdf_dir.mkdir()
    original = _write_invoice_fixture(pdf_dir)
    env = _clean_env(tmp_path)
    expected_name = _run_dry_run_phase(tmp_path, pdf_dir, original, env)
    renamed, rename_log_path = _run_apply_phase(tmp_path, pdf_dir, original, expected_name, env)
    _run_undo_phase(tmp_path, original, renamed, rename_log_path, env)


def test_cli_validate_config_accepts_local_heuristic_run_defaults(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "language": "en",
                "desired_case": "kebabCase",
                "use_llm": False,
                "dry_run": True,
            }
        ),
        encoding="utf-8",
    )

    result = _run_cli(
        "folionym",
        [
            "--validate-config",
            "--config",
            str(config_path),
            "--dir",
            str(tmp_path),
        ],
        cwd=tmp_path,
        env=_clean_env(tmp_path),
    )

    assert result.returncode == 0, result.stderr
    assert "Configuration valid." in result.stderr
