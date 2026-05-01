from __future__ import annotations

import json
import os
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


def _clean_env(tmp_path: Path) -> dict[str, str]:
    env = {key: value for key, value in os.environ.items() if not key.startswith("AI_PDF_RENAMER_")}
    env["PYTHONPATH"] = str(REPO_ROOT / "src")
    env["AI_PDF_RENAMER_LOG_FILE"] = str(tmp_path / "runtime.log")
    env["AI_PDF_RENAMER_CACHE_DIR"] = str(tmp_path / "cache")
    env["NO_COLOR"] = "1"
    return env


def _entrypoint(name: str) -> list[str]:
    sibling_script = Path(sys.executable).with_name(name)
    if sibling_script.exists():
        return [str(sibling_script)]
    if name == "ai-pdf-renamer":
        return [sys.executable, "-m", "ai_pdf_renamer.cli"]
    if name == "ai-pdf-renamer-undo":
        return [sys.executable, "-c", "from ai_pdf_renamer.undo_cli import main; main()"]
    pytest.fail(f"Missing CLI entry point: {name}")
    raise AssertionError(f"Unreachable: missing CLI entry point: {name}")


def _run_cli(
    name: str,
    args: Sequence[str],
    *,
    cwd: Path,
    env: Mapping[str, str],
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [*_entrypoint(name), *args],
        cwd=cwd,
        env=dict(env),
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )


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


def test_cli_dry_run_apply_and_undo_round_trip(tmp_path: Path) -> None:
    pdf_dir = tmp_path / "pdfs"
    pdf_dir.mkdir()
    original = pdf_dir / "incoming.pdf"
    _write_pdf(
        original,
        "Invoice Date: 2024-03-14\n"
        "Invoice number: INV-2024-0314\n"
        "Total amount: 42.00 EUR\n"
        "This deterministic fixture exercises the local heuristic path.",
    )

    env = _clean_env(tmp_path)
    plan_path = tmp_path / "plan.json"
    dry_summary_path = tmp_path / "dry-summary.json"
    apply_summary_path = tmp_path / "apply-summary.json"
    metadata_path = tmp_path / "metadata.json"
    rename_log_path = tmp_path / "rename.log"

    dry_run = _run_cli(
        "ai-pdf-renamer",
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
    expected_name = "20240314-invoice.pdf"
    assert plan[0]["old"] == str(original)
    assert plan[0]["new"].endswith(f"/{expected_name}")

    apply = _run_cli(
        "ai-pdf-renamer",
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
    assert renamed.exists()
    assert not original.exists()
    apply_summary = json.loads(apply_summary_path.read_text(encoding="utf-8"))
    assert apply_summary["processed"] == 1
    assert apply_summary["renamed"] == 1
    assert apply_summary["failed"] == 0
    assert apply_summary["dry_run"] is False
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata[0]["new_name"] == expected_name
    assert metadata[0]["category"] == "invoice"
    assert rename_log_path.read_text(encoding="utf-8").strip() == f"{original}\t{renamed}"

    undo = _run_cli(
        "ai-pdf-renamer-undo",
        ["--rename-log", str(rename_log_path)],
        cwd=tmp_path,
        env=env,
    )

    assert undo.returncode == 0, undo.stderr
    assert original.exists()
    assert not renamed.exists()
    assert "Reverted:" in undo.stdout


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
        "ai-pdf-renamer",
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
