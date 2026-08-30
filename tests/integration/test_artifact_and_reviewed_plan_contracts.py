"""Observable artifact and browser reviewed-plan contracts."""

from __future__ import annotations

import json
from pathlib import Path

import folionym.application.reviewed_plan as reviewed_plan
import folionym.renamer as renamer
from folionym.application.models import Proposal
from folionym.application.reviewed_plan import ApplyStatus, apply_reviewed_plan, create_preview_plan
from folionym.config import build_config


def test_plan_mode_and_real_rename_preserve_their_distinct_artifact_contracts(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "source.pdf"
    source.write_bytes(b"pdf")
    plan_path = tmp_path / "plan.json"
    metadata_path = tmp_path / "metadata.json"
    summary_path = tmp_path / "summary.json"
    log_path = tmp_path / "rename.log"
    plan_config = build_config(
        {
            "use_llm": False,
            "dry_run": False,
            "plan_file_path": str(plan_path),
        }
    )
    metadata = {
        "category": "invoice",
        "summary": "August invoice",
        "keywords": ["august", "invoice"],
        "category_source": "heuristic",
        "llm_failed": False,
        "used_vision_fallback": False,
        "invoice_id": "INV-1",
        "amount": "10.00",
        "company": "Example",
    }
    monkeypatch.setattr(
        renamer,
        "produce_rename_results",
        lambda files, _config, **_kwargs: [(files[0], "approved-invoice", metadata, None)],
    )

    planned = renamer.rename_pdfs_in_directory(tmp_path, config=plan_config)

    target = tmp_path / "approved-invoice.pdf"
    assert planned == {target}
    assert source.exists() and not target.exists()
    assert json.loads(plan_path.read_text(encoding="utf-8")) == [{"old": str(source), "new": str(target)}]

    apply_config = build_config(
        {
            "use_llm": False,
            "dry_run": False,
            "export_metadata_path": str(metadata_path),
            "summary_json_path": str(summary_path),
            "rename_log_path": str(log_path),
        }
    )
    renamed = renamer.rename_pdfs_in_directory(tmp_path, config=apply_config)

    assert renamed == {target}
    exported = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert exported == [{"path": str(source), "new_name": target.name, **metadata}]
    assert json.loads(summary_path.read_text(encoding="utf-8")) == {
        "directory": str(tmp_path),
        "processed": 1,
        "renamed": 1,
        "skipped": 0,
        "failed": 0,
        "dry_run": False,
        "failures": [],
    }
    assert log_path.read_text(encoding="utf-8") == f"{source}\t{target}\n"


def _preview_plan_with_proposals(tmp_path: Path, monkeypatch, proposals: list[str]):
    sources = []
    for index, _proposal in enumerate(proposals, start=1):
        source = tmp_path / f"source-{index}.pdf"
        source.write_bytes(b"pdf")
        sources.append(source)
    config = build_config({"use_llm": False, "dry_run": True})
    monkeypatch.setattr(
        reviewed_plan,
        "produce_proposals",
        lambda files, _config, **_kwargs: [
            Proposal(path, proposal, {"category": "invoice"}) for path, proposal in zip(files, proposals, strict=True)
        ],
    )
    return create_preview_plan(tmp_path, config)


def test_reviewed_plan_rejects_a_source_that_changed_after_preview(tmp_path: Path, monkeypatch) -> None:
    plan = _preview_plan_with_proposals(tmp_path, monkeypatch, ["approved"])
    plan.items[0].source.write_bytes(b"changed pdf bytes")

    report = apply_reviewed_plan(plan, [plan.items[0].id])

    assert report.items[0].status is ApplyStatus.FAILED
    assert report.items[0].reason == "The source changed after Preview."
    assert plan.items[0].source.exists()
    assert not (tmp_path / "approved.pdf").exists()


def test_reviewed_plan_rejects_each_selected_exact_target_collision(tmp_path: Path, monkeypatch) -> None:
    plan = _preview_plan_with_proposals(tmp_path, monkeypatch, ["approved", "approved"])

    report = apply_reviewed_plan(plan, [item.id for item in plan.items])

    assert [item.status for item in report.items] == [ApplyStatus.FAILED, ApplyStatus.FAILED]
    assert [item.reason for item in report.items] == [
        "Another selected item has the same reviewed target.",
        "Another selected item has the same reviewed target.",
    ]
    assert all(item.source.exists() for item in plan.items)
    assert not (tmp_path / "approved.pdf").exists()


def test_reviewed_plan_refuses_an_occupied_exact_target_without_selecting_a_suffix(tmp_path: Path, monkeypatch) -> None:
    plan = _preview_plan_with_proposals(tmp_path, monkeypatch, ["approved"])
    occupied = tmp_path / "approved.pdf"
    occupied.write_bytes(b"occupied")

    report = apply_reviewed_plan(plan, [plan.items[0].id])

    assert report.items[0].status is ApplyStatus.FAILED
    assert report.items[0].reason == "The reviewed target already exists."
    assert occupied.read_bytes() == b"occupied"
    assert plan.items[0].source.exists()
    assert not (tmp_path / "approved_1.pdf").exists()
