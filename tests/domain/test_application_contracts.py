"""Focused contracts for the application orchestration package."""

from __future__ import annotations

import csv
from pathlib import Path

import folionym.application.batch as batch
from folionym.application.artifacts import _write_json_or_csv
from folionym.application.models import ApplyPolicy, PreviewStatus, Proposal
from folionym.application.scheduling import (
    ProposalProductionDependencies,
    ProposalProductionRequest,
    produce_proposals_with,
)
from folionym.config import RenamerConfig


def test_typed_proposal_statuses_preserve_empty_and_error_distinctions(tmp_path: Path) -> None:
    source = tmp_path / "source.pdf"

    assert Proposal(source, "invoice", {}).status is PreviewStatus.READY
    assert Proposal(source, None, None).status is PreviewStatus.SKIPPED
    assert Proposal(source, None, None, ValueError("bad PDF")).status is PreviewStatus.FAILED


def test_scheduler_preserves_input_order_and_stops_before_sequential_work(tmp_path: Path) -> None:
    files = [tmp_path / f"{index}.pdf" for index in range(3)]
    config = RenamerConfig()
    completed: list[Path] = []

    def process(path: Path, _config: RenamerConfig, _rules: object, _client: object) -> Proposal:
        completed.append(path)
        return Proposal(path, f"renamed-{path.stem}", {})

    dependencies = ProposalProductionDependencies(
        process_one_file=process,
        stop_requested=lambda _config: False,
        recoverable_exceptions=(ValueError,),
        logger=batch.logger,
    )
    request = ProposalProductionRequest(files, config, None, None, 2, None)

    proposals = produce_proposals_with(request, dependencies)

    assert [proposal.source for proposal in proposals] == files
    assert set(completed) == set(files)

    stopped = produce_proposals_with(
        ProposalProductionRequest(files, config, None, None, 1, None),
        ProposalProductionDependencies(
            process_one_file=process,
            stop_requested=lambda _config: True,
            recoverable_exceptions=(ValueError,),
            logger=batch.logger,
        ),
    )
    assert stopped == []


def test_apply_policy_keeps_exact_reviewed_distinct_from_unique_available(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "source.pdf"
    observed: list[bool] = []

    def apply(_path: Path, _base: str, options: batch.RenameApplyOptions) -> tuple[bool, Path]:
        observed.append(options.exact_target)
        return (True, source.with_name("target.pdf"))

    monkeypatch.setattr(batch, "apply_single_rename", apply)
    options = batch.RenameApplyOptions()
    batch.apply_rename_with_policy(source, "target", options, ApplyPolicy.UNIQUE_AVAILABLE)
    batch.apply_rename_with_policy(source, "target", options, ApplyPolicy.EXACT_REVIEWED)

    assert observed == [False, True]


def test_artifact_json_and_csv_schemas_remain_machine_readable(tmp_path: Path) -> None:
    rows = [{"path": "/tmp/source.pdf", "new_name": "=formula.pdf"}]
    json_path = tmp_path / "metadata.json"
    csv_path = tmp_path / "metadata.csv"

    _write_json_or_csv(json_path, rows, ["path", "new_name"])
    _write_json_or_csv(csv_path, rows, ["path", "new_name"])

    assert (
        json_path.read_text(encoding="utf-8")
        == '[\n  {\n    "path": "/tmp/source.pdf",\n    "new_name": "=formula.pdf"\n  }\n]'
    )
    with csv_path.open(newline="", encoding="utf-8") as handle:
        assert list(csv.DictReader(handle)) == [{"path": "/tmp/source.pdf", "new_name": "'=formula.pdf"}]
