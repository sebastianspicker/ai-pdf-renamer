"""Small direct contracts for safe local renames and untrusted LLM output."""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import requests

from folionym import renamer_hooks
from folionym.llm_backend import HttpLLMBackend
from folionym.llm_parsing import extract_and_validate_json, parse_json_field
from folionym.rename_ops import RenameApplyOptions, apply_single_rename, backups, filesystem, sanitize_filename_base
from folionym.undo_cli import run_undo


def test_rename_is_previewable_no_overwrite_and_blocks_traversal(tmp_path: Path) -> None:
    source = tmp_path / "source.pdf"
    occupied = tmp_path / "invoice.pdf"
    source.write_bytes(b"pdf")
    occupied.write_bytes(b"existing")

    preview, candidate = apply_single_rename(source, "invoice", RenameApplyOptions(dry_run=True))
    assert preview and candidate.name == "invoice_1.pdf" and source.exists()

    applied, destination = apply_single_rename(source, "invoice", RenameApplyOptions())
    assert applied and destination == candidate and destination.read_bytes() == b"pdf"
    assert occupied.read_bytes() == b"existing"

    escaped = tmp_path / "again.pdf"
    escaped.write_bytes(b"pdf")
    with pytest.raises(ValueError):
        apply_single_rename(escaped, "../outside", RenameApplyOptions())


def test_filename_policy_and_llm_json_salvage_are_conservative() -> None:
    assert sanitize_filename_base("CON") == "CON_"
    assert parse_json_field('```json\n{"summary":"invoice"}\n```', key="summary") == "invoice"
    assert parse_json_field('"keywords":[" invoice ","2026"]', key="keywords", lenient=True) == [
        "invoice",
        "2026",
    ]
    assert extract_and_validate_json(
        'prefix "summary":"fallback"',
        expected_keys={"summary"},
        lenient_keys={"summary"},
    ) == {"summary": "fallback"}


def test_undo_refuses_cross_directory_log_entries(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    original = tmp_path / "original"
    renamed = tmp_path / "renamed"
    original.mkdir()
    renamed.mkdir()
    moved = renamed / "document.pdf"
    moved.write_bytes(b"pdf")
    log = tmp_path / "rename.log"
    log.write_text(f"{original / 'document.pdf'}\t{moved}\n", encoding="utf-8")

    run_undo(log, dry_run=False)

    assert moved.exists()
    assert "cross-directory undo denied" in capsys.readouterr().err


@pytest.mark.skipif(os.name == "nt", reason="POSIX symlink identity contract")
def test_rename_refuses_source_symlinks_and_preserves_dangling_targets(tmp_path: Path) -> None:
    original = tmp_path / "original.pdf"
    source_link = tmp_path / "source-link.pdf"
    original.write_bytes(b"pdf")
    source_link.symlink_to(original)
    with pytest.raises(OSError, match="Source path changed"):
        apply_single_rename(source_link, "renamed", RenameApplyOptions())
    assert source_link.is_symlink() and original.exists()
    assert not (tmp_path / "renamed.pdf").exists()

    source = tmp_path / "source.pdf"
    dangling_target = tmp_path / "invoice.pdf"
    source.write_bytes(b"pdf")
    dangling_target.symlink_to(tmp_path / "missing.pdf")
    applied, destination = apply_single_rename(source, "invoice", RenameApplyOptions())
    assert applied and destination.name == "invoice_1.pdf"
    assert dangling_target.is_symlink() and destination.read_bytes() == b"pdf"


def test_copy_failure_keeps_source_and_removes_reserved_target(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = tmp_path / "source.pdf"
    destination = tmp_path / "invoice.pdf"
    source.write_bytes(b"pdf")
    monkeypatch.setattr(filesystem, "_try_hard_link_without_overwrite", lambda *_args: None)

    def fail_copy(*_args: object) -> None:
        raise OSError("forced copy failure")

    monkeypatch.setattr(filesystem, "_copy_file_to_fd", fail_copy)
    with pytest.raises(OSError, match="forced copy failure"):
        apply_single_rename(source, "invoice", RenameApplyOptions())
    assert source.read_bytes() == b"pdf"
    assert not destination.exists()


@pytest.mark.skipif(os.name == "nt", reason="POSIX private backup directory contract")
def test_backups_keep_occupied_names_private_bytes_and_clean_failed_copies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.pdf"
    backup_dir = tmp_path / "backups"
    source.write_bytes(b"exact original bytes")
    backup_dir.mkdir(mode=0o700)
    occupied = backup_dir / source.name
    occupied.write_bytes(b"occupied bytes")

    backups._write_backup(source, backup_dir)
    created = backup_dir / "source_1.pdf"
    assert occupied.read_bytes() == b"occupied bytes"
    assert created.read_bytes() == source.read_bytes()
    assert (created.stat().st_mode & 0o777) == 0o600

    linked_dir = tmp_path / "linked-backups"
    linked_dir.symlink_to(backup_dir, target_is_directory=True)
    with pytest.raises(OSError):
        backups._write_backup(source, linked_dir)

    def fail_copy(*_args: object) -> None:
        raise OSError("forced backup copy failure")

    monkeypatch.setattr(backups, "_copy_file_to_fd", fail_copy)
    with pytest.raises(OSError, match="forced backup copy failure"):
        backups._write_backup(source, tmp_path / "failed-backups")
    assert not (tmp_path / "failed-backups" / source.name).exists()


def test_llm_http_error_logs_status_without_secret_payloads(
    caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    secret = "provider-secret"
    backend = HttpLLMBackend(use_chat=False)
    response = MagicMock(status_code=502)

    def fail_request(*_args: object, **_kwargs: object) -> None:
        raise requests.HTTPError(f"endpoint?token={secret} body={secret}", response=response)

    monkeypatch.setattr(backend.session, "post", fail_request)
    caplog.set_level("WARNING", logger="folionym.llm_backend")
    assert backend.complete(f"prompt containing {secret}") == ""
    assert secret not in caplog.text
    assert "response body and endpoint redacted" in caplog.text


def test_post_rename_hook_rejects_unsafe_urls_proxies_and_redirects(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    response = MagicMock(status_code=204)
    calls: list[tuple[str, dict[str, object]]] = []

    class FakeSession:
        trust_env = True

        def __enter__(self) -> FakeSession:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def post(self, url: str, **kwargs: object) -> MagicMock:
            calls.append((url, kwargs))
            return response

    session = FakeSession()
    monkeypatch.setattr(requests, "Session", lambda: session)
    old_path, new_path = tmp_path / "old.pdf", tmp_path / "new.pdf"
    renamer_hooks.run_post_rename_hook("https://hooks.example.test/rename", old_path, new_path, {})
    assert not session.trust_env
    assert calls[0][1]["allow_redirects"] is False
    assert calls[0][1]["timeout"] == 10

    for unsafe in ("https://user:secret@example.test/hook", "http://example.test/hook"):
        renamer_hooks.run_post_rename_hook(unsafe, old_path, new_path, {})
    assert len(calls) == 1

    response.status_code = 302
    with pytest.raises(requests.TooManyRedirects):
        renamer_hooks._post_hook_http("https://hooks.example.test/redirect", old_path, new_path, {})
