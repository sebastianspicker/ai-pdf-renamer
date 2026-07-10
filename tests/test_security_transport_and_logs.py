"""Regression tests for hook transport policy and sensitive-log redaction."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import requests

from ai_pdf_renamer import llm_schema, renamer_hooks

SENSITIVE_MARKER = "DOCUMENT_MARKER_6f4fda2c"


def _successful_session() -> MagicMock:
    response = MagicMock()
    response.raise_for_status.return_value = None
    session = MagicMock()
    session.__enter__.return_value = session
    session.__exit__.return_value = False
    session.post.return_value = response
    return session


@pytest.mark.parametrize(
    "url",
    [
        "https://hooks.example.test/renamed",
        "http://127.0.0.1:8000/renamed",
        "http://[::1]:8000/renamed",
    ],
)
def test_hook_transport_policy_allows_https_and_literal_loopback_http(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    url: str,
) -> None:
    session = _successful_session()
    session_factory = MagicMock(return_value=session)
    monkeypatch.setattr(renamer_hooks.requests, "Session", session_factory)

    renamer_hooks.run_post_rename_hook(url, tmp_path / "old.pdf", tmp_path / "new.pdf", {"category": "test"})

    session_factory.assert_called_once_with()
    assert session.trust_env is False
    session.post.assert_called_once()
    assert session.post.call_args.args[0] == url
    assert session.post.call_args.kwargs["allow_redirects"] is False


@pytest.mark.parametrize(
    "url",
    [
        "http://hooks.example.test/renamed",
        "http://192.0.2.1/renamed",
        "http://localhost/renamed",
        "https://",
        "https://user:password@",
        "https://user:password@hooks.example.test/renamed",
        "https://hooks.example.test:invalid/renamed",
        "https://bad host.example/renamed",
        "file:///tmp/hook",
        "printf document",
    ],
)
def test_hook_transport_policy_rejects_unsafe_values_before_requests(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    url: str,
) -> None:
    session_factory = MagicMock()
    monkeypatch.setattr(renamer_hooks.requests, "Session", session_factory)

    renamer_hooks.run_post_rename_hook(url, tmp_path / "old.pdf", tmp_path / "new.pdf", {})

    session_factory.assert_not_called()


def test_hook_logs_never_include_url_payload_or_request_error_values(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    tmp_path: Path,
) -> None:
    session = _successful_session()
    session.post.side_effect = requests.ConnectionError(f"request failed for {SENSITIVE_MARKER}")
    monkeypatch.setattr(renamer_hooks.requests, "Session", MagicMock(return_value=session))
    old_path = tmp_path / f"old-{SENSITIVE_MARKER}.pdf"
    new_path = tmp_path / f"new-{SENSITIVE_MARKER}.pdf"

    with caplog.at_level(logging.DEBUG, logger="ai_pdf_renamer.renamer"):
        renamer_hooks.run_post_rename_hook(
            "https://hooks.example.test/renamed",
            old_path,
            new_path,
            {"summary": SENSITIVE_MARKER},
        )

    assert "Post-rename hook HTTP call failed." in caplog.text
    assert SENSITIVE_MARKER not in caplog.text
    assert str(old_path) not in caplog.text
    assert str(new_path) not in caplog.text


def test_rejected_hook_url_is_redacted_from_logs(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    tmp_path: Path,
) -> None:
    session_factory = MagicMock()
    monkeypatch.setattr(renamer_hooks.requests, "Session", session_factory)

    with caplog.at_level(logging.WARNING, logger="ai_pdf_renamer.renamer"):
        renamer_hooks.run_post_rename_hook(
            f"http://{SENSITIVE_MARKER}.example/renamed",
            tmp_path / "old.pdf",
            tmp_path / "new.pdf",
            {},
        )

    session_factory.assert_not_called()
    assert "Post-rename hook URL rejected" in caplog.text
    assert SENSITIVE_MARKER not in caplog.text


def test_schema_validation_log_omits_error_message_and_invalid_instance(
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class MarkerValidationError(Exception):
        validator = "type"
        absolute_schema_path = ("properties", "summary", "type")

    fake_jsonschema = MagicMock()
    fake_jsonschema.ValidationError = MarkerValidationError
    fake_jsonschema.validate.side_effect = MarkerValidationError(SENSITIVE_MARKER)
    monkeypatch.setitem(sys.modules, "jsonschema", fake_jsonschema)
    parsed: dict[str, object] = {
        "summary": {"private_document_text": SENSITIVE_MARKER},
        "keywords": [],
        "category": "test",
    }

    with caplog.at_level(logging.INFO, logger="ai_pdf_renamer.llm_schema"):
        result = llm_schema.validate_llm_document_result(parsed)

    assert result.summary == llm_schema.DEFAULT_LLM_SUMMARY
    fake_jsonschema.validate.assert_called_once()
    assert "validator=type" in caplog.text
    assert "schema_path=properties.<field>.type" in caplog.text
    assert SENSITIVE_MARKER not in caplog.text
    assert "private_document_text" not in caplog.text
