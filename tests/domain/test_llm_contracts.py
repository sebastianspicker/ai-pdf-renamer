"""LLM parsing, redaction, and loopback transport behavior."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import requests

from folionym.llm.http import HttpLLMBackend
from folionym.llm.parsing import extract_and_validate_json, parse_json_field


def test_llm_json_parser_recovers_permitted_shapes_without_logging_document_content(
    caplog: pytest.LogCaptureFixture,
) -> None:
    private_text = "PRIVATE_PDF_SNIPPET_1234567890"
    assert parse_json_field('```json\n{"summary":"invoice"}\n```', key="summary") == "invoice"
    assert extract_and_validate_json(
        'prefix "keywords":[" invoice ", "2026"]',
        expected_keys={"keywords"},
        lenient_keys={"keywords"},
    ) == {"keywords": ["invoice", "2026"]}

    assert parse_json_field(f"not json {private_text}", key="summary") is None
    assert private_text not in caplog.text


def test_llm_transport_uses_chat_payload_without_proxy_or_redirects(monkeypatch: pytest.MonkeyPatch) -> None:
    backend = HttpLLMBackend(base_url="http://127.0.0.1:8080/v1/completions", model="local-model")
    response = MagicMock()
    response.json.return_value = {"choices": [{"message": {"content": "  answer  "}}]}
    response.raise_for_status.return_value = None
    calls: list[tuple[str, dict[str, object]]] = []

    def post(url: str, **kwargs: object) -> MagicMock:
        calls.append((url, kwargs))
        return response

    monkeypatch.setattr(backend.session, "post", post)
    assert (
        backend.complete("document-derived prompt", max_tokens=42, response_format={"type": "json_object"}) == "answer"
    )
    assert backend.session.trust_env is False
    assert calls[0][0] == "http://127.0.0.1:8080/v1/chat/completions"
    assert calls[0][1]["allow_redirects"] is False
    assert calls[0][1]["json"] == {
        "model": "local-model",
        "messages": [
            {"role": "system", "content": "You are a document analysis assistant. Respond only with valid JSON."},
            {"role": "user", "content": "document-derived prompt"},
        ],
        "temperature": 0.0,
        "max_tokens": 42,
        "response_format": {"type": "json_object"},
    }


def test_llm_http_errors_redact_endpoint_body_and_prompt(
    caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    secret = "provider-secret"
    backend = HttpLLMBackend(use_chat=False)
    response = MagicMock(status_code=502)

    def fail(*_args: object, **_kwargs: object) -> None:
        raise requests.HTTPError(f"endpoint?token={secret} body={secret}", response=response)

    monkeypatch.setattr(backend.session, "post", fail)
    caplog.set_level("WARNING", logger="folionym.llm.http")
    assert backend.complete(f"prompt containing {secret}") == ""
    assert secret not in caplog.text
    assert "response body and endpoint redacted" in caplog.text
