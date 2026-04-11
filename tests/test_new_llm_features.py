"""Tests for Phase 1-3 features: single-call analysis, chat API, JSON mode."""

from __future__ import annotations

import json
import re
from datetime import date
from unittest.mock import MagicMock, patch

import pytest

from ai_pdf_renamer.config import RenamerConfig
from ai_pdf_renamer.llm_backend import HttpLLMBackend, create_llm_client_from_config
from ai_pdf_renamer.llm_schema import DocumentAnalysisResult

REFERENCE_TODAY = date(2026, 4, 8)

# ---------------------------------------------------------------------------
# Phase 1: get_document_analysis (single-call)
# ---------------------------------------------------------------------------


class FakeLLMClient:
    """Fake LLM client that returns a predetermined JSON response."""

    def __init__(self, response: str) -> None:
        self._response = response
        self.calls: list[dict] = []

    @property
    def model(self) -> str:
        return "test"

    @property
    def base_url(self) -> str:
        return "http://test"

    def complete(
        self,
        prompt: str,
        *,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        response_format: dict | None = None,
    ) -> str:
        self.calls.append(
            {
                "prompt": prompt,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "response_format": response_format,
            }
        )
        return self._response

    def complete_vision(
        self, image_b64: str, prompt: str, *, model: str | None = None, timeout_s: float = 120.0
    ) -> str:
        return ""

    def close(self) -> None:
        pass


def test_get_document_analysis_parses_valid_json():
    from ai_pdf_renamer.llm import get_document_analysis

    response = json.dumps(
        {
            "summary": "This is an invoice from Amazon.",
            "keywords": ["invoice", "amazon", "payment"],
            "category": "invoice",
        }
    )
    client = FakeLLMClient(response)
    result = get_document_analysis(
        client,
        "This is a long document content about an invoice from Amazon for a payment " * 5,
        language="en",
    )
    assert isinstance(result, DocumentAnalysisResult)
    assert result.summary == "This is an invoice from Amazon."
    assert result.keywords == ("invoice", "amazon", "payment")
    assert result.category == "invoice"
    assert len(client.calls) == 1  # Single call


def test_get_document_analysis_returns_defaults_on_empty_content():
    from ai_pdf_renamer.llm import get_document_analysis

    client = FakeLLMClient("")
    result = get_document_analysis(client, "", language="en")
    assert result.summary == "na"
    assert result.category == "unknown"
    assert result.keywords == ()
    assert len(client.calls) == 0  # No call made


def test_get_document_analysis_returns_defaults_on_short_content():
    from ai_pdf_renamer.llm import get_document_analysis

    client = FakeLLMClient("")
    result = get_document_analysis(client, "short", language="en")
    assert result.summary == "na"
    assert len(client.calls) == 0


def test_get_document_analysis_handles_malformed_json():
    from ai_pdf_renamer.llm import get_document_analysis

    client = FakeLLMClient("not json at all")
    result = get_document_analysis(
        client,
        "This is a long document content " * 10,
        language="en",
    )
    assert isinstance(result, DocumentAnalysisResult)
    assert result.summary == "na"  # Fallback


def test_get_document_analysis_passes_json_mode():
    from ai_pdf_renamer.llm import get_document_analysis

    response = json.dumps({"summary": "test", "keywords": ["a"], "category": "doc"})
    client = FakeLLMClient(response)
    get_document_analysis(
        client,
        "This is a long document content " * 10,
        language="en",
        json_mode=True,
    )
    assert client.calls[0]["response_format"] == {"type": "json_object"}


def test_get_document_analysis_no_json_mode_by_default():
    from ai_pdf_renamer.llm import get_document_analysis

    response = json.dumps({"summary": "test", "keywords": ["a"], "category": "doc"})
    client = FakeLLMClient(response)
    get_document_analysis(
        client,
        "This is a long document content " * 10,
        language="en",
        json_mode=False,
    )
    assert client.calls[0]["response_format"] is None


# ---------------------------------------------------------------------------
# Phase 1: build_analysis_prompt
# ---------------------------------------------------------------------------


def test_build_analysis_prompt_de():
    from ai_pdf_renamer.llm_prompts import build_analysis_prompt

    prompt = build_analysis_prompt("de", "Ein Dokument über Steuern.")
    assert "summary" in prompt
    assert "keywords" in prompt
    assert "category" in prompt
    assert "Analysiere" in prompt
    assert "Beispiele" in prompt
    assert "Wenn mehrere Dokumenttypen" in prompt
    assert 'Generische Kategorien wie "document" oder "letter"' in prompt


def test_build_analysis_prompt_en():
    from ai_pdf_renamer.llm_prompts import build_analysis_prompt

    prompt = build_analysis_prompt("en", "A document about taxes.")
    assert "Analyze" in prompt
    assert "summary" in prompt
    assert "Examples" in prompt
    assert "If multiple document types seem possible" in prompt
    assert 'Do not return generic categories like "document" or "letter"' in prompt


def test_build_analysis_prompt_with_doc_type_hint():
    from ai_pdf_renamer.llm_prompts import build_analysis_prompt

    prompt = build_analysis_prompt("de", "text", suggested_doc_type="rechnung")
    assert "rechnung" in prompt


def test_build_analysis_prompt_with_allowed_categories():
    from ai_pdf_renamer.llm_prompts import build_analysis_prompt

    prompt = build_analysis_prompt("en", "text", allowed_categories=["invoice", "contract"])
    assert "invoice" in prompt
    assert "contract" in prompt


# ---------------------------------------------------------------------------
# Phase 2: Chat API mode
# ---------------------------------------------------------------------------


def _make_chat_response(content: str) -> MagicMock:
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.json.return_value = {"choices": [{"message": {"content": content}}]}
    return resp


def _make_completions_response(text: str) -> MagicMock:
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.json.return_value = {"choices": [{"text": text}]}
    return resp


def test_http_backend_chat_mode_sends_messages():
    backend = HttpLLMBackend(use_chat=True)
    with patch.object(backend._session, "post", return_value=_make_chat_response("result")) as mock_post:
        result = backend.complete("test prompt")
    assert result == "result"
    payload = mock_post.call_args[1]["json"]
    assert "messages" in payload
    assert len(payload["messages"]) == 2
    assert payload["messages"][0]["role"] == "system"
    assert payload["messages"][1]["role"] == "user"
    assert payload["messages"][1]["content"] == "test prompt"


def test_http_backend_text_mode_sends_prompt():
    backend = HttpLLMBackend(use_chat=False)
    with patch.object(backend._session, "post", return_value=_make_completions_response("result")) as mock_post:
        result = backend.complete("test prompt")
    assert result == "result"
    payload = mock_post.call_args[1]["json"]
    assert "prompt" in payload
    assert payload["prompt"] == "test prompt"
    assert "messages" not in payload


def test_factory_creates_chat_backend_by_default():
    config = RenamerConfig(use_llm=True)
    client = create_llm_client_from_config(config)
    assert isinstance(client, HttpLLMBackend)
    assert client.use_chat is True


def test_factory_creates_text_backend_when_chat_disabled():
    config = RenamerConfig(use_llm=True, llm_use_chat_api=False)
    client = create_llm_client_from_config(config)
    assert isinstance(client, HttpLLMBackend)
    assert client.use_chat is False


# ---------------------------------------------------------------------------
# Phase 3: JSON mode / response_format
# ---------------------------------------------------------------------------


def test_complete_json_with_retry_json_mode_reduces_retries():
    from ai_pdf_renamer.llm import complete_json_with_retry

    client = FakeLLMClient("not json")
    complete_json_with_retry(client, "prompt", json_mode=True, max_retries=5)
    # With json_mode=True, effective retries should be 1
    assert len(client.calls) == 1


def test_complete_json_with_retry_normal_mode_uses_max_retries():
    from ai_pdf_renamer.llm import complete_json_with_retry

    client = FakeLLMClient("not json")
    complete_json_with_retry(client, "prompt", json_mode=False, max_retries=3)
    assert len(client.calls) == 3


def test_complete_json_with_retry_json_mode_passes_response_format():
    from ai_pdf_renamer.llm import complete_json_with_retry

    response = json.dumps({"summary": "ok"})
    client = FakeLLMClient(response)
    complete_json_with_retry(client, "prompt", json_mode=True)
    assert client.calls[0]["response_format"] == {"type": "json_object"}


def test_http_backend_response_format_in_payload():
    """Verify response_format is included in the HTTP payload."""
    backend = HttpLLMBackend(use_chat=True)
    with patch.object(backend._session, "post", return_value=_make_chat_response('{"ok": true}')) as mock_post:
        backend.complete("prompt", response_format={"type": "json_object"})
    payload = mock_post.call_args[1]["json"]
    assert payload["response_format"] == {"type": "json_object"}


def test_http_backend_no_response_format_when_none():
    """Verify response_format is NOT included when None."""
    backend = HttpLLMBackend(use_chat=True)
    with patch.object(backend._session, "post", return_value=_make_chat_response("result")) as mock_post:
        backend.complete("prompt")
    payload = mock_post.call_args[1]["json"]
    assert "response_format" not in payload


# ---------------------------------------------------------------------------
# Phase 1: Integration with generate_filename (single-call path)
# ---------------------------------------------------------------------------


def test_generate_filename_single_call_path(monkeypatch):
    import ai_pdf_renamer.filename as filename_mod
    from ai_pdf_renamer.heuristics import HeuristicRule, HeuristicScorer
    from ai_pdf_renamer.text_utils import Stopwords

    analysis_result = DocumentAnalysisResult(
        summary="Invoice from Amazon for electronics.",
        keywords=("invoice", "amazon", "electronics"),
        category="invoice",
        final_summary_tokens=("amazon", "electronics"),
    )
    monkeypatch.setattr(filename_mod, "get_document_analysis", lambda *a, **k: analysis_result)

    scorer = HeuristicScorer(
        rules=[HeuristicRule(pattern=re.compile("invoice", re.IGNORECASE), category="invoice", score=10)]
    )

    name, _meta = filename_mod.generate_filename(
        "Invoice from Amazon 2024-01-09 for electronics purchase",
        config=RenamerConfig(language="en", desired_case="kebabCase", use_single_llm_call=True),
        llm_client=FakeLLMClient(""),
        heuristic_scorer=scorer,
        stopwords=Stopwords(words=set()),
        today=REFERENCE_TODAY,
    )
    assert name
    assert "invoice" in name.lower()


# ---------------------------------------------------------------------------
# Config fields
# ---------------------------------------------------------------------------


def test_config_has_new_fields():
    config = RenamerConfig()
    assert config.use_single_llm_call is True
    assert config.llm_use_chat_api is True
    assert config.llm_json_mode is True


def test_config_new_fields_can_be_disabled():
    config = RenamerConfig(use_single_llm_call=False, llm_use_chat_api=False, llm_json_mode=False)
    assert config.use_single_llm_call is False
    assert config.llm_use_chat_api is False
    assert config.llm_json_mode is False


# ---------------------------------------------------------------------------
# LLM Preset resolution
# ---------------------------------------------------------------------------


def test_preset_apple_silicon_defaults():
    from ai_pdf_renamer.config_resolver import build_config

    cfg = build_config({"llm_preset": "apple-silicon"})
    assert cfg.llm_model == "qwen2.5:3b"
    assert cfg.llm_base_url == "http://127.0.0.1:11434/v1/completions"
    assert cfg.max_context_chars == 120_000
    assert cfg.llm_preset == "apple-silicon"


def test_preset_gpu_defaults():
    from ai_pdf_renamer.config_resolver import build_config

    cfg = build_config({"llm_preset": "gpu"})
    assert cfg.llm_model == "qwen2.5:7b-instruct"
    assert cfg.llm_base_url == "http://127.0.0.1:11434/v1/completions"
    assert cfg.max_context_chars == 480_000
    assert cfg.llm_preset == "gpu"


def test_preset_gpu_explicit_model_overrides():
    from ai_pdf_renamer.config_resolver import build_config

    cfg = build_config({"llm_preset": "gpu", "llm_model": "custom-model"})
    assert cfg.llm_model == "custom-model"
    assert cfg.max_context_chars == 480_000


def test_no_preset_uses_apple_silicon_defaults():
    from ai_pdf_renamer.config_resolver import build_config

    cfg = build_config({})
    assert cfg.llm_model == "qwen2.5:3b"
    assert cfg.llm_base_url == "http://127.0.0.1:11434/v1/completions"
    assert cfg.max_context_chars == 120_000


def test_invalid_preset_warns_and_falls_back(caplog: pytest.LogCaptureFixture) -> None:
    from ai_pdf_renamer.config_resolver import build_config

    with caplog.at_level("WARNING", logger="ai_pdf_renamer.config_resolver"):
        cfg = build_config({"llm_preset": "typo-preset"})

    assert cfg.llm_model == "qwen2.5:3b"
    assert "Unknown llm_preset='typo-preset'" in caplog.text
    assert "'apple-silicon'" in caplog.text


def test_explicit_max_content_chars_overrides_preset():
    from ai_pdf_renamer.config_resolver import build_config

    cfg = build_config({"llm_preset": "gpu", "max_context_chars": 200_000})
    assert cfg.max_context_chars == 200_000


def test_explicit_llm_url_overrides_preset():
    from ai_pdf_renamer.config_resolver import build_config

    cfg = build_config({"llm_preset": "gpu", "llm_base_url": "http://custom:9999/v1/completions"})
    assert cfg.llm_base_url == "http://custom:9999/v1/completions"
