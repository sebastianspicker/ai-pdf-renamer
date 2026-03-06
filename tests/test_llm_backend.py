"""Tests for llm_backend: LLMClient protocol, HttpLLMBackend, factory, backward compat."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from ai_pdf_renamer.config import RenamerConfig
from ai_pdf_renamer.llm_backend import (
    HttpLLMBackend,
    LLMClient,
    LocalLLMClient,
    _chat_url_from_completions_url,
    _config_or_env,
    create_llm_client_from_config,
)

# ---------------------------------------------------------------------------
# _config_or_env
# ---------------------------------------------------------------------------


def test_config_or_env_uses_value():
    assert _config_or_env("myval", "SOME_ENV", "default") == "myval"


def test_config_or_env_falls_back_to_env(monkeypatch):
    monkeypatch.setenv("SOME_ENV", "fromenv")
    assert _config_or_env(None, "SOME_ENV", "default") == "fromenv"


def test_config_or_env_falls_back_to_default(monkeypatch):
    monkeypatch.delenv("SOME_ENV", raising=False)
    assert _config_or_env(None, "SOME_ENV", "default") == "default"


# ---------------------------------------------------------------------------
# _chat_url_from_completions_url
# ---------------------------------------------------------------------------


def test_chat_url_from_completions_url_standard():
    assert _chat_url_from_completions_url("http://127.0.0.1:8080/v1/completions") == (
        "http://127.0.0.1:8080/v1/chat/completions"
    )


def test_chat_url_from_completions_url_already_chat():
    url = "http://127.0.0.1:8080/v1/chat/completions"
    assert _chat_url_from_completions_url(url) == url


def test_chat_url_from_completions_url_arbitrary_base():
    assert _chat_url_from_completions_url("http://myhost:9000") == ("http://myhost:9000/v1/chat/completions")


# ---------------------------------------------------------------------------
# HttpLLMBackend — complete()
# ---------------------------------------------------------------------------


def _make_completions_response(text: str) -> MagicMock:
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.json.return_value = {"choices": [{"text": text}]}
    return resp


def test_http_backend_complete_ok():
    backend = HttpLLMBackend()
    with patch.object(backend._session, "post", return_value=_make_completions_response("hello")) as mock_post:
        result = backend.complete("test prompt")
    assert result == "hello"
    mock_post.assert_called_once()


def test_http_backend_complete_empty_choices():
    backend = HttpLLMBackend()
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.json.return_value = {"choices": []}
    with patch.object(backend._session, "post", return_value=resp):
        result = backend.complete("test prompt")
    assert result == ""


def test_http_backend_complete_network_error():
    import requests

    backend = HttpLLMBackend()
    with patch.object(backend._session, "post", side_effect=requests.ConnectionError("refused")):
        result = backend.complete("test prompt")
    assert result == ""


def test_http_backend_complete_bad_json():
    import json as json_mod

    backend = HttpLLMBackend()
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.json.side_effect = json_mod.JSONDecodeError("bad json", "", 0)
    with patch.object(backend._session, "post", return_value=resp):
        result = backend.complete("test prompt")
    assert result == ""


# ---------------------------------------------------------------------------
# HttpLLMBackend — complete_vision()
# ---------------------------------------------------------------------------


def _make_chat_response(content: str) -> MagicMock:
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.json.return_value = {"choices": [{"message": {"content": content}}]}
    return resp


def test_http_backend_complete_vision_ok():
    backend = HttpLLMBackend()
    with patch.object(backend._session, "post", return_value=_make_chat_response("vision result")) as mock_post:
        result = backend.complete_vision("base64data", "describe this")
    assert result == "vision result"
    # Should hit /v1/chat/completions
    url_called = mock_post.call_args[0][0]
    assert "chat/completions" in url_called


def test_http_backend_complete_vision_failure():
    import requests

    backend = HttpLLMBackend()
    with patch.object(backend._session, "post", side_effect=requests.ConnectionError("no")):
        result = backend.complete_vision("b64", "prompt")
    assert result == ""


# ---------------------------------------------------------------------------
# LocalLLMClient backward compat alias
# ---------------------------------------------------------------------------


def test_local_llm_client_is_http_backend():
    assert LocalLLMClient is HttpLLMBackend


def test_local_llm_client_instantiates():
    client = LocalLLMClient()
    assert isinstance(client, HttpLLMBackend)


# ---------------------------------------------------------------------------
# LLMClient protocol
# ---------------------------------------------------------------------------


def test_http_backend_implements_protocol():
    backend = HttpLLMBackend()
    assert isinstance(backend, LLMClient)


def test_http_backend_model_property():
    backend = HttpLLMBackend(model="mymodel")
    assert backend.model == "mymodel"


def test_http_backend_base_url_property():
    backend = HttpLLMBackend(base_url="http://localhost:9999/v1/completions")
    assert backend.base_url == "http://localhost:9999/v1/completions"


# ---------------------------------------------------------------------------
# create_llm_client_from_config — HTTP backend
# ---------------------------------------------------------------------------


def test_factory_http_backend_default():
    config = RenamerConfig(use_llm=True)
    client = create_llm_client_from_config(config)
    assert isinstance(client, HttpLLMBackend)


def test_factory_http_backend_custom_url():
    config = RenamerConfig(use_llm=True, llm_base_url="http://myserver:9000/v1/completions")
    client = create_llm_client_from_config(config)
    assert isinstance(client, HttpLLMBackend)
    assert client.base_url == "http://myserver:9000/v1/completions"


def test_factory_http_backend_env_url(monkeypatch):
    monkeypatch.setenv("AI_PDF_RENAMER_LLM_URL", "http://envserver:1234/v1/completions")
    config = RenamerConfig(use_llm=True)
    client = create_llm_client_from_config(config)
    assert isinstance(client, HttpLLMBackend)
    assert client.base_url == "http://envserver:1234/v1/completions"


def test_factory_http_backend_model_from_config():
    config = RenamerConfig(use_llm=True, llm_model="qwen3:8b")
    client = create_llm_client_from_config(config)
    assert client.model == "qwen3:8b"


def test_factory_http_backend_model_from_env(monkeypatch):
    monkeypatch.setenv("AI_PDF_RENAMER_LLM_MODEL", "mistral")
    config = RenamerConfig(use_llm=True)
    client = create_llm_client_from_config(config)
    assert client.model == "mistral"


# ---------------------------------------------------------------------------
# create_llm_client_from_config — in-process backend (import patching)
# ---------------------------------------------------------------------------


def test_factory_in_process_no_model_path_falls_back_to_http():
    config = RenamerConfig(use_llm=True, llm_backend="in-process")
    # No model_path set -> should still return HttpLLMBackend (no path to load)
    client = create_llm_client_from_config(config)
    assert isinstance(client, HttpLLMBackend)


def test_factory_auto_no_model_path_uses_http():
    config = RenamerConfig(use_llm=True, llm_backend="auto")
    client = create_llm_client_from_config(config)
    assert isinstance(client, HttpLLMBackend)


def test_factory_in_process_missing_llama_cpp_raises(tmp_path):
    model_file = tmp_path / "model.gguf"
    model_file.write_bytes(b"fake")
    config = RenamerConfig(use_llm=True, llm_backend="in-process", llm_model_path=str(model_file))
    with patch.dict("sys.modules", {"llama_cpp": None}):
        with pytest.raises(ImportError):
            create_llm_client_from_config(config)


def test_factory_auto_missing_llama_cpp_falls_back_to_http(tmp_path):
    model_file = tmp_path / "model.gguf"
    model_file.write_bytes(b"fake")
    config = RenamerConfig(use_llm=True, llm_backend="auto", llm_model_path=str(model_file))
    with patch.dict("sys.modules", {"llama_cpp": None}):
        client = create_llm_client_from_config(config)
    assert isinstance(client, HttpLLMBackend)


# ---------------------------------------------------------------------------
# close()
# ---------------------------------------------------------------------------


def test_http_backend_close():
    backend = HttpLLMBackend()
    backend.close()  # Should not raise
