"""Tests for llm_backend: LLMClient protocol, HttpLLMBackend, factory, backward compat."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest

from ai_pdf_renamer.config import RenamerConfig
from ai_pdf_renamer.llm_backend import (
    HttpLLMBackend,
    InProcessLLMBackend,
    LLMClient,
    LocalLLMClient,
    _chat_url_from_completions_url,
    _config_or_env,
    _warn_if_plaintext_remote,
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


def test_http_backend_complete_text_mode_ok():
    """Legacy text completions mode (use_chat=False)."""
    backend = HttpLLMBackend(use_chat=False)
    with patch.object(backend._session, "post", return_value=_make_completions_response("hello")) as mock_post:
        result = backend.complete("test prompt")
    assert result == "hello"
    mock_post.assert_called_once()


def test_http_backend_complete_chat_mode_ok():
    """Default chat completions mode (use_chat=True)."""
    backend = HttpLLMBackend(use_chat=True)
    with patch.object(backend._session, "post", return_value=_make_chat_response("hello chat")) as mock_post:
        result = backend.complete("test prompt")
    assert result == "hello chat"
    url_called = mock_post.call_args[0][0]
    assert "chat/completions" in url_called


def test_http_backend_complete_chat_mode_with_response_format():
    """Chat mode with response_format parameter."""
    backend = HttpLLMBackend(use_chat=True)
    with patch.object(backend._session, "post", return_value=_make_chat_response('{"summary":"test"}')) as mock_post:
        result = backend.complete("test prompt", response_format={"type": "json_object"})
    assert '"summary"' in result
    payload = mock_post.call_args[1]["json"]
    assert payload["response_format"] == {"type": "json_object"}


def test_http_backend_complete_empty_choices():
    backend = HttpLLMBackend(use_chat=False)
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


def test_http_complete_vision_success():
    """Mock requests.Session.post to return valid chat completion JSON with image content."""
    backend = HttpLLMBackend()
    mock_resp = _make_chat_response("This document is an invoice dated 2024-01-15.")
    with patch.object(backend._session, "post", return_value=mock_resp) as mock_post:
        result = backend.complete_vision("aW1hZ2VkYXRh", "Describe the contents of this PDF page.")
    assert result == "This document is an invoice dated 2024-01-15."
    # Verify the payload structure contains image_url content
    payload = mock_post.call_args[1]["json"]
    user_msg = payload["messages"][0]
    assert user_msg["role"] == "user"
    assert any(part["type"] == "image_url" for part in user_msg["content"])
    assert any(part["type"] == "text" for part in user_msg["content"])


def test_http_complete_vision_failure_returns_empty():
    """Mock requests.Session.post to raise requests.ConnectionError. Verify returns ''."""
    import requests as req_mod

    backend = HttpLLMBackend()
    with patch.object(backend._session, "post", side_effect=req_mod.ConnectionError("Connection refused")):
        result = backend.complete_vision("b64data", "describe this")
    assert result == ""


def test_http_complete_vision_invalid_json():
    """Mock response with invalid JSON body. Verify returns ''."""
    import json as json_mod

    backend = HttpLLMBackend()
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.json.side_effect = json_mod.JSONDecodeError("Expecting value", "", 0)
    with patch.object(backend._session, "post", return_value=resp):
        result = backend.complete_vision("b64data", "describe this")
    assert result == ""


def test_http_complete_vision_uses_model_override():
    """Verify that model='llava' param is passed through in the payload."""
    backend = HttpLLMBackend()
    with patch.object(backend._session, "post", return_value=_make_chat_response("ok")) as mock_post:
        backend.complete_vision("b64data", "describe this", model="llava")
    payload = mock_post.call_args[1]["json"]
    assert payload["model"] == "llava"


def test_http_complete_vision_timeout_passthrough():
    """Verify timeout_s is passed to session.post."""
    backend = HttpLLMBackend()
    with patch.object(backend._session, "post", return_value=_make_chat_response("ok")) as mock_post:
        backend.complete_vision("b64data", "describe this", timeout_s=42.5)
    assert mock_post.call_args[1]["timeout"] == 42.5


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
    with patch.dict("sys.modules", {"llama_cpp": None}), pytest.raises(ImportError):
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


# ---------------------------------------------------------------------------
# InProcessLLMBackend (mock llama_cpp)
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_llama_cpp(monkeypatch: pytest.MonkeyPatch) -> tuple[MagicMock, MagicMock]:
    """Inject a mock llama_cpp module and return (mock_module, mock_llama_instance)."""
    mock_module = MagicMock()
    mock_instance = MagicMock()
    mock_module.Llama.return_value = mock_instance
    monkeypatch.setitem(sys.modules, "llama_cpp", mock_module)
    return mock_module, mock_instance


def test_in_process_backend_init(mock_llama_cpp: tuple[MagicMock, MagicMock]) -> None:
    """Mock llama_cpp.Llama, verify model loaded."""
    mock_module, mock_instance = mock_llama_cpp
    backend = InProcessLLMBackend("/tmp/model.gguf")
    mock_module.Llama.assert_called_once_with(model_path="/tmp/model.gguf", verbose=False)
    assert backend._llama is mock_instance


def test_in_process_backend_complete_chat(mock_llama_cpp: tuple[MagicMock, MagicMock]) -> None:
    """use_chat=True, mock create_chat_completion. Verify text extracted from chat response."""
    _mock_module, mock_instance = mock_llama_cpp
    mock_instance.create_chat_completion.return_value = {
        "choices": [{"message": {"content": "chat reply"}}],
    }
    backend = InProcessLLMBackend("/tmp/model.gguf", use_chat=True)
    result = backend.complete("test prompt")
    assert result == "chat reply"
    mock_instance.create_chat_completion.assert_called_once()


def test_in_process_backend_complete_text(mock_llama_cpp: tuple[MagicMock, MagicMock]) -> None:
    """use_chat=False, mock create_completion. Verify text extracted."""
    _mock_module, mock_instance = mock_llama_cpp
    mock_instance.create_completion.return_value = {
        "choices": [{"text": "text reply"}],
    }
    backend = InProcessLLMBackend("/tmp/model.gguf", use_chat=False)
    result = backend.complete("test prompt")
    assert result == "text reply"
    mock_instance.create_completion.assert_called_once()


def test_in_process_backend_complete_error(mock_llama_cpp: tuple[MagicMock, MagicMock]) -> None:
    """Mock create_chat_completion to raise RuntimeError. Verify returns ''."""
    _mock_module, mock_instance = mock_llama_cpp
    mock_instance.create_chat_completion.side_effect = RuntimeError("model crash")
    backend = InProcessLLMBackend("/tmp/model.gguf", use_chat=True)
    result = backend.complete("test prompt")
    assert result == ""


def test_in_process_backend_complete_vision(mock_llama_cpp: tuple[MagicMock, MagicMock]) -> None:
    """Mock create_chat_completion with image. Verify text returned."""
    _mock_module, mock_instance = mock_llama_cpp
    mock_instance.create_chat_completion.return_value = {
        "choices": [{"message": {"content": "I see a document"}}],
    }
    backend = InProcessLLMBackend("/tmp/model.gguf")
    result = backend.complete_vision("aW1hZ2VkYXRh", "Describe this image")
    assert result == "I see a document"
    call_kwargs = mock_instance.create_chat_completion.call_args
    messages = call_kwargs[1].get("messages") or call_kwargs[0][0]
    # The user message should contain image_url content
    user_msg = messages[0]
    assert any(part["type"] == "image_url" for part in user_msg["content"])


def test_in_process_backend_close(mock_llama_cpp: tuple[MagicMock, MagicMock]) -> None:
    """Verify del self._llama called."""
    _mock_module, _mock_instance = mock_llama_cpp
    backend = InProcessLLMBackend("/tmp/model.gguf")
    assert hasattr(backend, "_llama")
    backend.close()
    assert not hasattr(backend, "_llama")


def test_in_process_backend_missing_llama_cpp(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mock llama_cpp import to fail. Verify ImportError raised."""
    monkeypatch.setitem(sys.modules, "llama_cpp", None)
    with pytest.raises(ImportError, match="llama-cpp-python"):
        InProcessLLMBackend("/tmp/model.gguf")


def test_in_process_backend_properties(mock_llama_cpp: tuple[MagicMock, MagicMock]) -> None:
    """Verify model and base_url properties."""
    _mock_module, _mock_instance = mock_llama_cpp
    backend = InProcessLLMBackend("/tmp/model.gguf")
    assert backend.model == "/tmp/model.gguf"
    assert backend.base_url == "file:///tmp/model.gguf"


# ---------------------------------------------------------------------------
# _warn_if_plaintext_remote
# ---------------------------------------------------------------------------


def test_warn_plaintext_remote_http_external_enforce() -> None:
    """enforce=True with http://example.com raises ValueError."""
    with pytest.raises(ValueError, match="plain HTTP"):
        _warn_if_plaintext_remote("http://example.com/v1/completions", enforce=True)


def test_warn_plaintext_remote_http_localhost_ok() -> None:
    """http://127.0.0.1 does not raise even with enforce=True."""
    _warn_if_plaintext_remote("http://127.0.0.1:8080/v1/completions", enforce=True)


def test_warn_plaintext_remote_https_ok() -> None:
    """https:// does not warn or raise."""
    _warn_if_plaintext_remote("https://example.com/v1/completions", enforce=True)


def test_warn_plaintext_remote_empty_url() -> None:
    """Empty string does not error."""
    _warn_if_plaintext_remote("", enforce=True)


# ---------------------------------------------------------------------------
# Factory edge cases
# ---------------------------------------------------------------------------


def test_factory_timeout_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """AI_PDF_RENAMER_LLM_TIMEOUT env var used when config.llm_timeout_s is None."""
    monkeypatch.setenv("AI_PDF_RENAMER_LLM_TIMEOUT", "42.5")
    config = RenamerConfig(use_llm=True, llm_timeout_s=None)
    client = create_llm_client_from_config(config)
    assert isinstance(client, HttpLLMBackend)
    assert client.timeout_s == 42.5


def test_factory_in_process_fallback_to_http(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: object,
) -> None:
    """backend='auto', model_path set but llama_cpp unavailable -> falls back to HTTP."""
    from pathlib import Path

    tmp = Path(str(tmp_path))
    model_file = tmp / "model.gguf"
    model_file.write_bytes(b"fake")
    monkeypatch.setitem(sys.modules, "llama_cpp", None)
    config = RenamerConfig(use_llm=True, llm_backend="auto", llm_model_path=str(model_file))
    client = create_llm_client_from_config(config)
    assert isinstance(client, HttpLLMBackend)
