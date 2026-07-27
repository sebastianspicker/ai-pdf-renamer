"""Tests for the HTTP-only LLM backend and its configuration factory."""

from __future__ import annotations

import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock, patch

import pytest

from folionym.config import build_config_from_flat_dict
from folionym.config_resolver import build_config
from folionym.llm_backend import (
    HttpLLMBackend,
    LLMClient,
    SerializedLLMClient,
    VisionCompletionOptions,
    _chat_url_from_completions_url,
    _config_or_env,
    _warn_if_plaintext_remote,
    create_llm_client_from_config,
)


def _config(**values: object):
    return build_config_from_flat_dict(values)


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


def test_chat_url_from_completions_url_preserves_query_after_rewriting_path():
    assert _chat_url_from_completions_url("https://myhost/v1/completions?api-version=2026-07-01") == (
        "https://myhost/v1/chat/completions?api-version=2026-07-01"
    )


# ---------------------------------------------------------------------------
# HttpLLMBackend: complete()
# ---------------------------------------------------------------------------


def _make_completions_response(text: str) -> MagicMock:
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.json.return_value = {"choices": [{"text": text}]}
    return resp


def test_http_backend_complete_text_mode_ok():
    """Legacy text completions mode (use_chat=False)."""
    backend = HttpLLMBackend(use_chat=False)
    with patch.object(backend.session, "post", return_value=_make_completions_response("hello")) as mock_post:
        result = backend.complete("test prompt")
    assert result == "hello"
    mock_post.assert_called_once()
    assert mock_post.call_args.kwargs["allow_redirects"] is False


def test_http_backend_complete_chat_mode_ok():
    """Default chat completions mode (use_chat=True)."""
    backend = HttpLLMBackend(use_chat=True)
    with patch.object(backend.session, "post", return_value=_make_chat_response("hello chat")) as mock_post:
        result = backend.complete("test prompt")
    assert result == "hello chat"
    url_called = mock_post.call_args[0][0]
    assert "chat/completions" in url_called
    assert mock_post.call_args.kwargs["allow_redirects"] is False


def test_http_backend_complete_chat_mode_with_response_format():
    """Chat mode with response_format parameter."""
    backend = HttpLLMBackend(use_chat=True)
    with patch.object(backend.session, "post", return_value=_make_chat_response('{"summary":"test"}')) as mock_post:
        result = backend.complete("test prompt", response_format={"type": "json_object"})
    assert '"summary"' in result
    payload = mock_post.call_args[1]["json"]
    assert payload["response_format"] == {"type": "json_object"}


def test_http_backend_complete_empty_choices():
    backend = HttpLLMBackend(use_chat=False)
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.json.return_value = {"choices": []}
    with patch.object(backend.session, "post", return_value=resp):
        result = backend.complete("test prompt")
    assert result == ""


def test_http_backend_complete_network_error():
    import requests

    backend = HttpLLMBackend()
    with patch.object(backend.session, "post", side_effect=requests.ConnectionError("refused")):
        result = backend.complete("test prompt")
    assert result == ""


def test_http_backend_complete_bad_json():
    import json as json_mod

    backend = HttpLLMBackend()
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.json.side_effect = json_mod.JSONDecodeError("bad json", "", 0)
    with patch.object(backend.session, "post", return_value=resp):
        result = backend.complete("test prompt")
    assert result == ""


def test_http_backend_http_error_does_not_log_response_body(caplog: pytest.LogCaptureFixture) -> None:
    import requests

    backend = HttpLLMBackend()
    sentinel = "PDF_SECRET_SENTINEL_SHOULD_NOT_REACH_LOGS"
    response = MagicMock()
    response.status_code = 500
    response.text = f"server echoed prompt fragment: {sentinel}"
    response.raise_for_status.side_effect = requests.HTTPError(sentinel, response=response)

    with (
        patch.object(backend.session, "post", return_value=response),
        caplog.at_level(logging.WARNING, logger="folionym.llm_backend"),
    ):
        result = backend.complete(f"Summarize this PDF text: {sentinel}")

    assert result == ""
    assert "status=500" in caplog.text
    assert sentinel not in caplog.text


# ---------------------------------------------------------------------------
# HttpLLMBackend: complete_vision()
# ---------------------------------------------------------------------------


def _make_chat_response(content: str) -> MagicMock:
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.json.return_value = {"choices": [{"message": {"content": content}}]}
    return resp


def test_http_backend_complete_vision_ok():
    backend = HttpLLMBackend()
    with patch.object(backend.session, "post", return_value=_make_chat_response("vision result")) as mock_post:
        result = backend.complete_vision("base64data", "describe this")
    assert result == "vision result"
    # Should hit /v1/chat/completions
    url_called = mock_post.call_args[0][0]
    assert "chat/completions" in url_called
    assert mock_post.call_args.kwargs["allow_redirects"] is False


def test_http_backend_complete_vision_failure():
    import requests

    backend = HttpLLMBackend()
    with patch.object(backend.session, "post", side_effect=requests.ConnectionError("no")):
        result = backend.complete_vision("b64", "prompt")
    assert result == ""


def test_http_complete_vision_success():
    """Mock requests.Session.post to return valid chat completion JSON with image content."""
    backend = HttpLLMBackend()
    mock_resp = _make_chat_response("This document is an invoice dated 2024-01-15.")
    with patch.object(backend.session, "post", return_value=mock_resp) as mock_post:
        result = backend.complete_vision("aW1hZ2VkYXRh", "Describe the contents of this PDF page.")
    assert result == "This document is an invoice dated 2024-01-15."
    # Verify the payload structure contains image_url content
    payload = mock_post.call_args[1]["json"]
    user_msg = payload["messages"][0]
    assert user_msg["role"] == "user"
    assert any(part["type"] == "image_url" for part in user_msg["content"])
    assert any(part["type"] == "text" for part in user_msg["content"])


def test_http_complete_vision_uses_png_data_url_mime() -> None:
    backend = HttpLLMBackend()
    with patch.object(backend.session, "post", return_value=_make_chat_response("ok")) as mock_post:
        backend.complete_vision(
            "pngdata",
            "describe this",
            VisionCompletionOptions(image_mime_type="image/png"),
        )

    payload = mock_post.call_args[1]["json"]
    image_part = next(part for part in payload["messages"][0]["content"] if part["type"] == "image_url")
    assert image_part["image_url"]["url"] == "data:image/png;base64,pngdata"


def test_http_complete_vision_failure_returns_empty():
    """Mock requests.Session.post to raise requests.ConnectionError. Verify returns ''."""
    import requests as req_mod

    backend = HttpLLMBackend()
    with patch.object(backend.session, "post", side_effect=req_mod.ConnectionError("Connection refused")):
        result = backend.complete_vision("b64data", "describe this")
    assert result == ""


def test_http_complete_vision_invalid_json():
    """Mock response with invalid JSON body. Verify returns ''."""
    import json as json_mod

    backend = HttpLLMBackend()
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.json.side_effect = json_mod.JSONDecodeError("Expecting value", "", 0)
    with patch.object(backend.session, "post", return_value=resp):
        result = backend.complete_vision("b64data", "describe this")
    assert result == ""


def test_http_complete_vision_uses_model_override():
    """Verify that model='llava' param is passed through in the payload."""
    backend = HttpLLMBackend()
    with patch.object(backend.session, "post", return_value=_make_chat_response("ok")) as mock_post:
        backend.complete_vision("b64data", "describe this", VisionCompletionOptions(model="llava"))
    payload = mock_post.call_args[1]["json"]
    assert payload["model"] == "llava"


def test_http_complete_vision_timeout_passthrough():
    """Verify timeout_s is passed to session.post."""
    backend = HttpLLMBackend()
    with patch.object(backend.session, "post", return_value=_make_chat_response("ok")) as mock_post:
        backend.complete_vision("b64data", "describe this", VisionCompletionOptions(timeout_s=42.5))
    assert mock_post.call_args[1]["timeout"] == 42.5


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
# create_llm_client_from_config: HTTP backend
# ---------------------------------------------------------------------------


def test_factory_http_backend_default():
    config = _config(use_llm=True)
    client = create_llm_client_from_config(config)
    assert isinstance(client, HttpLLMBackend)


def test_factory_http_backend_custom_url():
    config = _config(use_llm=True, llm_base_url="http://myserver:9000/v1/completions")
    client = create_llm_client_from_config(config)
    assert isinstance(client, HttpLLMBackend)
    assert client.base_url == "http://myserver:9000/v1/completions"


def test_factory_http_backend_env_url(monkeypatch):
    monkeypatch.setenv("FOLIONYM_LLM_URL", "http://envserver:1234/v1/completions")
    config = _config(use_llm=True)
    client = create_llm_client_from_config(config)
    assert isinstance(client, HttpLLMBackend)
    assert client.base_url == "http://envserver:1234/v1/completions"


def test_factory_http_backend_model_from_config():
    config = _config(use_llm=True, llm_model="qwen3:8b")
    client = create_llm_client_from_config(config)
    assert client.model == "qwen3:8b"


def test_factory_http_backend_model_from_env(monkeypatch):
    monkeypatch.setenv("FOLIONYM_LLM_MODEL", "mistral")
    config = _config(use_llm=True)
    client = create_llm_client_from_config(config)
    assert client.model == "mistral"


def test_factory_build_config_env_overrides_preset_defaults() -> None:
    env = {
        "FOLIONYM_LLM_URL": "http://envserver:1234/v1/completions",
        "FOLIONYM_LLM_MODEL": "env-model",
    }
    config = build_config({"llm_preset": "gpu"}, env=env)
    client = create_llm_client_from_config(config)
    assert isinstance(client, HttpLLMBackend)
    assert client.base_url == "http://envserver:1234/v1/completions"
    assert client.model == "env-model"


def test_factory_build_config_explicit_values_beat_env() -> None:
    env = {
        "FOLIONYM_LLM_URL": "http://envserver:1234/v1/completions",
        "FOLIONYM_LLM_MODEL": "env-model",
    }
    config = build_config(
        {
            "llm_preset": "gpu",
            "llm_base_url": "http://cli:9999/v1/completions",
            "llm_model": "cli-model",
        },
        env=env,
    )
    client = create_llm_client_from_config(config)
    assert isinstance(client, HttpLLMBackend)
    assert client.base_url == "http://cli:9999/v1/completions"
    assert client.model == "cli-model"


def test_build_config_env_beats_config_file_for_preset_derived_values() -> None:
    config = build_config(
        {},
        file_defaults={
            "llm_preset": "gpu",
        },
        env={
            "FOLIONYM_LLM_URL": "http://envserver:1234/v1/completions",
            "FOLIONYM_LLM_MODEL": "env-model",
        },
    )
    assert config.llm.backend.llm_base_url == "http://envserver:1234/v1/completions"
    assert config.llm.backend.llm_model == "env-model"


# ---------------------------------------------------------------------------
# close()
# ---------------------------------------------------------------------------


def test_http_backend_close():
    backend = HttpLLMBackend()
    backend.close()  # Should not raise


def test_serialized_client_prevents_overlapping_backend_calls() -> None:
    raw = MagicMock()
    raw.model = "test-model"
    raw.base_url = "http://localhost"
    first_entered = threading.Event()
    second_entered = threading.Event()
    release_first = threading.Event()
    second_lock_attempted = threading.Event()
    call_count = 0
    call_count_lock = threading.Lock()

    class ObservedLock:
        def __init__(self) -> None:
            self._lock = threading.Lock()
            self._attempts = 0
            self._attempts_lock = threading.Lock()

        def __enter__(self) -> None:
            with self._attempts_lock:
                self._attempts += 1
                if self._attempts == 2:
                    second_lock_attempted.set()
            self._lock.acquire()

        def __exit__(self, *_args: object) -> None:
            self._lock.release()

    def blocking_complete(*_args: object, **_kwargs: object) -> str:
        nonlocal call_count
        with call_count_lock:
            call_count += 1
            current_call = call_count
        if current_call == 1:
            first_entered.set()
            assert release_first.wait(timeout=1)
        else:
            second_entered.set()
        return f"result-{current_call}"

    raw.complete.side_effect = blocking_complete
    client = SerializedLLMClient(raw)
    vars(client)["_lock"] = ObservedLock()
    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(client.complete, "prompt")
        assert first_entered.wait(timeout=1)
        second = executor.submit(client.complete, "prompt")
        try:
            assert second_lock_attempted.wait(timeout=1)
            assert not second_entered.is_set()
        finally:
            release_first.set()
        results = [first.result(timeout=1), second.result(timeout=1)]

    assert sorted(results) == ["result-1", "result-2"]
    assert second_entered.is_set()


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


def test_warn_plaintext_remote_http_ipv6_loopback_ok() -> None:
    """Bracketed IPv6 loopback URLs are treated as local."""
    _warn_if_plaintext_remote("http://[::1]:8080/v1/completions", enforce=True)


def test_warn_plaintext_remote_http_ipv4_loopback_range_ok() -> None:
    """The full 127.0.0.0/8 IPv4 range is loopback."""
    _warn_if_plaintext_remote("http://127.0.0.2:8080/v1/completions", enforce=True)


def test_warn_plaintext_remote_https_ok() -> None:
    """https:// does not warn or raise."""
    _warn_if_plaintext_remote("https://example.com/v1/completions", enforce=True)


def test_warn_plaintext_remote_empty_url_is_invalid() -> None:
    """An empty endpoint is structurally invalid."""
    with pytest.raises(ValueError, match="must not be empty"):
        _warn_if_plaintext_remote("", enforce=True)


@pytest.mark.parametrize(
    "url",
    [
        "file:///tmp/model",
        "ftp://models.example.test/v1/completions",
        "https://",
        "https://user:password@models.example.test/v1/completions",
        "https://models.example.test:",
        "https://models.example.test:invalid/v1/completions",
        "https://bad host.example/v1/completions",
        "https://models.example.test\\v1\\completions",
        "https://models.example.test/v1/completions#unsafe-fragment",
    ],
)
def test_http_backend_rejects_structurally_invalid_endpoint(url: str) -> None:
    with pytest.raises(ValueError, match="Invalid LLM endpoint URL"):
        HttpLLMBackend(base_url=url)


@pytest.mark.parametrize(
    ("url", "normalized"),
    [
        (" https://models.example.test/v1/completions ", "https://models.example.test/v1/completions"),
        ("http://localhost:8080/v1/completions", "http://localhost:8080/v1/completions"),
        ("http://127.0.0.2:8080/v1/completions", "http://127.0.0.2:8080/v1/completions"),
        ("http://[::1]:8080/v1/completions", "http://[::1]:8080/v1/completions"),
    ],
)
def test_http_backend_accepts_and_normalizes_valid_endpoint(url: str, normalized: str) -> None:
    assert HttpLLMBackend(base_url=url).base_url == normalized


def test_build_config_rejects_invalid_llm_endpoint() -> None:
    with pytest.raises(ValueError, match="llm_base_url is invalid"):
        build_config({"llm_base_url": "file:///tmp/model"})


def test_build_config_enforces_remote_https_when_requested() -> None:
    with pytest.raises(ValueError, match="must use HTTPS"):
        build_config({"llm_base_url": "http://models.example.test/v1/completions", "require_https": True})


# ---------------------------------------------------------------------------
# Factory edge cases
# ---------------------------------------------------------------------------


def test_factory_timeout_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """FOLIONYM_LLM_TIMEOUT env var used when config.llm_timeout_s is None."""
    monkeypatch.setenv("FOLIONYM_LLM_TIMEOUT", "42.5")
    config = _config(use_llm=True, llm_timeout_s=None)
    client = create_llm_client_from_config(config)
    assert isinstance(client, HttpLLMBackend)
    assert client.timeout_s == 42.5


def test_factory_require_https_accepts_true_env_value(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FOLIONYM_REQUIRE_HTTPS", "true")
    config = _config(use_llm=True, llm_base_url="http://example.com/v1/completions")
    with pytest.raises(ValueError, match="plain HTTP"):
        create_llm_client_from_config(config)
