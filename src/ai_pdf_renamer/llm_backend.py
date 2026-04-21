"""
LLM backend abstraction: HTTP (any OpenAI-compatible server) and optional in-process
(llama-cpp-python) backends.

Usage:
    from .llm_backend import create_llm_client_from_config, HttpLLMBackend, LocalLLMClient

    client = create_llm_client_from_config(config)
    text = client.complete("your prompt here")
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol, runtime_checkable
from urllib.parse import urlsplit

import requests

if TYPE_CHECKING:
    from .config import RenamerConfig

logger = logging.getLogger(__name__)

# Default endpoint for llama.cpp server (--port 8080 is the llama.cpp default)
_DEFAULT_LLM_URL = "http://127.0.0.1:8080/v1/completions"
_DEFAULT_LLM_MODEL = "default"
_DEFAULT_LLM_TIMEOUT_S = 60.0


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class LLMClient(Protocol):
    """Protocol for LLM backends. All backends expose complete() and complete_vision()."""

    @property
    def model(self) -> str: ...

    @property
    def base_url(self) -> str: ...

    def complete(
        self,
        prompt: str,
        *,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        response_format: dict[str, str] | None = None,
    ) -> str: ...

    def complete_vision(
        self,
        image_b64: str,
        prompt: str,
        *,
        model: str | None = None,
        image_mime_type: str = "image/jpeg",
        timeout_s: float = 120.0,
    ) -> str: ...

    def close(self) -> None: ...


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _config_or_env(value: str | None, env_key: str, default: str) -> str:
    """Resolve a string from config value, then env var, then built-in default."""
    s = (value or "").strip() or (os.environ.get(env_key) or "").strip()
    return s or default


def _env_truthy(env_key: str) -> bool:
    return (os.environ.get(env_key, "") or "").strip().lower() in {"1", "true", "yes", "on"}


def _chat_url_from_completions_url(completions_url: str) -> str:
    """Derive /v1/chat/completions URL from /v1/completions URL."""
    base = (completions_url or "").strip().rstrip("/")
    # P2: Use endswith and suffix replacement instead of str.replace to avoid
    # replacing /v1/completions if it appears in the hostname or path prefix
    if base.endswith("/v1/completions"):
        return base[: -len("/v1/completions")] + "/v1/chat/completions"
    if base.endswith("/v1/chat/completions"):
        return base
    return base + "/v1/chat/completions" if base else "http://127.0.0.1:8080/v1/chat/completions"


def _extract_chat_message_content(data: dict[str, object]) -> str:
    """Extract text content from an OpenAI-compatible chat completion response dict."""
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    msg = choices[0].get("message", {})
    if not isinstance(msg, dict):
        return ""
    content = msg.get("content")
    return str(content).strip() if content is not None else ""


# ---------------------------------------------------------------------------
# HTTP backend
# ---------------------------------------------------------------------------


@dataclass
class HttpLLMBackend:
    """
    HTTP LLM backend. Works with any OpenAI-compatible server including:
    - llama.cpp server (default port 8080)
    - Ollama (port 11434)
    - LM Studio, vLLM, etc.

    A single requests.Session is created per backend instance for connection reuse.
    """

    base_url: str = _DEFAULT_LLM_URL
    model: str = _DEFAULT_LLM_MODEL
    timeout_s: float = _DEFAULT_LLM_TIMEOUT_S
    use_chat: bool = True
    _session: requests.Session = field(default_factory=requests.Session, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        self._session.trust_env = False  # Never route LLM traffic through proxy

    def _complete_text(
        self,
        prompt: str,
        *,
        temperature: float,
        max_tokens: int | None,
        response_format: dict[str, str] | None,
    ) -> str:
        """Legacy /v1/completions text completion path."""
        payload: dict[str, object] = {
            "model": self.model,
            "prompt": prompt,
            "temperature": temperature,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if response_format is not None:
            payload["response_format"] = response_format
        resp = self._session.post(self.base_url, json=payload, timeout=self.timeout_s)
        resp.raise_for_status()
        data = resp.json()
        if not isinstance(data, dict):
            logger.warning("LLM response is not a dict: %s", type(data).__name__)
            return ""
        choices = data.get("choices")
        if not isinstance(choices, list) or len(choices) == 0:
            logger.warning(
                "LLM response missing or empty 'choices' (status=%s)",
                getattr(resp, "status_code", None),
            )
            return ""
        first_choice = choices[0]
        if not isinstance(first_choice, dict):
            logger.warning(
                "LLM response 'choices[0]' is not a dict: %s",
                type(first_choice).__name__,
            )
            return ""
        text = first_choice.get("text", "")
        return str(text).strip() if text is not None else ""

    def _complete_chat(
        self,
        prompt: str,
        *,
        temperature: float,
        max_tokens: int | None,
        response_format: dict[str, str] | None,
    ) -> str:
        """Chat /v1/chat/completions path for instruct-tuned models."""
        chat_url = _chat_url_from_completions_url(self.base_url)
        payload: dict[str, object] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You are a document analysis assistant. Respond only with valid JSON."},
                {"role": "user", "content": prompt},
            ],
            "temperature": temperature,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if response_format is not None:
            payload["response_format"] = response_format
        resp = self._session.post(chat_url, json=payload, timeout=self.timeout_s)
        resp.raise_for_status()
        data = resp.json()
        if not isinstance(data, dict):
            return ""
        return _extract_chat_message_content(data)

    def complete(
        self,
        prompt: str,
        *,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        response_format: dict[str, str] | None = None,
    ) -> str:
        try:
            if self.use_chat:
                return self._complete_chat(
                    prompt, temperature=temperature, max_tokens=max_tokens, response_format=response_format
                )
            return self._complete_text(
                prompt, temperature=temperature, max_tokens=max_tokens, response_format=response_format
            )
        except (IndexError, AttributeError, TypeError, KeyError) as exc:
            logger.warning("LLM response structure unexpected: %s", exc)
            return ""
        except requests.HTTPError as exc:
            logger.warning(
                "LLM HTTP error: %s (status=%s, body=%s)",
                exc,
                getattr(exc.response, "status_code", None),
                (getattr(exc.response, "text", None) or "")[:500],
            )
            return ""
        except requests.RequestException as exc:
            logger.error(
                "LLM unreachable (%s). Category/summary will use heuristic or 'na' for this document.",
                exc,
            )
            return ""
        except json.JSONDecodeError as exc:
            logger.error(
                "LLM response not valid JSON: %s. Using fallback for this document.",
                exc,
            )
            return ""

    def complete_vision(
        self,
        image_b64: str,
        prompt: str,
        *,
        model: str | None = None,
        image_mime_type: str = "image/jpeg",
        timeout_s: float = 120.0,
    ) -> str:
        """Send image + text prompt to the OpenAI-compatible /v1/chat/completions endpoint."""
        chat_url = _chat_url_from_completions_url(self.base_url)
        vision_model = model or self.model
        payload = {
            "model": vision_model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{image_mime_type};base64,{image_b64}"},
                        },
                    ],
                }
            ],
            "stream": False,
        }
        try:
            resp = self._session.post(chat_url, json=payload, timeout=timeout_s)
            resp.raise_for_status()
            data = resp.json()
            if not isinstance(data, dict):
                return ""
            return _extract_chat_message_content(data)
        except (requests.RequestException, json.JSONDecodeError, KeyError, IndexError, TypeError, ValueError) as exc:
            logger.warning("Vision API failed: %s", exc)
            return ""

    def close(self) -> None:
        try:
            self._session.close()
        except OSError as exc:
            logger.debug("Could not close LLM session cleanly: %s", exc)


# Backward compatibility alias
LocalLLMClient = HttpLLMBackend


# ---------------------------------------------------------------------------
# In-process backend (llama-cpp-python)
# ---------------------------------------------------------------------------


class InProcessLLMBackend:
    """
    In-process LLM backend using llama-cpp-python.
    Requires: pip install llama-cpp-python (or install with [llama-cpp] extra)

    No server needed. Loads a GGUF model directly into the process.
    """

    def __init__(
        self,
        model_path: str,
        *,
        timeout_s: float = _DEFAULT_LLM_TIMEOUT_S,
        use_chat: bool = True,
    ) -> None:
        try:
            import llama_cpp
        except ImportError as exc:
            raise ImportError(
                "llama-cpp-python is required for the in-process backend. Install it with: pip install llama-cpp-python"
            ) from exc
        self._model_path = model_path
        self._timeout_s = timeout_s
        self._use_chat = use_chat
        logger.info("Loading GGUF model from %s ...", model_path)
        self._llama = llama_cpp.Llama(model_path=model_path, verbose=False)
        logger.info("Model loaded.")

    @property
    def model(self) -> str:
        return self._model_path

    @property
    def base_url(self) -> str:
        return f"file://{self._model_path}"

    def complete(
        self,
        prompt: str,
        *,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        response_format: dict[str, str] | None = None,
    ) -> str:
        try:
            kwargs: dict[str, object] = {
                "temperature": temperature,
                "max_tokens": max_tokens or 1024,
            }
            if response_format is not None:
                kwargs["response_format"] = response_format
            if self._use_chat:
                sys_msg = "You are a document analysis assistant. Respond only with valid JSON."
                result = self._llama.create_chat_completion(
                    messages=[
                        {"role": "system", "content": sys_msg},
                        {"role": "user", "content": prompt},
                    ],
                    **kwargs,
                )
                return _extract_chat_message_content(result)
            result = self._llama.create_completion(
                prompt=prompt,
                **kwargs,
            )
            choices = result.get("choices", [])
            if not choices:
                return ""
            text = choices[0].get("text", "")
            return str(text).strip() if text else ""
        except (RuntimeError, ValueError, TypeError, AttributeError, KeyError, IndexError) as exc:
            logger.error("In-process LLM error: %s", exc)
            return ""

    def complete_vision(
        self,
        image_b64: str,
        prompt: str,
        *,
        model: str | None = None,
        image_mime_type: str = "image/jpeg",
        timeout_s: float = 120.0,
    ) -> str:
        try:
            result = self._llama.create_chat_completion(
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:{image_mime_type};base64,{image_b64}"},
                            },
                        ],
                    }
                ]
            )
            return _extract_chat_message_content(result)
        except (RuntimeError, ValueError, TypeError, AttributeError, KeyError, IndexError) as exc:
            logger.warning("In-process vision failed: %s", exc)
            return ""

    def close(self) -> None:
        with contextlib.suppress(AttributeError, RuntimeError):
            del self._llama


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "::1", "localhost"})


def _warn_if_plaintext_remote(url: str, *, enforce: bool = False) -> None:
    """Warn or raise when PDF content will be sent to a non-loopback host over plain HTTP."""
    stripped = (url or "").strip()
    if not stripped.lower().startswith("http://"):
        return
    host = (urlsplit(stripped).hostname or "").lower()
    if host not in _LOOPBACK_HOSTS:
        msg = (
            f"LLM endpoint uses plain HTTP with a non-loopback host ({host}). "
            "PDF content will be transmitted unencrypted. Use HTTPS for remote endpoints."
        )
        if enforce:
            raise ValueError(msg)
        logger.warning(msg)


def create_llm_client_from_config(config: RenamerConfig) -> LLMClient:
    """
    Build an LLM backend from config + env vars.

    Backend selection (in priority order):
    1. config.llm_backend == "in-process" or "auto" with model_path -> InProcessLLMBackend
    2. Otherwise -> HttpLLMBackend

    Env vars (all override config when config value is absent):
      AI_PDF_RENAMER_LLM_BACKEND   - "http", "in-process", or "auto"
      AI_PDF_RENAMER_LLM_MODEL_PATH - path to GGUF model file
      AI_PDF_RENAMER_LLM_URL       - HTTP endpoint URL
      AI_PDF_RENAMER_LLM_MODEL     - model name for HTTP backend
      AI_PDF_RENAMER_LLM_TIMEOUT   - timeout in seconds
    """
    backend_str = _config_or_env(
        config.llm_backend,
        "AI_PDF_RENAMER_LLM_BACKEND",
        "http",
    ).lower()

    model_path = _config_or_env(
        config.llm_model_path,
        "AI_PDF_RENAMER_LLM_MODEL_PATH",
        "",
    )

    # Resolve timeout
    timeout_s = config.llm_timeout_s
    if timeout_s is None or timeout_s <= 0:
        try:
            timeout_s = float(os.environ.get("AI_PDF_RENAMER_LLM_TIMEOUT", "") or 0)
        except ValueError:
            timeout_s = _DEFAULT_LLM_TIMEOUT_S
        if timeout_s <= 0:
            timeout_s = _DEFAULT_LLM_TIMEOUT_S

    use_chat = config.llm_use_chat_api

    # In-process backend
    use_in_process = backend_str == "in-process" or (backend_str == "auto" and bool(model_path))
    if use_in_process and model_path:
        try:
            return InProcessLLMBackend(model_path, timeout_s=timeout_s, use_chat=use_chat)
        except ImportError as exc:
            if backend_str == "in-process":
                raise
            logger.warning("llama-cpp-python not available, falling back to HTTP: %s", exc)

    # HTTP backend
    base_url = _config_or_env(
        config.llm_base_url,
        "AI_PDF_RENAMER_LLM_URL",
        _DEFAULT_LLM_URL,
    )
    model = _config_or_env(
        config.llm_model,
        "AI_PDF_RENAMER_LLM_MODEL",
        _DEFAULT_LLM_MODEL,
    )
    require_https = config.require_https or _env_truthy("AI_PDF_RENAMER_REQUIRE_HTTPS")
    _warn_if_plaintext_remote(base_url, enforce=require_https)
    return HttpLLMBackend(base_url=base_url, model=model, timeout_s=timeout_s, use_chat=use_chat)
