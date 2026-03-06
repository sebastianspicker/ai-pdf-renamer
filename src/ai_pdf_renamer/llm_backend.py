"""
LLM backend abstraction: HTTP (any OpenAI-compatible server) and optional in-process
(llama-cpp-python) backends.

Usage:
    from .llm_backend import create_llm_client_from_config, HttpLLMBackend, LocalLLMClient

    client = create_llm_client_from_config(config)
    text = client.complete("your prompt here")
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol, runtime_checkable

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
    ) -> str: ...

    def complete_vision(
        self,
        image_b64: str,
        prompt: str,
        *,
        model: str | None = None,
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


def _chat_url_from_completions_url(completions_url: str) -> str:
    """Derive /v1/chat/completions URL from /v1/completions URL."""
    base = (completions_url or "").strip().rstrip("/")
    if "/v1/completions" in base:
        return base.replace("/v1/completions", "") + "/v1/chat/completions"
    if base.endswith("/v1/chat/completions"):
        return base
    return base + "/v1/chat/completions" if base else "http://127.0.0.1:8080/v1/chat/completions"


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
    _session: requests.Session = field(default_factory=requests.Session, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        self._session.trust_env = False  # Never route LLM traffic through proxy

    def complete(
        self,
        prompt: str,
        *,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> str:
        payload: dict[str, object] = {
            "model": self.model,
            "prompt": prompt,
            "temperature": temperature,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        try:
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
                            "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
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
            choices = data.get("choices")
            if not isinstance(choices, list) or not choices:
                return ""
            msg = choices[0].get("message", {})
            if not isinstance(msg, dict):
                return ""
            content = msg.get("content")
            return str(content).strip() if content is not None else ""
        except Exception as exc:
            logger.warning("Vision API failed: %s", exc)
            return ""

    def close(self) -> None:
        try:
            self._session.close()
        except Exception as exc:
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

    def __init__(self, model_path: str, *, timeout_s: float = _DEFAULT_LLM_TIMEOUT_S) -> None:
        try:
            import llama_cpp  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError(
                "llama-cpp-python is required for the in-process backend. Install it with: pip install llama-cpp-python"
            ) from exc
        self._model_path = model_path
        self._timeout_s = timeout_s
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
    ) -> str:
        try:
            result = self._llama.create_completion(
                prompt=prompt,
                temperature=temperature,
                max_tokens=max_tokens or 1024,
            )
            choices = result.get("choices", [])
            if not choices:
                return ""
            text = choices[0].get("text", "")
            return str(text).strip() if text else ""
        except Exception as exc:
            logger.error("In-process LLM error: %s", exc)
            return ""

    def complete_vision(
        self,
        image_b64: str,
        prompt: str,
        *,
        model: str | None = None,
        timeout_s: float = 120.0,
    ) -> str:
        try:
            import base64

            image_bytes = base64.b64decode(image_b64)
            result = self._llama.create_chat_completion(
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
                        ],
                    }
                ]
            )
            _ = image_bytes  # referenced to avoid F841
            choices = result.get("choices", [])
            if not choices:
                return ""
            msg = choices[0].get("message", {})
            content = msg.get("content", "")
            return str(content).strip() if content else ""
        except Exception as exc:
            logger.warning("In-process vision failed: %s", exc)
            return ""

    def close(self) -> None:
        try:
            del self._llama
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


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
        getattr(config, "llm_backend", None),
        "AI_PDF_RENAMER_LLM_BACKEND",
        "http",
    ).lower()

    model_path = _config_or_env(
        getattr(config, "llm_model_path", None),
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

    # In-process backend
    use_in_process = backend_str == "in-process" or (backend_str == "auto" and bool(model_path))
    if use_in_process and model_path:
        try:
            return InProcessLLMBackend(model_path, timeout_s=timeout_s)
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
    return HttpLLMBackend(base_url=base_url, model=model, timeout_s=timeout_s)
