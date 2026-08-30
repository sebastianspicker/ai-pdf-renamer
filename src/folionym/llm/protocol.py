"""Framework-free client protocols and serialization helpers."""

from __future__ import annotations

import threading
from typing import Protocol, runtime_checkable

from .models import VisionCompletionOptions


@runtime_checkable
class LLMClient(Protocol):
    """Contract implemented by text and vision completion clients."""

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

    def complete_vision(self, image_b64: str, prompt: str, options: VisionCompletionOptions | None = None) -> str: ...

    def close(self) -> None: ...


class SerializedLLMClient:
    """Serialize access to one run-scoped client shared by worker threads."""

    def __init__(self, client: LLMClient) -> None:
        self._client = client
        self._lock = threading.Lock()

    @property
    def model(self) -> str:
        return self._client.model

    @property
    def base_url(self) -> str:
        return self._client.base_url

    def complete(
        self,
        prompt: str,
        *,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        response_format: dict[str, str] | None = None,
    ) -> str:
        with self._lock:
            return self._client.complete(
                prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format=response_format,
            )

    def complete_vision(self, image_b64: str, prompt: str, options: VisionCompletionOptions | None = None) -> str:
        with self._lock:
            return self._client.complete_vision(image_b64, prompt, options)

    def close(self) -> None:
        with self._lock:
            self._client.close()
