"""Hugging Face model provider implementation."""

import logging
from typing import Any
import httpx
from huggingface_hub import AsyncInferenceClient

from backend.src.config import settings

logger = logging.getLogger(__name__)


class HuggingFaceProvider:
    """Model provider communicating with Hugging Face Serverless / Dedicated Inference API."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        timeout: float | None = None,
        client: AsyncInferenceClient | None = None,
    ) -> None:
        self.api_key = api_key if api_key is not None else settings.hf_api_key
        self.model = model if model is not None else settings.hf_model
        self.base_url = base_url if base_url is not None else settings.hf_base_url
        self.default_temperature = (
            temperature if temperature is not None else settings.model_temperature
        )
        self.default_max_tokens = (
            max_tokens if max_tokens is not None else settings.model_max_tokens
        )
        if timeout is not None:
            self.timeout = float(timeout) if timeout > 0 else None
        elif settings.model_timeout_seconds is not None and settings.model_timeout_seconds > 0:
            self.timeout = float(settings.model_timeout_seconds)
        else:
            self.timeout = None

        self._client = client

        # Validation: do not allow empty credentials or model unless pre-injected client is provided
        if not self._client:
            if not self.api_key or not self.api_key.strip():
                raise ValueError(
                    "HF_API_KEY must be configured in environment/.env to use Hugging Face provider."
                )
            if not self.model or not self.model.strip():
                raise ValueError(
                    "HF_MODEL must be configured in environment/.env to use Hugging Face provider."
                )

    def _get_client(self) -> AsyncInferenceClient:
        if self._client is not None:
            return self._client
        return AsyncInferenceClient(
            token=self.api_key,
            model=self.base_url or self.model,
            timeout=self.timeout,
        )

    async def generate(
        self,
        prompt: str,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        system_prompt: str | None = None,
    ) -> str:
        """Generate response from Hugging Face model.

        Returns raw model response string without any post-processing.
        """
        temp = temperature if temperature is not None else self.default_temperature
        max_toks = max_tokens if max_tokens is not None else self.default_max_tokens

        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        client = self._get_client()

        try:
            response = await client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temp,
                max_tokens=max_toks,
            )
            choices = getattr(response, "choices", None)
            if not choices:
                raise ValueError("Hugging Face returned response with no choices.")
            message = getattr(choices[0], "message", None)
            content = getattr(message, "content", "") if message else ""
            return content or ""

        except httpx.ConnectError as err:
            raise ConnectionError(
                f"Failed to connect to Hugging Face API: {err}"
            ) from err
        except httpx.TimeoutException as err:
            raise TimeoutError(
                f"Hugging Face request timed out after {self.timeout}s: {err}"
            ) from err
        except httpx.HTTPStatusError as err:
            raise RuntimeError(
                f"Hugging Face HTTP {err.response.status_code} error: {err.response.text}"
            ) from err
        except Exception as err:
            if not isinstance(err, (ConnectionError, TimeoutError, RuntimeError, ValueError)):
                raise RuntimeError(f"Unexpected error communicating with Hugging Face: {err}") from err
            raise
