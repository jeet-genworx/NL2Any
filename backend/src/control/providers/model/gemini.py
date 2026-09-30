"""Google Gemini model provider implementation."""

import logging
from typing import Any
from google import genai
from google.genai import types
from google.genai.errors import APIError

from backend.src.config import settings

logger = logging.getLogger(__name__)


class GeminiProvider:
    """Model provider communicating with Google Gemini API via official google-genai SDK."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        timeout: float | None = None,
        client: genai.Client | None = None,
    ) -> None:
        self.api_key = api_key if api_key is not None else settings.gemini_api_key
        self.model = model if model is not None else settings.gemini_model
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
                    "GEMINI_API_KEY must be configured in environment/.env to use Gemini provider."
                )
            if not self.model or not self.model.strip():
                raise ValueError(
                    "GEMINI_MODEL must be configured in environment/.env to use Gemini provider."
                )

    def _get_client(self) -> genai.Client:
        if self._client is not None:
            return self._client
        return genai.Client(api_key=self.api_key)

    async def generate(
        self,
        prompt: str,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        system_prompt: str | None = None,
    ) -> str:
        """Generate response from Gemini model.

        Returns raw model response string without any post-processing.
        """
        temp = temperature if temperature is not None else self.default_temperature
        max_toks = max_tokens if max_tokens is not None else self.default_max_tokens

        config = types.GenerateContentConfig(
            temperature=temp,
            max_output_tokens=max_toks,
            system_instruction=system_prompt,
        )

        client = self._get_client()

        try:
            response = await client.aio.models.generate_content(
                model=self.model,
                contents=prompt,
                config=config,
            )
            return response.text or ""

        except APIError as err:
            raise RuntimeError(f"Gemini API error: {err}") from err
        except TimeoutError as err:
            raise TimeoutError(f"Gemini request timed out: {err}") from err
        except Exception as err:
            if not isinstance(err, (ConnectionError, TimeoutError, RuntimeError, ValueError)):
                raise RuntimeError(f"Unexpected error communicating with Gemini: {err}") from err
            raise
