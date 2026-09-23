"""KoboldCpp model provider implementation."""

from typing import Any
import httpx

from nl2anyquery.core.config import settings


class KoboldCppProvider:
    """Model provider communicating with a local KoboldCpp OpenAI-compatible HTTP API."""

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        timeout: float | None = None,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self.base_url = (base_url or settings.koboldcpp_base_url).rstrip("/")
        self.model = model or settings.koboldcpp_model
        self.default_temperature = (
            temperature if temperature is not None else settings.model_temperature
        )
        self.default_max_tokens = (
            max_tokens if max_tokens is not None else settings.model_max_tokens
        )
        # Allow adequate time for local SLM generation (at least 90s)
        self.timeout = (
            timeout
            if timeout is not None
            else max(90.0, float(settings.query_timeout_seconds))
        )
        self._external_client = http_client

    async def _get_client(self) -> httpx.AsyncClient:
        if self._external_client is not None:
            return self._external_client
        return httpx.AsyncClient(timeout=self.timeout)

    async def generate(
        self,
        prompt: str,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        system_prompt: str | None = None,
    ) -> str:
        """Generate response from KoboldCpp.

        Returns the raw model response string without any post-processing.
        """
        temp = temperature if temperature is not None else self.default_temperature
        max_toks = max_tokens if max_tokens is not None else self.default_max_tokens

        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temp,
            "max_tokens": max_toks,
        }

        endpoint = f"{self.base_url}/chat/completions"
        client = await self._get_client()

        try:
            # If using self-managed client, manage context
            if self._external_client is None:
                async with client:
                    response = await client.post(endpoint, json=payload)
            else:
                response = await client.post(endpoint, json=payload)

            response.raise_for_status()
            data = response.json()
            choices = data.get("choices", [])
            if not choices:
                raise ValueError("KoboldCpp returned response with no choices.")

            message = choices[0].get("message", {})
            content = message.get("content", "")
            return content

        except httpx.ConnectError as err:
            raise ConnectionError(
                f"Failed to connect to KoboldCpp at '{self.base_url}'. "
                f"Please ensure KoboldCpp is running and accessible: {err}"
            ) from err
        except httpx.TimeoutException as err:
            raise TimeoutError(
                f"KoboldCpp request to '{self.base_url}' timed out after {self.timeout}s: {err}"
            ) from err
        except httpx.HTTPStatusError as err:
            raise RuntimeError(
                f"KoboldCpp HTTP {err.response.status_code} error: {err.response.text}"
            ) from err
        except Exception as err:
            if not isinstance(err, (ConnectionError, TimeoutError, RuntimeError, ValueError)):
                raise RuntimeError(f"Unexpected error communicating with KoboldCpp: {err}") from err
            raise

    async def test_connection(self) -> str:
        """Send a trivial prompt to verify KoboldCpp connectivity."""
        return await self.generate(
            prompt="Respond with: KoboldCpp connection successful",
            max_tokens=150,
            temperature=0.1,
        )
