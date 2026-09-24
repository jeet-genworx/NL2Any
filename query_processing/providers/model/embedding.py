"""KoboldCpp embedding provider implementation.

Talks to the same KoboldCpp server as KoboldCppProvider (same host/port), but
to its /v1/embeddings endpoint, which serves whatever model KoboldCpp was
launched with via --embeddingsmodel (e.g. all-MiniLM-L6-v2-Q8_0.gguf), loaded
alongside the main chat model.
"""

from typing import Any
import httpx

from query_processing.core.config import settings


class KoboldCppEmbeddingProvider:
    """Embedding provider communicating with KoboldCpp's OpenAI-compatible /v1/embeddings API."""

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        timeout: float | None = None,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self.base_url = (base_url or settings.koboldcpp_base_url).rstrip("/")
        self.model = model or settings.koboldcpp_embedding_model
        self.timeout = (
            timeout if timeout is not None else max(60.0, float(settings.query_timeout_seconds))
        )
        self._external_client = http_client

    async def _get_client(self) -> httpx.AsyncClient:
        if self._external_client is not None:
            return self._external_client
        return httpx.AsyncClient(timeout=self.timeout)

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts via KoboldCpp's embeddings endpoint.

        Returns one embedding vector per input text, in the same order.
        """
        payload: dict[str, Any] = {"model": self.model, "input": texts}
        endpoint = f"{self.base_url}/embeddings"
        client = await self._get_client()

        try:
            if self._external_client is None:
                async with client:
                    response = await client.post(endpoint, json=payload)
            else:
                response = await client.post(endpoint, json=payload)

            response.raise_for_status()
            data = response.json()
            items = data.get("data", [])
            if not items:
                raise ValueError("KoboldCpp returned an embeddings response with no data.")

            items.sort(key=lambda item: item.get("index", 0))
            return [item["embedding"] for item in items]

        except httpx.ConnectError as err:
            raise ConnectionError(
                f"Failed to connect to KoboldCpp at '{self.base_url}'. "
                f"Please ensure KoboldCpp is running with --embeddingsmodel: {err}"
            ) from err
        except httpx.TimeoutException as err:
            raise TimeoutError(
                f"KoboldCpp embeddings request to '{self.base_url}' timed out after {self.timeout}s: {err}"
            ) from err
        except httpx.HTTPStatusError as err:
            raise RuntimeError(
                f"KoboldCpp embeddings HTTP {err.response.status_code} error: {err.response.text}"
            ) from err
        except Exception as err:
            if not isinstance(err, (ConnectionError, TimeoutError, RuntimeError, ValueError)):
                raise RuntimeError(f"Unexpected error communicating with KoboldCpp embeddings: {err}") from err
            raise
