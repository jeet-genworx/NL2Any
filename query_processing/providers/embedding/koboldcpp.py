"""KoboldCpp embedding provider implementation."""

import logging
from typing import Any
import httpx

from query_processing.core.config import settings
from query_processing.providers.embedding.base import EmbeddingProvider

logger = logging.getLogger(__name__)

EXPECTED_DIMENSION = 384  # for all-MiniLM-L6-v2


class KoboldCppEmbeddingProvider(EmbeddingProvider):
    """Generates embeddings using KoboldCpp /v1/embeddings endpoint."""

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        timeout: float | None = None,
        client: httpx.AsyncClient | None = None,
        expected_dim: int = EXPECTED_DIMENSION,
    ) -> None:
        self.base_url = (base_url or settings.koboldcpp_base_url).rstrip("/")
        self.model = model or settings.embedding_model
        # Same KoboldCpp server as generation, so an embedding call can queue
        # behind an in-flight completion; share the model timeout.
        self.timeout = timeout if timeout is not None else float(settings.model_timeout_seconds)
        self._client = client
        self.expected_dim = expected_dim

    async def embed(self, text: str) -> list[float]:
        """Generate embedding vector for input text via KoboldCpp."""
        if not text or not text.strip():
            raise ValueError("Input text for embedding cannot be empty.")

        url = f"{self.base_url}/embeddings"
        payload: dict[str, Any] = {
            "input": text.strip(),
            "model": self.model,
        }

        try:
            if self._client is not None:
                resp = await self._client.post(url, json=payload, timeout=self.timeout)
                resp.raise_for_status()
                data = resp.json()
            else:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    resp = await client.post(url, json=payload)
                    resp.raise_for_status()
                    data = resp.json()
        except httpx.HTTPError as err:
            logger.error("KoboldCpp embedding request failed at %s: %s", url, err)
            raise ConnectionError(f"KoboldCpp embedding request failed: {err}") from err

        return self._parse_and_validate_response(data)

    def _parse_and_validate_response(self, data: Any) -> list[float]:
        """Parse and strictly validate the embedding response."""
        if not isinstance(data, dict):
            raise ValueError(f"Invalid embedding response format: expected dict, got {type(data).__name__}")

        raw_data = data.get("data")
        if not isinstance(raw_data, list) or not raw_data:
            raise ValueError(f"Invalid embedding response: missing or empty 'data' list in {data}")

        embedding_item = raw_data[0]
        if not isinstance(embedding_item, dict):
            raise ValueError(f"Invalid embedding item: expected dict, got {type(embedding_item).__name__}")

        vector = embedding_item.get("embedding")
        if not isinstance(vector, list) or not vector:
            raise ValueError(f"Invalid embedding vector: expected non-empty list, got {type(vector).__name__}")

        float_vector: list[float] = []
        for idx, val in enumerate(vector):
            if not isinstance(val, (int, float)) or isinstance(val, bool):
                raise ValueError(f"Embedding vector component at index {idx} is non-numeric: {val!r}")
            float_vector.append(float(val))

        if len(float_vector) != self.expected_dim:
            raise ValueError(
                f"Unexpected embedding dimension: got {len(float_vector)}, expected {self.expected_dim}"
            )

        return float_vector
