"""Embedding provider protocol and base interfaces."""

from typing import Protocol, runtime_checkable


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Protocol for embedding generation providers."""

    async def embed(self, text: str) -> list[float]:
        """Generate embedding vector for input text."""
        ...
