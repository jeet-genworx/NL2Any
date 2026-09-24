"""Embedding providers module."""

from query_processing.providers.embedding.base import EmbeddingProvider
from query_processing.providers.embedding.koboldcpp import KoboldCppEmbeddingProvider

__all__ = ["EmbeddingProvider", "KoboldCppEmbeddingProvider"]
