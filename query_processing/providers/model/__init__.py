"""Model providers module."""

from query_processing.providers.model.base import ModelProvider
from query_processing.providers.model.koboldcpp import KoboldCppProvider

__all__ = ["ModelProvider", "KoboldCppProvider"]
