"""Model providers module."""

from nl2anyquery.providers.model.base import ModelProvider
from nl2anyquery.providers.model.koboldcpp import KoboldCppProvider

__all__ = ["ModelProvider", "KoboldCppProvider"]
