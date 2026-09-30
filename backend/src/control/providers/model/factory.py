"""Provider factory for instantiating ModelProvider based on selection."""

from backend.src.control.providers.model.base import ModelProvider
from backend.src.control.providers.model.gemini import GeminiProvider
from backend.src.control.providers.model.huggingface import HuggingFaceProvider
from backend.src.control.providers.model.koboldcpp import KoboldCppProvider


def resolve_model_provider(name: str = "koboldcpp") -> ModelProvider:
    """Resolve and return an instantiated ModelProvider based on key.

    Supported keys:
    - 'koboldcpp', 'slm' -> KoboldCppProvider
    - 'huggingface', 'hf' -> HuggingFaceProvider
    - 'gemini' -> GeminiProvider
    """
    normalized = (name or "koboldcpp").strip().lower()
    if normalized in ("koboldcpp", "slm"):
        return KoboldCppProvider()
    elif normalized in ("huggingface", "hf"):
        return HuggingFaceProvider()
    elif normalized == "gemini":
        return GeminiProvider()
    else:
        raise ValueError(
            f"Unsupported LLM provider: '{name}'. "
            "Supported providers are: 'koboldcpp', 'huggingface', 'gemini'."
        )
