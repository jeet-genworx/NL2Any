"""Provider factory for instantiating ModelProvider based on selection."""

from typing import Any

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


def get_available_providers() -> list[dict[str, Any]]:
    """Return list of supported LLM providers and whether they are configured in settings."""
    from backend.src.config.config import settings

    return [
        {
            "id": "koboldcpp",
            "name": "KoboldCpp (Local SLM)",
            "configured": True,
            "model": settings.koboldcpp_model,
        },
        {
            "id": "huggingface",
            "name": "Hugging Face",
            "configured": bool(settings.hf_api_key and settings.hf_model),
            "model": settings.hf_model or None,
        },
        {
            "id": "gemini",
            "name": "Google Gemini",
            "configured": bool(settings.gemini_api_key and settings.gemini_model),
            "model": settings.gemini_model or None,
        },
    ]

