"""Base protocol for model providers."""

from typing import Protocol, runtime_checkable


@runtime_checkable
class ModelProvider(Protocol):
    """Protocol defining interface for language model providers."""

    async def generate(
        self,
        prompt: str,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        system_prompt: str | None = None,
    ) -> str:
        """Generate text from a prompt.

        Args:
            prompt: User/instruction prompt.
            temperature: Optional sampling temperature override.
            max_tokens: Optional maximum tokens to generate override.
            system_prompt: Optional system prompt to steer behavior.

        Returns:
            The raw text response from the model.
        """
        ...
