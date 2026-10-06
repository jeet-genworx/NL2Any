"""Base query validator abstract interface."""

from abc import ABC, abstractmethod
from typing import Any

from backend.src.schemas.pipeline import ValidationResult


class QueryValidator(ABC):
    """Abstract base class for database query validators."""

    @abstractmethod
    async def validate(
        self,
        question: str,
        generated_query: Any = None,
        schema: Any = None,
        plan: Any = None,
        **kwargs: Any,
    ) -> ValidationResult:
        """Validate generated database query."""
        ...
