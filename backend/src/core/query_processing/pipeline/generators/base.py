"""Base protocol for database query generators."""

from typing import Any, Protocol, runtime_checkable
from backend.src.schemas.pipeline import GeneratedQuery, GeneratorMissingInfo, RelevantSchema


@runtime_checkable
class QueryGenerator(Protocol):
    """Protocol for generating native database queries directly from question and selected schema."""

    async def generate_query(
        self,
        question: str,
        schema: RelevantSchema,
        feedback: str | None = None,
        plan: Any = None,
    ) -> GeneratedQuery | GeneratorMissingInfo:
        """Generate a native database query or structured missing information feedback."""
        ...
