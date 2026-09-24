"""Base protocol for database query generators."""

from typing import Protocol, runtime_checkable
from query_processing.models.pipeline import GeneratedQuery, QueryPlan, RelevantSchema


@runtime_checkable
class QueryGenerator(Protocol):
    """Protocol for converting database-independent QueryPlan into native queries."""

    async def generate_query(
        self,
        question: str,
        plan: QueryPlan,
        schema: RelevantSchema,
        feedback: str | None = None,
    ) -> GeneratedQuery:
        """Generate a native database query based on the plan and relevant schema."""
        ...
