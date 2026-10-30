"""Show the selector the entire schema, and let the KV cache pay for it."""

from __future__ import annotations

from backend.src.core.query_processing.pipeline.fsm.context import PipelineContext
from backend.src.core.query_processing.pipeline.strategies.base import (
    SelectionScope,
    TableSelectionStrategy,
)

STRATEGY_NAME = "whole_schema"


class WholeSchemaStrategy(TableSelectionStrategy):
    """The default: every table, every time, with a cacheable prompt prefix.

    This is what the pipeline did before strategies existed, and extracting it
    is what makes the alternative expressible. It does no work of its own --
    the point is that the prefix never varies, so `cache_key` is the database
    key and the selector may restore its KV slot.
    """

    name = STRATEGY_NAME

    async def scope(self, context: PipelineContext) -> SelectionScope:
        return SelectionScope(
            schema_view=context.schema,
            cache_key=context.target_name,
        )
