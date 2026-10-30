"""Table-selection strategies: what schema view Stage 1 is shown."""

from backend.src.core.query_processing.pipeline.strategies.base import (
    SelectionScope,
    TableSelectionStrategy,
)
from backend.src.core.query_processing.pipeline.strategies.bfs import (
    BfsRetrievalUnavailable,
    BfsStrategy,
)
from backend.src.core.query_processing.pipeline.strategies.whole_schema import (
    WholeSchemaStrategy,
)

STRATEGY_NAMES = ("whole_schema", "bfs")

__all__ = [
    "STRATEGY_NAMES",
    "BfsRetrievalUnavailable",
    "BfsStrategy",
    "SelectionScope",
    "TableSelectionStrategy",
    "WholeSchemaStrategy",
]
