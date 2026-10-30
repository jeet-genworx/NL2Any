"""What the table selector is shown, and whether its prompt prefix is cacheable.

Stage 1 asks a model to pick tables. There are two defensible ways to set that
question up, and they pull in opposite directions:

* Show the model the whole schema every time. The prompt prefix is then
  byte-identical across questions, which is what makes the KoboldCpp KV-slot
  cache worth having -- the schema is evaluated once per database and restored
  from a slot thereafter.
* Show the model only the tables a question could plausibly need, found by
  embedding similarity and then widened along foreign keys. The prompt is far
  smaller, but it differs per question, so there is no constant prefix to cache.

Neither dominates: the first trades prompt size for cache hits, the second
trades cache hits for prompt size, and which wins depends on how large the
schema is and how well its table descriptions embed. So the choice is a
strategy, picked per request, rather than a decision baked into the selector.

A strategy answers one question -- what schema view goes into the prompt -- and
declares whether that view is stable enough to cache. Everything downstream is
unaffected: grounding, `package_selected_schema` and the FSM all keep working
against the full schema, so narrowing the view can only change which tables the
model is offered, never which tables are allowed to exist.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from backend.src.core.query_processing.pipeline.fsm.context import PipelineContext
from backend.src.data.models.schema import DatabaseSchema
from backend.src.schemas.pipeline import CandidateTable, TableNeighborhood


@dataclass
class SelectionScope:
    """The schema view for one table-selection attempt, plus its trace.

    `cache_key` is the identity of the prompt prefix built from `schema_view`:
    a value means "this prefix is stable for this key, the KV slot is valid",
    and None means the prefix varies per question and the cache must be
    bypassed. A strategy that narrows the schema must return None, otherwise a
    slot warmed on one question's tables would be restored for another's.
    """

    schema_view: DatabaseSchema
    cache_key: str | None = None
    candidates: list[CandidateTable] = field(default_factory=list)
    neighborhood: TableNeighborhood | None = None
    notice: str | None = None


@runtime_checkable
class TableSelectionStrategy(Protocol):
    """Decides what the table selector sees for a given attempt."""

    name: str

    async def scope(self, context: PipelineContext) -> SelectionScope:
        """Build the schema view for the current attempt.

        Called once per entry into TABLE_SELECTION, so a strategy sees every
        retry of the FSM's self-loop and may widen what it offers each time
        (`context.selector_retries` counts them).
        """
        ...
