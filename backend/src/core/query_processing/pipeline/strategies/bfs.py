"""Narrow the selector's view to the embedding matches and their FK neighborhood.

Two steps, in this order for a reason. Embedding similarity finds the tables a
question *sounds like*; a breadth-first walk of the foreign-key graph then adds
the tables a query over them has to *touch*. The second step is not an
optimization of the first -- it covers a case the first structurally cannot. A
bridge table's description never resembles any question, so it is never
retrieved, so it can never be selected, and the generator is left to invent the
join. Walking out from each match supplies it.

The prompt this produces differs per question, so the KV prefix cache is off
here: `cache_key` is always None. That is the trade, and it is why this is a
strategy rather than the behaviour.

PostgreSQL only. The walk is over foreign keys, and MongoDB has none -- the
neighborhood of any seed would be the seed itself, so this would cost an
embedding call to arrive at a worse version of the whole-schema view.
"""

from __future__ import annotations

import logging

from backend.src.config import settings
from backend.src.control.providers.embedding.base import EmbeddingProvider
from backend.src.core.query_processing.pipeline.fsm.context import PipelineContext
from backend.src.core.query_processing.pipeline.join_graph import (
    JoinGraph,
    build_table_neighborhood,
)
from backend.src.core.query_processing.pipeline.strategies.base import (
    SelectionScope,
    TableSelectionStrategy,
)
from backend.src.core.query_processing.retrieval.vector import VectorRetriever
from backend.src.data.models.schema import DatabaseSchema
from backend.src.schemas.pipeline import CandidateTable

logger = logging.getLogger(__name__)

STRATEGY_NAME = "bfs"


class BfsRetrievalUnavailable(Exception):
    """Raised when this strategy cannot run for a target, so the caller can fall back."""


def narrow_schema(schema: DatabaseSchema, names: list[str]) -> DatabaseSchema:
    """A schema holding only `names`, with the relationships among them.

    Built as a real DatabaseSchema rather than a formatted string so the
    selector needs no change: it formats whatever schema it is handed, and
    grounding against this narrowed copy is what stops the model from naming a
    table it was never shown.
    """
    wanted = {n.lower() for n in names}
    objects = [obj for obj in schema.objects if obj.name.lower() in wanted]
    kept = {obj.name.lower() for obj in objects}
    relationships = [
        rel
        for rel in schema.relationships
        if rel.from_object.lower() in kept and rel.to_object.lower() in kept
    ]
    return DatabaseSchema(
        database_type=schema.database_type,
        database_name=schema.database_name,
        schema_version=schema.schema_version,
        last_updated=schema.last_updated,
        description_generated_at=schema.description_generated_at,
        objects=objects,
        relationships=relationships,
    )


class BfsStrategy(TableSelectionStrategy):
    """Embedding retrieval, then a bounded foreign-key walk, then a narrowed schema."""

    name = STRATEGY_NAME

    def __init__(
        self,
        retriever: VectorRetriever,
        embedding_provider: EmbeddingProvider,
        graph: JoinGraph | None = None,
    ) -> None:
        self.retriever = retriever
        self.embedding_provider = embedding_provider
        self.graph = graph
        # Carried between attempts so a retry widens the candidate set rather
        # than recomputing the same one. The FSM re-enters TABLE_SELECTION with
        # the selector's complaint in `s1_system_retry`; without this the
        # second attempt would be handed exactly the tables that just failed.
        self._candidates: dict[str, float] = {}

    async def _embed(self, text: str) -> list[float]:
        return await self.embedding_provider.embed(text)

    async def _seed_candidates(self, question: str) -> list[CandidateTable]:
        """Embedding matches for the question."""
        vector = await self._embed(question)
        return self.retriever.retrieve(vector)

    async def _widen(self, hint: str) -> list[CandidateTable]:
        """Merge fresh matches for a hint into the running candidate set.

        Merged, never replaced: a retry must offer the selector a wider view
        than the attempt that failed, never a different narrow one.
        """
        try:
            new_candidates = self.retriever.retrieve(await self._embed(hint))
        except Exception as err:
            logger.warning("BFS retry retrieval failed, keeping previous candidates: %s", err)
            new_candidates = []

        for cand in new_candidates:
            prior = self._candidates.get(cand.table_name)
            if prior is None or cand.similarity > prior:
                self._candidates[cand.table_name] = cand.similarity

        ranked = sorted(self._candidates.items(), key=lambda kv: kv[1], reverse=True)
        ranked = ranked[: settings.effective_similarity_top_k]
        return [
            CandidateTable(table_name=name, similarity=sim, rank=rank)
            for rank, (name, sim) in enumerate(ranked, start=1)
        ]

    async def scope(self, context: PipelineContext) -> SelectionScope:
        schema = context.schema
        question = context.processed_question or context.question

        try:
            if context.selector_retries and self._candidates:
                hint = context.s1_system_retry or question
                candidates = await self._widen(hint)
            else:
                candidates = await self._seed_candidates(question)
                self._candidates = {c.table_name: c.similarity for c in candidates}
        except Exception as err:
            raise BfsRetrievalUnavailable(f"Vector retrieval failed: {err}") from err

        if not candidates:
            raise BfsRetrievalUnavailable(
                "No table matched the question above the similarity threshold."
            )

        neighborhood = build_table_neighborhood(
            seed_tables=[(c.table_name, c.similarity, c.rank) for c in candidates],
            schema=schema,
            max_levels=settings.table_neighborhood_max_levels,
            max_tables=settings.table_neighborhood_max_tables,
            graph=self.graph,
        )
        names = neighborhood.table_names()
        if not names:
            raise BfsRetrievalUnavailable(
                "The foreign-key walk produced no tables for the retrieved matches."
            )

        logger.info(
            "BFS scope: %d candidate(s) -> %d table(s) within %d hop(s) "
            "(of %d in schema)%s.",
            len(candidates),
            len(names),
            neighborhood.max_levels,
            len(schema.objects),
            f", {len(neighborhood.discarded_objects)} discarded by the cap"
            if neighborhood.truncated
            else "",
        )

        return SelectionScope(
            schema_view=narrow_schema(schema, names),
            # Always None: the prefix is question-specific, so a KV slot warmed
            # for one question must never be restored for another.
            cache_key=None,
            candidates=candidates,
            neighborhood=neighborhood,
        )
