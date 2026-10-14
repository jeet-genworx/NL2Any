"""Candidate retrieval, neighborhood expansion and table selection.

Stages 5 to 7 of the pipeline, plus the deterministic repair that makes the
selector's answer joinable. Kept apart from the orchestrator because this is
the one part of the run that loops on its own terms: a retry here re-embeds a
hint, re-walks the foreign-key graph and asks again, rather than simply
repeating the previous call.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from backend.src.config import settings
from backend.src.control.providers.embedding.base import EmbeddingProvider
from backend.src.core.query_processing.pipeline.approval import assess_selection
from backend.src.core.query_processing.pipeline.join_graph import (
    JoinGraph,
    build_table_neighborhood,
    connect_selection,
)
from backend.src.core.query_processing.pipeline.selector import (
    TableSelector,
    filter_relationships,
)
from backend.src.core.query_processing.retrieval.vector import VectorRetriever
from backend.src.data.models.schema import DatabaseSchema, Relationship
from backend.src.schemas.pipeline import (
    CandidateTable,
    QuestionAnalysis,
    SelectionUncertainty,
    TableNeighborhood,
)

logger = logging.getLogger(__name__)


@dataclass
class SelectionOutcome:
    """What table selection settled on, including the trace behind it."""

    selected_objects: list[str] = field(default_factory=list)
    relationships: list[Relationship] = field(default_factory=list)
    connector_objects: list[str] = field(default_factory=list)
    unjoinable_objects: list[str] = field(default_factory=list)
    candidates: list[CandidateTable] = field(default_factory=list)
    neighborhood: TableNeighborhood | None = None
    retries: int = 0
    notice: str | None = None
    error: str | None = None
    uncertainty: SelectionUncertainty | None = None


def build_neighborhood(
    candidates: list[CandidateTable],
    schema: DatabaseSchema,
    graph: JoinGraph,
) -> TableNeighborhood:
    """Walk the foreign-key graph out from the embedding matches.

    The embedding matches alone are not the candidate space: a bridge table is
    never semantically similar to the question, so it is never retrieved, so it
    can never be selected, and the generator ends up inventing the join.
    Breadth-first traversal from each match supplies those tables, and the
    single de-duplicated set it produces is what the selector chooses from.
    """
    neighborhood = build_table_neighborhood(
        seed_tables=[(c.table_name, c.similarity, c.rank) for c in candidates],
        schema=schema,
        max_levels=settings.table_neighborhood_max_levels,
        max_tables=settings.table_neighborhood_max_tables,
        graph=graph,
    )
    logger.info(
        "Table neighborhood: %d seed(s) -> %d table(s) within %d hop(s)%s.",
        len(neighborhood.seed_objects),
        len(neighborhood.tables),
        neighborhood.max_levels,
        f", {len(neighborhood.discarded_objects)} discarded by the cap"
        if neighborhood.truncated
        else "",
    )
    return neighborhood


def resolve_selection(
    selected_objects: list[str],
    neighborhood: TableNeighborhood | None,
    schema: DatabaseSchema,
    graph: JoinGraph,
) -> tuple[list[str], list[Relationship], list[str], list[str]]:
    """Make the selection joinable, then derive its relationships.

    The selector is told to include the intermediate tables its choice needs,
    and it does not reliably do so -- it will name a bridge in its reasoning and
    leave it out of the answer. Whether two tables can be joined is a fact about
    the schema, not a judgement, so the gap is closed deterministically here,
    walking only tables the selector was shown. Relationships are derived
    afterwards, over the repaired set, so they describe what the planner
    actually receives.
    """
    if not selected_objects:
        return [], [], [], []

    allowed = neighborhood.table_names() if neighborhood else None
    resolved, added, unjoinable = connect_selection(
        selected_objects,
        schema,
        graph=graph,
        allowed=allowed,
    )
    if added:
        logger.info(
            "Selection was not joinable; added bridge table(s) %s to connect %s.",
            added,
            selected_objects,
        )
    if unjoinable:
        logger.warning(
            "Dropped %s from the selection: no foreign-key path to the rest "
            "of the query, so any join would have to be invented.",
            unjoinable,
        )
    source = neighborhood.relationships if neighborhood else schema.relationships
    return resolved, filter_relationships(resolved, source), added, unjoinable


async def _retrieve_candidates(
    analysis: QuestionAnalysis,
    query_vector: list[float],
    retriever: VectorRetriever,
    embedding_provider: EmbeddingProvider,
) -> list[CandidateTable]:
    """Embedding matches for the question, with a keyword hint as a fallback.

    A question whose phrasing matches nothing can still name its subject, so a
    second pass embeds just the nouns and subjective terms before giving up.
    """
    candidates = retriever.retrieve(query_vector)
    if candidates or not (analysis.subjective or analysis.nouns):
        return candidates

    fallback_hint = " ".join(dict.fromkeys(analysis.subjective + analysis.nouns))
    try:
        fallback_vec = await embedding_provider.embed(fallback_hint)
        return retriever.retrieve(fallback_vec)
    except Exception as fallback_err:
        logger.warning("Fallback candidate retrieval failed: %s", fallback_err)
        return candidates


async def _widen_candidates(
    question: str,
    selection,
    current_candidates: list[CandidateTable],
    retriever: VectorRetriever,
    embedding_provider: EmbeddingProvider,
) -> list[CandidateTable]:
    """Merge fresh matches for the selector's hint into the candidate set.

    The hint is what the selector said it was missing, so the new matches are
    additional BFS seeds rather than a replacement: the next attempt is offered
    a wider neighborhood, never a narrower one.
    """
    hint = selection.retrieval_hint or f"{question} {' '.join(selection.missing_objects)}"
    try:
        hint_vec = await embedding_provider.embed(hint)
        new_cands = retriever.retrieve(hint_vec)
    except Exception as embed_err:
        logger.warning("Retry embedding retrieval failed: %s", embed_err)
        new_cands = []

    merged = {c.table_name: c.similarity for c in current_candidates}
    for nc in new_cands:
        if nc.table_name not in merged or nc.similarity > merged[nc.table_name]:
            merged[nc.table_name] = nc.similarity

    sorted_merged = sorted(merged.items(), key=lambda x: x[1], reverse=True)[
        : settings.effective_similarity_top_k
    ]
    return [
        CandidateTable(table_name=tbl, similarity=sim, rank=r)
        for r, (tbl, sim) in enumerate(sorted_merged, start=1)
    ]


async def retrieve_and_select(
    *,
    question: str,
    analysis: QuestionAnalysis,
    query_vector: list[float],
    schema: DatabaseSchema,
    graph: JoinGraph,
    selector: TableSelector,
    retriever: VectorRetriever,
    embedding_provider: EmbeddingProvider,
    max_retries: int,
) -> SelectionOutcome:
    """Embedding match -> BFS neighborhood -> table selector, with bounded retry."""
    try:
        initial_candidates = await _retrieve_candidates(
            analysis, query_vector, retriever, embedding_provider
        )
    except Exception as err:
        logger.error("Vector retrieval failed: %s", err)
        return SelectionOutcome(error=f"Vector retrieval failed: {err}")

    if not initial_candidates:
        return SelectionOutcome(
            error="I could not find any relevant tables in the database schema matching your question."
        )

    current_candidates = list(initial_candidates)
    neighborhood = build_neighborhood(current_candidates, schema, graph)
    attempt = 0
    feedback: str | None = None

    while True:
        selection = await selector.select(
            question_analysis=analysis,
            neighborhood=neighborhood,
            schema=schema,
            feedback=feedback,
        )
        resolved, relationships, added, unjoinable = resolve_selection(
            selection.selected_objects, neighborhood, schema, graph
        )

        exhausted = attempt >= max_retries
        if not (selection.sufficient and resolved):
            if not exhausted:
                attempt += 1
                current_candidates = await _widen_candidates(
                    question, selection, current_candidates, retriever, embedding_provider
                )
                neighborhood = build_neighborhood(current_candidates, schema, graph)
                missing_info = (
                    f" Missing: {selection.missing_objects}."
                    if selection.missing_objects
                    else ""
                )
                feedback = (
                    f"Previous attempt selected {selection.selected_objects} but was marked "
                    f"insufficient. Reason: {selection.reason}.{missing_info}"
                )
                continue

            logger.warning(
                "Table selector still reports insufficiency after %d retries: %s",
                attempt,
                selection.reason,
            )
            if not resolved:
                return SelectionOutcome(
                    candidates=current_candidates,
                    neighborhood=neighborhood,
                    retries=attempt,
                    error="Relevant schema could not be identified reliably after maximum retries.",
                )
            # A joinable selection is worth attempting even when the model
            # doubts it: the usual reason it reports insufficiency is a join it
            # could not see, which the repair above has already settled.
            # Failing here instead would discard a usable answer, so the doubt
            # is carried on the outcome rather than ending the run -- and it is
            # what a caller willing to ask a human stops on.

        # A selection the model is happy with can still have been salvaged -- a
        # name it invented was dropped, or the stage failed and the top
        # embedding match stood in for an answer.
        uncertainty = assess_selection(
            selection, unjoinable, exhausted_retries=not (selection.sufficient and resolved)
        )
        return SelectionOutcome(
            selected_objects=resolved,
            relationships=relationships,
            connector_objects=added,
            unjoinable_objects=unjoinable,
            candidates=current_candidates,
            neighborhood=neighborhood,
            retries=attempt,
            notice=uncertainty.detail if uncertainty else None,
            uncertainty=uncertainty,
        )
