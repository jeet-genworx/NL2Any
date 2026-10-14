"""The planner's feedback loop back into table selection.

The planner is the first stage that reads the schema slice with the question in
hand, so it is the first that can notice a table the selector left out. When it
does, selection is asked again with that gap spelled out and the slice is
rebuilt -- bounded, because a selector that will not find the table will not
find it on the fourth pass either.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from typing import Any, NamedTuple

from backend.src.core.query_processing.pipeline.join_graph import JoinGraph
from backend.src.core.query_processing.pipeline.planner import QueryPlanner
from backend.src.core.query_processing.pipeline.selection import resolve_selection
from backend.src.core.query_processing.pipeline.selector import TableSelector
from backend.src.data.models.schema import DatabaseSchema, DatabaseType, Relationship
from backend.src.schemas.pipeline import (
    QuestionAnalysis,
    RelevantSchema,
    TableNeighborhood,
)

logger = logging.getLogger(__name__)

# Rebuilds the schema slice for a widened selection.
Materializer = Callable[[list[str], list[Relationship]], Awaitable[RelevantSchema]]


class PlanningOutcome(NamedTuple):
    """The plan, and the selection it ended up being planned against."""

    plan: Any
    selected_objects: list[str]
    relationships: list[Relationship]
    connector_objects: list[str]
    unjoinable_objects: list[str]
    relevant_schema: RelevantSchema
    extra_selection_retries: int


def _selector_feedback(missing: list[str], question: str) -> str:
    missing_names = ", ".join(missing)
    return (
        f"The query plan identified that table(s) [{missing_names}] are required to answer "
        f"the question '{question}', but their schema is not in the retrieved schema. "
        f"Please select ALL required tables (including {missing_names}) from the candidate tables."
    )


async def plan_with_table_feedback(
    *,
    question: str,
    analysis: QuestionAnalysis,
    relevant_schema: RelevantSchema,
    schema: DatabaseSchema,
    neighborhood: TableNeighborhood | None,
    graph: JoinGraph,
    db_type: DatabaseType,
    planner: QueryPlanner,
    selector: TableSelector,
    materialize: Materializer,
    max_retries: int,
    selected_objects: list[str],
    relationships: list[Relationship],
    connector_objects: list[str],
    unjoinable_objects: list[str],
    tables_are_human_approved: bool = False,
) -> PlanningOutcome:
    """Plan, and widen the selection while the planner reports missing tables.

    With `tables_are_human_approved` the selection is never widened: a human
    settled that list, and adding to it here would put tables into the query
    behind their back, which is the whole thing the approval gate exists to
    prevent. The gap is logged and the plan returned as it stands.
    """
    selected = list(selected_objects)
    rels = list(relationships)
    connectors = list(connector_objects)
    unjoinable = list(unjoinable_objects)
    extra_retries = 0
    query_plan = None

    for attempt in range(max_retries + 1):
        query_plan = await planner.plan(
            question_analysis=analysis,
            relevant_schema=relevant_schema,
            database_type=db_type,
        )
        if not query_plan.missing_tables:
            break

        if tables_are_human_approved:
            logger.warning(
                "Planner identified missing tables %s, but the tables were "
                "approved by a human and are not re-selected.",
                query_plan.missing_tables,
            )
            break

        if attempt >= max_retries:
            logger.warning(
                "Planner identified missing tables %s, but reached max retries.",
                query_plan.missing_tables,
            )
            break

        extra_retries += 1
        re_selection = await selector.select(
            question_analysis=analysis,
            neighborhood=neighborhood,
            schema=schema,
            feedback=_selector_feedback(query_plan.missing_tables, question),
        )
        if not re_selection.selected_objects:
            break

        widened = list(dict.fromkeys(selected + re_selection.selected_objects))
        if set(widened) == set(selected):
            break

        # Repaired and re-derived over the widened selection rather than merged,
        # so an edge can never outlive the tables it joins and a table added by
        # the planner's feedback still gets its bridge.
        selected, rels, added, dropped = resolve_selection(widened, neighborhood, schema, graph)
        connectors = list(dict.fromkeys(connectors + added))
        unjoinable = list(dict.fromkeys(unjoinable + dropped))
        relevant_schema = await materialize(selected, rels)

    return PlanningOutcome(
        plan=query_plan,
        selected_objects=selected,
        relationships=rels,
        connector_objects=connectors,
        unjoinable_objects=unjoinable,
        relevant_schema=relevant_schema,
        extra_selection_retries=extra_retries,
    )
