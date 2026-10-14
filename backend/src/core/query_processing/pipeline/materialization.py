"""Turning a settled table selection into the slice the planner reads."""

from __future__ import annotations

import logging
from typing import Any

from backend.src.core.query_processing.pipeline.descriptions_client import DescriptionsClient
from backend.src.core.query_processing.pipeline.expansion import SchemaExpander
from backend.src.data.models.schema import DatabaseSchema, DatabaseType, Relationship
from backend.src.schemas.pipeline import RelevantSchema

logger = logging.getLogger(__name__)


def _target_slug(database: Any) -> str:
    return getattr(database, "key", None) or (
        database.value if isinstance(database, DatabaseType) else str(database)
    )


async def enrich_schema_descriptions(
    relevant_schema: RelevantSchema,
    database: Any,
    client: DescriptionsClient | None,
) -> None:
    """Fetch table and column descriptions from the Descriptions API and populate in place."""
    if not client:
        return

    slug = _target_slug(database)
    for obj in relevant_schema.objects:
        try:
            desc = await client.get_table_description(
                database_type=slug,
                table_name=obj.name,
            )
            if not desc:
                continue
            if desc.description:
                obj.description = desc.description
            for col_name, col_desc in desc.columns.items():
                if not col_desc:
                    continue
                field = obj.get_field(col_name)
                if field:
                    field.description = col_desc
        except Exception as err:
            logger.warning(
                "Failed to enrich descriptions for %s.%s: %s", slug, obj.name, err
            )


async def materialize_schema(
    *,
    selected_objects: list[str],
    relationships: list[Relationship],
    schema: DatabaseSchema,
    database: Any,
    expander: SchemaExpander,
    descriptions_client: DescriptionsClient | None,
) -> RelevantSchema:
    """Turn the selector's tables and relationships into the planner's slice.

    Nothing is added or removed here. The selector chose from the whole
    breadth-first neighborhood, bridge tables included, so its answer is the
    final word on which tables the query touches; this fills in the columns it
    did not see and the descriptions the Descriptions API holds.
    """
    relevant_schema = expander.materialize(
        selected_object_names=selected_objects,
        relationships=relationships,
        schema=schema,
    )
    await enrich_schema_descriptions(relevant_schema, database, descriptions_client)
    return relevant_schema
