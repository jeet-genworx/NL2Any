"""Schema metadata API routes: the canonical schema and its generated
documentation.

Read-only views over the artifacts ingestion produces. Nothing here talks to a
live database or to the model; if an artifact is missing, the response says
which command regenerates it.
"""

from typing import Any

from fastapi import APIRouter, HTTPException

from backend.src.api.rest.dependencies import resolve_db_type
from backend.src.data.models.schema import DatabaseSchema, DatabaseType
from backend.src.data.repositories import (
    description_repository,
    paths,
    schema_repository,
)
from backend.src.schemas.metadata import TableDescriptionResponse

router = APIRouter(tags=["Schema"])


def _load_schema(db_type: DatabaseType) -> DatabaseSchema:
    """Load the canonical schema, or fail with a 404 naming the fix."""
    schema_path = paths.schema_path(db_type)
    if not schema_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Schema file not found at {schema_path}. Run 'init-schema' first.",
        )
    return schema_repository.load_schema(schema_path)


@router.get("/schema/{database_type}")
def get_schema_endpoint(database_type: str) -> dict[str, Any]:
    """Retrieve canonical semantic schema information."""
    schema = _load_schema(resolve_db_type(database_type))
    return {
        "database_type": schema.database_type.value,
        "database_name": schema.database_name,
        "schema_version": schema.schema_version,
        "last_updated": schema.last_updated,
        "description_generated_at": schema.description_generated_at,
        "object_count": len(schema.objects),
        "objects": [
            {
                "name": obj.name,
                "kind": obj.kind.value,
                "description": obj.description,
                "fields": [
                    {
                        "name": f.name,
                        "type": f.type,
                        "description": f.description,
                        "nullable": f.nullable,
                    }
                    for f in obj.fields
                ],
            }
            for obj in schema.objects
        ],
        "relationships": [
            {
                "from_object": r.from_object,
                "from_field": r.from_field,
                "to_object": r.to_object,
                "to_field": r.to_field,
                "type": r.relationship_type,
            }
            for r in schema.relationships
        ],
    }


@router.get(
    "/descriptions/{database_type}/{table_name}",
    response_model=TableDescriptionResponse,
)
def get_table_description_endpoint(
    database_type: str,
    table_name: str,
) -> TableDescriptionResponse:
    """Return one table's description and a description for each of its columns.

    Served entirely from the documentation TOML that ingestion wrote: nothing
    here re-reads the schema or recomputes anything the file already holds, and
    the parse itself is reused until the file changes on disk.

    Every column of the table is listed -- MongoDB subdocument fields as dotted
    paths -- with an empty string where a description has not been generated yet.
    """
    db_type = resolve_db_type(database_type)

    try:
        documentation = description_repository.load_documentation(
            paths.descriptions_path(db_type)
        )
    except FileNotFoundError as err:
        raise HTTPException(status_code=404, detail=str(err))

    table = documentation.objects.get(table_name)
    if table is None:
        raise HTTPException(
            status_code=404,
            detail=(
                f"'{table_name}' not found in the {db_type.value} documentation. "
                f"Available: {', '.join(sorted(documentation.objects))}."
            ),
        )

    return TableDescriptionResponse(
        database_type=documentation.database_type.value,
        database_name=documentation.database_name,
        table=table_name,
        kind=documentation.object_kind.value,
        description=table.description,
        column_count=len(table.columns),
        columns=table.columns,
    )
