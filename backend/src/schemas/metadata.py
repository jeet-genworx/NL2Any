"""Response models for the schema metadata endpoints."""

from pydantic import BaseModel, Field


class TableDescriptionResponse(BaseModel):
    """One table's generated documentation: its own description, and one per column."""

    database_type: str
    database_name: str
    table: str
    kind: str = Field(description="'table' for PostgreSQL, 'collection' for MongoDB.")
    description: str
    column_count: int
    columns: dict[str, str] = Field(
        description=(
            "Column name -> description, covering every column of the table. "
            "MongoDB subdocument fields appear as dotted paths. A column whose "
            "description has not been generated yet maps to an empty string."
        )
    )
