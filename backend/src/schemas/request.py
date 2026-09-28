"""API request and input schemas."""

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """Request model for the natural language query endpoint."""

    database: str = Field(
        default="postgres",
        description="Database target: 'postgres' or 'mongo'",
    )
    question: str = Field(
        ...,
        description="Natural language question to query",
    )
