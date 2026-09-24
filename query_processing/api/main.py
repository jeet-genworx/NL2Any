"""FastAPI application for NL2AnyQuery."""

from typing import Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from query_processing.models.pipeline import PipelineResponse
from query_processing.models.schema import DatabaseType
from query_processing.pipeline.orchestrator import NL2AnyQueryOrchestrator
from ingestion.schema.manager import get_default_schema_path
from ingestion.schema.toml_store import load_schema_file

app = FastAPI(
    title="NL2AnyQuery API",
    description="Natural Language to Database Query Execution API",
    version="0.2.0",
)

orchestrator = NL2AnyQueryOrchestrator()


class QueryRequest(BaseModel):
    database: str = Field(default="postgres", description="Database target: 'postgres' or 'mongo'")
    question: str = Field(..., description="Natural language question to query")


@app.get("/health")
def health_check() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "ok", "version": "0.2.0"}


@app.post("/query", response_model=PipelineResponse)
async def query_endpoint(request: QueryRequest) -> PipelineResponse:
    """Execute natural language query pipeline."""
    try:
        response = await orchestrator.execute_pipeline(
            question=request.question,
            database=request.database,
        )
        return response
    except Exception as err:
        raise HTTPException(status_code=500, detail=str(err))


@app.get("/schema/{database_type}")
def get_schema_endpoint(database_type: str) -> dict[str, Any]:
    """Retrieve canonical semantic schema information."""
    db_clean = database_type.lower()
    if db_clean in ("postgres", "postgresql"):
        db_type = DatabaseType.POSTGRESQL
    elif db_clean in ("mongo", "mongodb"):
        db_type = DatabaseType.MONGODB
    else:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported database '{database_type}'. Use 'postgres' or 'mongo'.",
        )

    schema_path = get_default_schema_path(db_type)
    if not schema_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Schema file not found at {schema_path}. Run 'init-schema' first.",
        )

    schema = load_schema_file(schema_path)
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
