"""Query and ingestion API routes."""

from typing import Any
from fastapi import APIRouter, Depends, HTTPException

from backend.src.api.rest.dependencies import get_orchestrator, resolve_db_type
from backend.src.core.ingestion.pipeline import run_ingestion_pipeline
from backend.src.core.query_processing.pipeline.orchestrator import NL2AnyQueryOrchestrator
from backend.src.schemas.pipeline import PipelineResponse
from backend.src.schemas.request import QueryRequest

router = APIRouter(tags=["Query & Ingestion"])


@router.post("/ingest/{database_type}")
async def ingest_endpoint(database_type: str, use_mst: bool = True) -> dict[str, Any]:
    """Run the full ingestion pipeline for a database: extract metadata, build
    the schema graph (and MST, for Postgres), generate SLM table descriptions,
    and embed them. Triggered when a user selects a database in the frontend
    and clicks "Initialize Database". Requires a running KoboldCpp instance
    with an embeddings model loaded; this call blocks until the whole
    pipeline finishes (can take from several seconds to a couple of minutes).
    """
    db_type = resolve_db_type(database_type)
    try:
        return await run_ingestion_pipeline(db_type, use_mst=use_mst)
    except FileNotFoundError as err:
        raise HTTPException(status_code=404, detail=str(err))
    except ConnectionError as err:
        raise HTTPException(status_code=502, detail=str(err))
    except Exception as err:
        raise HTTPException(status_code=500, detail=str(err))


@router.post("/query", response_model=PipelineResponse)
async def query_endpoint(
    request: QueryRequest,
    orchestrator: NL2AnyQueryOrchestrator = Depends(get_orchestrator),
) -> PipelineResponse:
    """Execute natural language query pipeline."""
    try:
        response = await orchestrator.execute_pipeline(
            question=request.question,
            database=request.database,
        )
        return response
    except Exception as err:
        raise HTTPException(status_code=500, detail=str(err))
