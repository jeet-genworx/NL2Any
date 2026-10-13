"""Query and ingestion API routes."""

import logging
from typing import Any
from fastapi import APIRouter, Depends, HTTPException

from backend.src.api.rest.dependencies import get_orchestrator, resolve_db_target
from backend.src.control.providers.model.factory import get_available_providers
from backend.src.core.ingestion.pipeline import run_ingestion_pipeline
from backend.src.core.query_processing.pipeline.orchestrator import NL2AnyQueryOrchestrator
from backend.src.schemas.pipeline import PipelineResponse
from backend.src.schemas.request import QueryRequest, RerunRequest

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Query & Ingestion"])


@router.get("/providers")
async def list_providers_endpoint() -> list[dict[str, Any]]:
    """Return available LLM providers and their configuration status."""
    return get_available_providers()


@router.post("/ingest/{database_type}")
async def ingest_endpoint(
    database_type: str,
    use_mst: bool = True,
    force: bool = False,
) -> dict[str, Any]:
    """Run the full ingestion pipeline for a database: extract metadata, build
    the schema graph (and MST, for Postgres), generate SLM table descriptions,
    and embed them. Triggered when a user selects a database in the frontend
    and clicks "Initialize Database". Requires a running KoboldCpp instance
    with an embeddings model loaded; this call blocks until the whole
    pipeline finishes (can take from several seconds to a couple of minutes).

    Returns immediately with `"skipped": true` when this database already has
    embeddings on disk. Pass `force=true` to re-ingest regardless.
    """
    target = resolve_db_target(database_type)
    try:
        return await run_ingestion_pipeline(target, use_mst=use_mst, force=force)
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
            provider=request.provider,
            stages=request.stages,
            jargons=request.jargons,
        )
        return response
    except ValueError as err:
        raise HTTPException(status_code=400, detail=str(err))
    except RuntimeError as err:
        logger.error("Provider execution failed: %s", err)
        raise HTTPException(status_code=502, detail=str(err))
    except Exception as err:
        logger.exception("Unexpected error executing query: %s", err)
        raise HTTPException(status_code=500, detail=str(err))


@router.post("/rerun", response_model=PipelineResponse)
async def rerun_endpoint(
    request: RerunRequest,
    orchestrator: NL2AnyQueryOrchestrator = Depends(get_orchestrator),
) -> PipelineResponse:
    """Execute partial pipeline rerun from a specified stage using saved run state."""
    try:
        response = await orchestrator.rerun_pipeline(
            run_id=request.run_id,
            starting_stage=request.starting_stage,
            feedback=request.feedback,
        )
        return response
    except FileNotFoundError as err:
        raise HTTPException(status_code=404, detail=str(err))
    except ValueError as err:
        raise HTTPException(status_code=400, detail=str(err))
    except RuntimeError as err:
        logger.error("Provider execution failed during rerun: %s", err)
        raise HTTPException(status_code=502, detail=str(err))
    except Exception as err:
        logger.exception("Unexpected error executing rerun: %s", err)
        raise HTTPException(status_code=500, detail=str(err))
