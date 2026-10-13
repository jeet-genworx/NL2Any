"""Query and ingestion API routes."""

from typing import Any
from fastapi import APIRouter, Depends, HTTPException

from backend.src.api.rest.dependencies import get_orchestrator, resolve_db_target
from backend.src.core.ingestion.pipeline import run_ingestion_pipeline
from backend.src.core.query_processing.pipeline.orchestrator import NL2AnyQueryOrchestrator
from backend.src.core.query_processing.pipeline.approval import PendingRunNotFound
from backend.src.schemas.pipeline import PipelineResponse
from backend.src.schemas.request import QueryRequest, TableApprovalDecision

router = APIRouter(tags=["Query & Ingestion"])


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
    """Execute natural language query pipeline.

    Normally returns the finished run. With `allow_table_approval` set on the
    request, a run whose table selection is uncertain comes back instead with
    `status="awaiting_table_approval"` and an `approval` block; finish it by
    posting the chosen tables to `/query/{run_id}/tables`.

    `provider` sets the model for the whole run; `selector_provider` and
    `generator_provider` override it for table selection and for query
    generation, so the two stages can run on different models. The response
    reports what actually ran each stage on `providers`.
    """
    try:
        response = await orchestrator.execute_pipeline(
            question=request.question,
            database=request.database,
            provider=request.provider,
            selector_provider=request.selector_provider,
            generator_provider=request.generator_provider,
            jargons=request.jargons,
            allow_table_approval=request.allow_table_approval,
        )
        return response
    except ValueError as err:
        raise HTTPException(status_code=400, detail=str(err))
    except Exception as err:
        raise HTTPException(status_code=500, detail=str(err))


@router.post("/query/{run_id}/tables", response_model=PipelineResponse)
async def approve_tables_endpoint(
    run_id: str,
    decision: TableApprovalDecision,
    orchestrator: NL2AnyQueryOrchestrator = Depends(get_orchestrator),
) -> PipelineResponse:
    """Settle the tables of a run paused at the table gate and finish it.

    The run is held in the API process only, and only until it expires, so a
    `run_id` that is unknown here is either wrong or too old; either way there
    is nothing to resume and the question has to be asked again.
    """
    try:
        return await orchestrator.resume_pipeline(
            run_id=run_id,
            approved_objects=decision.approved_objects,
        )
    except PendingRunNotFound:
        raise HTTPException(
            status_code=404,
            detail=(
                f"No run awaiting table approval with id {run_id}. It may have "
                "expired or already been resumed; please re-run the question."
            ),
        )
    except ValueError as err:
        raise HTTPException(status_code=400, detail=str(err))
    except Exception as err:
        raise HTTPException(status_code=500, detail=str(err))
