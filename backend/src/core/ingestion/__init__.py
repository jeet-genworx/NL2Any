"""Ingestion: turns a live database into the schema artifacts query processing reads.

`pipeline` orchestrates the flow; `schema/` holds one module per stage.
"""

from backend.src.core.ingestion.pipeline import (
    extract_and_save_schema,
    run_ingestion_pipeline,
)

__all__ = ["extract_and_save_schema", "run_ingestion_pipeline"]
