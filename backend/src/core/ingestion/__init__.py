"""Ingestion pipeline package."""

from backend.src.core.ingestion.pipeline import (
    extract_and_save_schema,
    get_adapter,
    run_ingestion_pipeline,
)

__all__ = [
    "extract_and_save_schema",
    "get_adapter",
    "run_ingestion_pipeline",
]
