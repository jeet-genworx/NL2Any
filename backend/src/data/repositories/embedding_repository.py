"""Reads and writes the table description embedding vectors.

A flat `{"customers": [0.01, ...]}` mapping, written compactly because it is
bulk numeric data no one reads by eye. Ingestion writes the same file query
processing's EmbeddingStore reads at query time, so regenerating embeddings
feeds live retrieval with no separate copy or sync step.
"""

from pathlib import Path

from backend.src.data.repositories import files

_MISSING_HINT = "Run 'uv run ingest-schema --database <postgres|mongo>' first."


def save_embeddings(embeddings: dict[str, list[float]], file_path: Path | str) -> Path:
    """Write the table_name -> vector mapping to a JSON file on disk."""
    return files.write_json(file_path, embeddings)


def load_embeddings(file_path: Path | str) -> dict[str, list[float]]:
    """Read a table_name -> vector mapping from a JSON file on disk."""
    return files.read_json_cached(file_path, hint=_MISSING_HINT)
