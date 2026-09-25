"""Embeds each table's SLM-generated description.

Takes the table_name -> description JSON produced by describe.py and passes
each description through the local sentence-transformer embedding model
(all-MiniLM-L6-v2-Q8_0.gguf, served by KoboldCpp's --embeddingsmodel
endpoint), storing the resulting vectors as a flat table_name -> embedding
JSON file.

Writes directly to the same file query_processing.retrieval.store.EmbeddingStore
reads at query time -- settings.postgres_embeddings_path for PostgreSQL and
settings.mongo_embeddings_path for MongoDB -- so running the ingestion pipeline
feeds live retrieval directly, with no separate copy/sync step.
"""

import json
from pathlib import Path

from query_processing.core.config import settings
from query_processing.models.schema import DatabaseType
from query_processing.providers.model.base import EmbeddingProvider
from query_processing.providers.model.embedding import KoboldCppEmbeddingProvider
from ingestion.schema.describe import TableDescription, parse_table_entry


def get_default_embeddings_path(database_type: DatabaseType) -> Path:
    """Return canonical embeddings JSON path for the database type.

    Both point at the exact files query_processing's live
    EmbeddingStore/VectorRetriever read from.
    """
    if database_type == DatabaseType.POSTGRESQL:
        return Path(settings.postgres_embeddings_path)
    return Path(settings.mongo_embeddings_path)


async def embed_descriptions(
    descriptions: dict[str, str],
    provider: EmbeddingProvider | None = None,
) -> dict[str, list[float]]:
    """Embed each table's description, preserving table_name -> vector mapping."""
    if not descriptions:
        return {}

    embed_provider = provider or KoboldCppEmbeddingProvider()
    table_names = list(descriptions.keys())
    vectors = await embed_provider.embed([descriptions[name] for name in table_names])

    if len(vectors) != len(table_names):
        raise ValueError(
            f"Embedding provider returned {len(vectors)} vectors for {len(table_names)} descriptions."
        )

    return dict(zip(table_names, vectors))


def load_descriptions(file_path: Path | str) -> dict[str, TableDescription]:
    """Load the generated documentation from a JSON file on disk.

    Accepts the current shape, where each table maps to its description plus a
    description per column, as well as the older flat table_name -> description
    shape, so descriptions files written before columns were generated still
    load (they simply carry no column descriptions).
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Descriptions file not found at: {path}. Run 'uv run describe-schema' first."
        )
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {name: parse_table_entry(entry) for name, entry in raw.items()}


def save_embeddings(embeddings: dict[str, list[float]], file_path: Path | str) -> None:
    """Write the table_name -> embedding mapping to a JSON file on disk."""
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(embeddings, ensure_ascii=False), encoding="utf-8")


def load_embeddings(file_path: Path | str) -> dict[str, list[float]]:
    """Load a table_name -> embedding mapping from a JSON file on disk.

    Read by query processing's semantic retriever at query time.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Embeddings file not found at: {path}. "
            f"Run 'uv run ingest-schema --database <postgres|mongo>' first."
        )
    return json.loads(path.read_text(encoding="utf-8"))
