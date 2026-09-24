"""Embeds each table's SLM-generated description.

Takes the table_name -> description JSON produced by describe.py and passes
each description through the local sentence-transformer embedding model
(all-MiniLM-L6-v2-Q8_0.gguf, served by KoboldCpp's --embeddingsmodel
endpoint), storing the resulting vectors as a flat table_name -> embedding
JSON file.

For PostgreSQL, this writes directly to settings.postgres_embeddings_path --
the same file query_processing.retrieval.store.EmbeddingStore reads at query
time -- so running the ingestion pipeline feeds live retrieval directly, with
no separate copy/sync step. MongoDB has no equivalent live consumer yet (its
retrieval still uses the legacy BM25 path), so its embeddings are written
under query_processing/embeddings/ for now.
"""

import json
from pathlib import Path
from typing import Any, Protocol, runtime_checkable
import httpx

from query_processing.core.config import settings
from query_processing.models.schema import DatabaseType


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Protocol defining interface for text embedding providers."""

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts.

        Args:
            texts: Texts to embed, in order.

        Returns:
            One embedding vector per input text, in the same order.
        """
        ...


class KoboldCppEmbeddingProvider:
    """Embedding provider communicating with KoboldCpp's OpenAI-compatible /v1/embeddings API."""

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        timeout: float | None = None,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self.base_url = (base_url or settings.koboldcpp_base_url).rstrip("/")
        self.model = model or settings.koboldcpp_embedding_model
        self.timeout = (
            timeout if timeout is not None else max(60.0, float(settings.query_timeout_seconds))
        )
        self._external_client = http_client

    async def _get_client(self) -> httpx.AsyncClient:
        if self._external_client is not None:
            return self._external_client
        return httpx.AsyncClient(timeout=self.timeout)

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts via KoboldCpp's embeddings endpoint.

        Returns one embedding vector per input text, in the same order.
        """
        payload: dict[str, Any] = {"model": self.model, "input": texts}
        endpoint = f"{self.base_url}/embeddings"
        client = await self._get_client()

        try:
            if self._external_client is None:
                async with client:
                    response = await client.post(endpoint, json=payload)
            else:
                response = await client.post(endpoint, json=payload)

            response.raise_for_status()
            data = response.json()
            items = data.get("data", [])
            if not items:
                raise ValueError("KoboldCpp returned an embeddings response with no data.")

            items.sort(key=lambda item: item.get("index", 0))
            return [item["embedding"] for item in items]

        except httpx.ConnectError as err:
            raise ConnectionError(
                f"Failed to connect to KoboldCpp at '{self.base_url}'. "
                f"Please ensure KoboldCpp is running with --embeddingsmodel: {err}"
            ) from err
        except httpx.TimeoutException as err:
            raise TimeoutError(
                f"KoboldCpp embeddings request to '{self.base_url}' timed out after {self.timeout}s: {err}"
            ) from err
        except httpx.HTTPStatusError as err:
            raise RuntimeError(
                f"KoboldCpp embeddings HTTP {err.response.status_code} error: {err.response.text}"
            ) from err
        except Exception as err:
            if not isinstance(err, (ConnectionError, TimeoutError, RuntimeError, ValueError)):
                raise RuntimeError(f"Unexpected error communicating with KoboldCpp embeddings: {err}") from err
            raise



def get_default_embeddings_path(database_type: DatabaseType) -> Path:
    """Return canonical embeddings JSON path for the database type.

    PostgreSQL: settings.postgres_embeddings_path, the exact file
    query_processing's live EmbeddingStore/VectorRetriever read from.
    MongoDB: query_processing/embeddings/mongo_embeddings.json (not yet
    wired into a live consumer).
    """
    if database_type == DatabaseType.POSTGRESQL:
        return Path(settings.postgres_embeddings_path)
    return Path("query_processing/embeddings") / "mongo_embeddings.json"


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


def load_descriptions(file_path: Path | str) -> dict[str, str]:
    """Load a table_name -> description mapping from a JSON file on disk."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Descriptions file not found at: {path}. Run 'uv run describe-schema' first."
        )
    return json.loads(path.read_text(encoding="utf-8"))


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
