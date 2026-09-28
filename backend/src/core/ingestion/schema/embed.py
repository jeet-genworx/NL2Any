"""Embeds each table's generated description.

Takes the table_name -> description mapping produced by the description stage
and passes it through the local sentence-transformer model (all-MiniLM-L6-v2,
served by KoboldCpp's --embeddingsmodel endpoint), returning one vector per
table. Column descriptions are never embedded; the caller narrows to table
descriptions before calling in.

Pure: producing the vectors is here, storing them is the embedding
repository's job.
"""

from backend.src.control.providers.model.base import EmbeddingProvider
from backend.src.control.providers.model.embedding import KoboldCppEmbeddingProvider


async def embed_descriptions(
    descriptions: dict[str, str],
    provider: EmbeddingProvider | None = None,
) -> dict[str, list[float]]:
    """Embed each table's description, preserving the table_name -> vector mapping."""
    if not descriptions:
        return {}

    embed_provider = provider or KoboldCppEmbeddingProvider()
    table_names = list(descriptions)
    vectors = await embed_provider.embed([descriptions[name] for name in table_names])

    if len(vectors) != len(table_names):
        raise ValueError(
            f"Embedding provider returned {len(vectors)} vectors "
            f"for {len(table_names)} descriptions."
        )

    return dict(zip(table_names, vectors))
