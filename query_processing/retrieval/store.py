"""Embedding store for reading and validating precomputed table embeddings."""

import json
import logging
from pathlib import Path

from query_processing.core.config import settings

logger = logging.getLogger(__name__)

EXPECTED_DIMENSION = 384  # for all-MiniLM-L6-v2


class EmbeddingStore:
    """Safely loads and validates precomputed table description embeddings from JSON."""

    def __init__(
        self,
        file_path: str | Path | None = None,
        expected_dim: int = EXPECTED_DIMENSION,
    ) -> None:
        self.file_path = Path(file_path or settings.postgres_embeddings_path)
        self.expected_dim = expected_dim
        self._embeddings: dict[str, list[float]] = {}
        self._loaded: bool = False

    def load(self) -> dict[str, list[float]]:
        """Load and validate the embedding JSON file."""
        if not self.file_path.exists():
            raise FileNotFoundError(f"Embedding file not found: {self.file_path}")

        try:
            raw_text = self.file_path.read_text(encoding="utf-8")
            data = json.loads(raw_text)
        except json.JSONDecodeError as err:
            raise ValueError(f"Malformed JSON in embedding file {self.file_path}: {err}") from err

        if not isinstance(data, dict):
            raise ValueError(
                f"Embedding file root must be a JSON object/dictionary, got {type(data).__name__}"
            )

        if not data:
            raise ValueError(f"Embedding file {self.file_path} is empty or contains no tables.")

        validated: dict[str, list[float]] = {}
        for table_name, vec in data.items():
            if not isinstance(table_name, str) or not table_name.strip():
                raise ValueError(f"Invalid table name in embedding file: {table_name!r}")

            if not isinstance(vec, list):
                raise ValueError(
                    f"Embedding for table '{table_name}' must be a list, got {type(vec).__name__}"
                )

            if len(vec) != self.expected_dim:
                raise ValueError(
                    f"Embedding for table '{table_name}' has dimension {len(vec)}, expected {self.expected_dim}"
                )

            float_vec: list[float] = []
            for idx, val in enumerate(vec):
                if not isinstance(val, (int, float)) or isinstance(val, bool):
                    raise ValueError(
                        f"Vector for table '{table_name}' has non-numeric value at index {idx}: {val!r}"
                    )
                float_vec.append(float(val))

            validated[table_name.strip()] = float_vec

        self._embeddings = validated
        self._loaded = True
        logger.info("Loaded embeddings for %d tables from %s", len(self._embeddings), self.file_path)
        return self._embeddings

    def get(self, table_name: str) -> list[float] | None:
        """Get embedding for a specific table."""
        if not self._loaded:
            self.load()
        return self._embeddings.get(table_name)

    def all_embeddings(self) -> dict[str, list[float]]:
        """Return a copy of all loaded table embeddings."""
        if not self._loaded:
            self.load()
        return dict(self._embeddings)
