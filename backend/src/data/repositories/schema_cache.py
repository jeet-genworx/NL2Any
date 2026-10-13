"""In-memory cache and loader for canonical database schemas enriched with descriptions."""

import logging
from typing import Any

from backend.src.data.models.schema import DatabaseSchema, DatabaseType
from backend.src.data.models.targets import DatabaseTarget, as_target, resolve_target
from backend.src.data.repositories import description_repository, paths, schema_repository
from backend.src.core.ingestion.schema.describe import apply_descriptions
from backend.src.core.ingestion.schema.manager import get_default_schema_path

logger = logging.getLogger(__name__)


class SchemaCache:
    """Manages loading, description enrichment, and in-memory caching of DatabaseSchema instances."""

    def __init__(self, schemas: dict[Any, DatabaseSchema] | None = None) -> None:
        self._schemas: dict[Any, DatabaseSchema] = schemas if schemas is not None else {}

    def get_or_load(self, database: DatabaseTarget | DatabaseType | str) -> DatabaseSchema:
        """Load and cache canonical schema from TOML file, enriched with table/column descriptions."""
        if database in self._schemas:
            return self._schemas[database]

        try:
            target = as_target(database) if not isinstance(database, str) else resolve_target(database)
            if target.key in self._schemas:
                return self._schemas[target.key]
            schema_path = paths.schema_path(target)
            desc_path = paths.descriptions_path(target)
            key = target.key
        except Exception:
            schema_path = get_default_schema_path(database)
            desc_path = None
            key = database

        if not schema_path.exists():
            raise FileNotFoundError(
                f"Canonical schema not found at {schema_path}. Run 'init-schema' first."
            )
        schema = schema_repository.load_schema(schema_path)

        # Merge descriptions directly into schema in-memory
        if desc_path and desc_path.exists():
            try:
                descriptions = description_repository.load_descriptions(desc_path)
                apply_descriptions(schema, descriptions)
            except Exception as err:
                logger.warning("Failed to load descriptions from %s: %s", desc_path, err)

        self._schemas[key] = schema
        return schema
