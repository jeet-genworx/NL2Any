"""Canonical on-disk locations for every ingestion artifact.

Ingestion produces five files per database -- the schema TOML, its graph and
MST views, the generated descriptions, and the embedding vectors -- and each
one used to resolve its own path next to the code that wrote it, repeating both
the "postgres or mongo" filename choice and the package-directory lookup in five
modules. Collecting them here means a stage asks where its artifact lives rather
than deciding, and relocating the schema directory is a one-line change.
"""

from pathlib import Path

from backend.src.config import settings
from backend.src.data.models.schema import DatabaseType
from backend.src.data.models.targets import DatabaseTarget, as_target

TargetLike = DatabaseTarget | DatabaseType

# backend/src, used to locate packaged files when the process was not started
# from the repository root (containers, editors, pytest invoked elsewhere).
_SRC_DIR = Path(__file__).resolve().parents[2]


def _resolve(relative: str) -> Path:
    """Locate a file shipped inside backend/src, preferring a repo-root-relative
    path so a working copy overrides the installed one; falls back to the
    package's own directory, and to the relative path when neither exists (the
    caller then reports a missing file at the path it expected).
    """
    from_cwd = Path("backend/src") / relative
    if from_cwd.exists():
        return from_cwd
    from_package = _SRC_DIR / relative
    return from_package if from_package.exists() else from_cwd


def _slug(database: TargetLike) -> str:
    """Filename stem for a database: the prefix every artifact shares.

    Comes from the target, not the engine type, because two PostgreSQL targets
    must not share one set of schema and embedding files.
    """
    return as_target(database).file_stem


def schemas_dir() -> Path:
    """Directory holding the schema, graph, MST and description artifacts."""
    return _resolve("data/schemas")


def schema_path(database: TargetLike) -> Path:
    """Canonical schema TOML -- the file query processing reads."""
    return schemas_dir() / f"{_slug(database)}.toml"


def graph_path(database: TargetLike) -> Path:
    """Nodes/edges graph view of the schema."""
    return schemas_dir() / f"{_slug(database)}_graph.toml"


def mst_path(database: TargetLike) -> Path:
    """Minimum spanning forest over the schema graph.

    Only written for PostgreSQL targets: MongoDB has no foreign keys, so its
    graph has no edges to reduce.
    """
    return schemas_dir() / f"{_slug(database)}_mst.toml"


def descriptions_path(database: TargetLike) -> Path:
    """Generated table and column descriptions, as TOML."""
    return schemas_dir() / f"{_slug(database)}_descriptions.toml"


def embeddings_path(database: TargetLike) -> Path:
    """Table description vectors.

    Configurable, unlike the artifacts above, because query processing's
    EmbeddingStore reads this same file at query time and docker-compose
    bind-mounts it so regenerated vectors survive a container rebuild.
    """
    return Path(as_target(database).embeddings_path)


def describe_prompt_path() -> Path:
    """Prompt template driving the description stage."""
    return _resolve("core/query_processing/prompts/table_description.txt")
