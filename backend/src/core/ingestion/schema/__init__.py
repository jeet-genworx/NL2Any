"""Schema processing and generation package for ingestion."""

from backend.src.core.ingestion.schema.describe import (
    TableDescription,
    describe_tables,
    get_default_descriptions_path,
    save_descriptions,
)
from backend.src.core.ingestion.schema.embed import (
    embed_descriptions,
    get_default_embeddings_path,
    load_embeddings,
    save_embeddings,
)
from backend.src.core.ingestion.schema.graph import (
    build_schema_graph,
    get_default_graph_path,
    save_schema_graph,
)
from backend.src.core.ingestion.schema.manager import get_default_schema_path
from backend.src.core.ingestion.schema.mst import (
    compute_minimum_spanning_tree,
    get_default_mst_path,
    save_minimum_spanning_tree,
)
from backend.src.core.ingestion.schema.toml_store import load_schema_file, save_schema_file

__all__ = [
    "TableDescription",
    "build_schema_graph",
    "compute_minimum_spanning_tree",
    "describe_tables",
    "embed_descriptions",
    "get_default_descriptions_path",
    "get_default_embeddings_path",
    "get_default_graph_path",
    "get_default_mst_path",
    "get_default_schema_path",
    "load_embeddings",
    "load_schema_file",
    "save_descriptions",
    "save_embeddings",
    "save_minimum_spanning_tree",
    "save_schema_file",
    "save_schema_graph",
]
