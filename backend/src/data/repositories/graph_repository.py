"""Reads and writes the nodes/edges TOML views of a schema.

Two artifacts share this one shape: the plain graph (every foreign key, in
extraction order) and the MST (a cycle-free subset, in tree-traversal order).
They differ only in which key holds their edges, so they are written by the same
function and read back by the same function -- the description stage consumes
either without caring which it was handed.
"""

from pathlib import Path
from typing import Any

from backend.src.data.repositories import files

_MISSING_HINT = "Run 'uv run init-schema' first."


def save_graph_document(document: dict[str, Any], file_path: Path | str) -> Path:
    """Write a nodes/edges document (graph or MST) to a TOML file on disk."""
    return files.write_toml(file_path, document)


def load_nodes_and_edges(
    file_path: Path | str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Read the nodes and edges from a graph or MST TOML file.

    MST files store their edges under `mst_edges`, plain graph files under
    `edges`; the caller gets one list either way.
    """
    document = files.read_toml(file_path, hint=_MISSING_HINT)
    nodes = document.get("nodes", [])
    edges = document.get("mst_edges", document.get("edges", []))
    return nodes, edges
