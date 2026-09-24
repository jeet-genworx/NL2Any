"""Computes a minimum spanning tree over the schema graph.

Edges (foreign-key relationships) are treated as uniform weight, so the MST
is simply a minimal, cycle-free set of edges that keeps every FK-connected
table reachable. The schema graph is not guaranteed to be fully connected
(tables with no FKs at all form isolated nodes), so this naturally produces a
minimum spanning FOREST -- one spanning tree per connected component -- via
Kruskal's algorithm with union-find. Postgres-only: MongoDB's graph has no
edges (no foreign-key concept), so there's nothing to reduce.
"""

from collections import deque
from pathlib import Path
from typing import Any
import tomli_w

from query_processing.models.schema import DatabaseSchema, DatabaseType
from ingestion.schema.graph import build_schema_graph


def get_default_mst_path(database_type: DatabaseType) -> Path:
    """Return canonical MST TOML path for the database type."""
    filename = "postgres_mst.toml" if database_type == DatabaseType.POSTGRESQL else "mongo_mst.toml"
    return Path("ingestion/schemas") / filename


class _UnionFind:
    """Disjoint-set structure with path compression and union by attachment."""

    def __init__(self, items: list[str]) -> None:
        self._parent = {item: item for item in items}

    def find(self, item: str) -> str:
        root = item
        while self._parent[root] != root:
            root = self._parent[root]
        while self._parent[item] != root:
            self._parent[item], item = root, self._parent[item]
        return root

    def union(self, a: str, b: str) -> bool:
        """Merge the sets containing a and b. Returns False if already connected."""
        root_a, root_b = self.find(a), self.find(b)
        if root_a == root_b:
            return False
        self._parent[root_a] = root_b
        return True


def _order_nodes_by_tree_traversal(node_ids: list[str], mst_edges: list[dict[str, Any]]) -> list[str]:
    """Order node ids so that MST-connected nodes appear sequentially.

    e.g. for A--B--C connected in the MST, returns [A, B, C] rather than
    whatever order the tables happened to be extracted in. Walks each
    connected component breadth-first from its alphabetically-smallest node,
    and orders components themselves by that same starting node, so the
    result is fully deterministic.
    """
    adjacency: dict[str, list[str]] = {node_id: [] for node_id in node_ids}
    for edge in mst_edges:
        adjacency[edge["from"]].append(edge["to"])
        adjacency[edge["to"]].append(edge["from"])
    for neighbors in adjacency.values():
        neighbors.sort()

    visited: set[str] = set()
    ordered: list[str] = []
    for start in sorted(node_ids):
        if start in visited:
            continue
        queue = deque([start])
        visited.add(start)
        while queue:
            current = queue.popleft()
            ordered.append(current)
            for neighbor in adjacency[current]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)

    return ordered


def compute_minimum_spanning_tree(schema: DatabaseSchema) -> dict[str, Any]:
    """Build the schema graph and reduce its edges to a minimum spanning forest.

    All edges carry uniform weight, so any spanning forest is minimum;
    Kruskal's algorithm is used with a stable edge order for determinism.
    """
    graph = build_schema_graph(schema)
    node_ids = [node["id"] for node in graph["nodes"]]
    node_id_set = set(node_ids)
    uf = _UnionFind(node_ids)

    sorted_edges = sorted(
        graph["edges"],
        key=lambda e: (e["from"], e["from_column"], e["to"], e["to_column"]),
    )

    mst_edges = []
    for edge in sorted_edges:
        if edge["from"] not in node_id_set or edge["to"] not in node_id_set:
            continue
        if uf.union(edge["from"], edge["to"]):
            mst_edges.append(edge)

    component_count = len({uf.find(node_id) for node_id in node_ids}) if node_ids else 0

    nodes_by_id = {node["id"]: node for node in graph["nodes"]}
    traversal_order = _order_nodes_by_tree_traversal(node_ids, mst_edges)
    ordered_nodes = [nodes_by_id[node_id] for node_id in traversal_order]

    return {
        "graph": {
            "database_type": graph["graph"]["database_type"],
            "database_name": graph["graph"]["database_name"],
            "generated_at": graph["graph"]["generated_at"],
            "node_count": len(node_ids),
            "total_edge_count": len(graph["edges"]),
            "mst_edge_count": len(mst_edges),
            "component_count": component_count,
        },
        "nodes": ordered_nodes,
        "mst_edges": mst_edges,
    }


def save_minimum_spanning_tree(schema: DatabaseSchema, file_path: Path | str) -> None:
    """Compute and write the minimum spanning tree/forest to a TOML file on disk."""
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    mst = compute_minimum_spanning_tree(schema)
    path.write_text(tomli_w.dumps(mst), encoding="utf-8")
