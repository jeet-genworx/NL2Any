"""Computes a minimum spanning tree over the schema graph.

Edges (foreign-key relationships) are treated as uniform weight, so the MST is
simply a minimal, cycle-free set of edges that keeps every FK-connected table
reachable. The schema graph is not guaranteed to be connected -- tables with no
foreign keys at all are isolated nodes -- so this produces a spanning FOREST,
one tree per component, via Kruskal's algorithm with union-find.

PostgreSQL only: MongoDB's graph has no edges to reduce.

Pure: computing the structure is here, writing it to disk is the graph
repository's job.
"""

from collections import deque
from typing import Any

from backend.src.core.ingestion.schema.graph import build_schema_graph
from backend.src.data.models.schema import DatabaseSchema


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


def _order_nodes_by_tree_traversal(
    node_ids: list[str],
    mst_edges: list[dict[str, Any]],
) -> list[str]:
    """Order node ids so that MST-connected nodes appear sequentially.

    For A--B--C connected in the MST this returns [A, B, C] rather than whatever
    order the tables happened to be extracted in, which is what lets the
    description stage batch related tables together. Walks each component
    breadth-first from its alphabetically smallest node and orders components by
    that same node, so the result is fully deterministic.
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

    All edges carry uniform weight, so any spanning forest is minimum; Kruskal's
    algorithm is used over a stable edge order for determinism.
    """
    graph = build_schema_graph(schema)
    node_ids = [node["id"] for node in graph["nodes"]]
    node_id_set = set(node_ids)
    union_find = _UnionFind(node_ids)

    sorted_edges = sorted(
        graph["edges"],
        key=lambda edge: (edge["from"], edge["from_column"], edge["to"], edge["to_column"]),
    )

    mst_edges = [
        edge
        for edge in sorted_edges
        if edge["from"] in node_id_set
        and edge["to"] in node_id_set
        and union_find.union(edge["from"], edge["to"])
    ]

    nodes_by_id = {node["id"]: node for node in graph["nodes"]}
    traversal_order = _order_nodes_by_tree_traversal(node_ids, mst_edges)

    return {
        "graph": {
            **{
                key: graph["graph"][key]
                for key in ("database_type", "database_name", "generated_at")
            },
            "node_count": len(node_ids),
            "total_edge_count": len(graph["edges"]),
            "mst_edge_count": len(mst_edges),
            "component_count": len({union_find.find(node_id) for node_id in node_ids}),
        },
        "nodes": [nodes_by_id[node_id] for node_id in traversal_order],
        "mst_edges": mst_edges,
    }
