"""Deterministic join-path graph over a schema's foreign keys.

Table selection picks the objects a question is *about*, which is not the same
as the objects a query has to *touch*. Asking for revenue per customer when
revenue lives on `order_items` and customers live on `customers` selects two
tables with no column in common: the generator, handed a schema slice with no
edge between them, invents a join condition. This module enumerates the real
paths through intervening tables so that guesswork is never required.

Pure: building the graph and enumerating paths is here, choosing between the
candidates is the path selector's job, and materializing the chosen tables into
a RelevantSchema remains the expander's.
"""

from collections import deque

from backend.src.schemas.pipeline import JoinPathCandidate
from backend.src.data.models.schema import DatabaseSchema, Relationship


def _edge_sort_key(item: tuple[str, Relationship]) -> tuple[str, str, str, str, str]:
    """Total order over adjacency entries, so enumeration is reproducible."""
    neighbor, rel = item
    return (neighbor, rel.from_object, rel.from_field, rel.to_object, rel.to_field)


class JoinGraph:
    """Undirected multigraph of schema objects linked by foreign-key relationships.

    Edges are undirected because a join reads the same from either side, and
    parallel edges are kept distinct rather than collapsed: two foreign keys
    from one table to the same target -- `flights.origin_airport_id` and
    `flights.destination_airport_id` both referencing `airports.id` -- are two
    different joins that mean different things, and deciding which the question
    wants is precisely what the path selector is for.

    The full relationship set is used, never the ingestion MST: that tree drops
    edges to break cycles, and a dropped edge is a join path that silently
    ceases to exist.
    """

    def __init__(self, schema: DatabaseSchema) -> None:
        self._canonical: dict[str, str] = {obj.name.lower(): obj.name for obj in schema.objects}
        self._adjacency: dict[str, list[tuple[str, Relationship]]] = {
            obj.name.lower(): [] for obj in schema.objects
        }

        for rel in schema.relationships:
            source = rel.from_object.lower()
            target = rel.to_object.lower()
            # A relationship naming an object outside the schema cannot be walked.
            if source not in self._adjacency or target not in self._adjacency:
                continue
            self._adjacency[source].append((target, rel))
            # A self-referencing foreign key (employees.manager_id -> employees.id)
            # is recorded once; it never helps reach a *different* table.
            if source != target:
                self._adjacency[target].append((source, rel))

        self._components = self._label_components()

    def _label_components(self) -> dict[str, int]:
        """Assign each object a connected-component id, walking from sorted starts."""
        labels: dict[str, int] = {}
        next_id = 0
        for start in sorted(self._adjacency):
            if start in labels:
                continue
            next_id += 1
            labels[start] = next_id
            queue = deque([start])
            while queue:
                current = queue.popleft()
                for neighbor, _ in self._adjacency[current]:
                    if neighbor not in labels:
                        labels[neighbor] = next_id
                        queue.append(neighbor)
        return labels

    def canonical(self, name: str) -> str:
        """Schema casing for an object name, or the name unchanged if unknown."""
        return self._canonical.get(name.lower(), name)

    def component_of(self, name: str) -> int:
        """Connected-component id of an object, or 0 when it is not in the schema."""
        return self._components.get(name.lower(), 0)

    def has_edges(self) -> bool:
        """Whether the schema has any foreign-key relationship to walk at all."""
        return any(self._adjacency.values())

    def find_paths(
        self,
        source: str,
        target: str,
        max_hops: int,
        max_paths: int,
    ) -> list[tuple[list[str], list[Relationship]]]:
        """Enumerate simple paths from source to target, shortest first.

        Breadth-first over partial paths, so truncating at `max_paths` keeps the
        tightest joins rather than an arbitrary slice of them. Revisiting a table
        is forbidden, which both keeps paths simple and bounds the walk.

        Returns (ordered tables, ordered edges) pairs; `len(edges)` is the hops.
        """
        src = source.lower()
        dst = target.lower()
        if src == dst or src not in self._adjacency or dst not in self._adjacency:
            return []
        if self._components.get(src) != self._components.get(dst):
            return []
        if max_hops < 1 or max_paths < 1:
            return []

        results: list[tuple[list[str], list[Relationship]]] = []
        queue: deque[tuple[list[str], list[Relationship]]] = deque([([src], [])])

        while queue and len(results) < max_paths:
            tables, edges = queue.popleft()
            if len(edges) >= max_hops:
                continue
            for neighbor, rel in sorted(self._adjacency[tables[-1]], key=_edge_sort_key):
                if neighbor == dst:
                    results.append((tables + [dst], edges + [rel]))
                    if len(results) >= max_paths:
                        break
                elif neighbor not in tables:
                    queue.append((tables + [neighbor], edges + [rel]))

        return results


def build_join_path_candidates(
    selected_object_names: list[str],
    schema: DatabaseSchema,
    max_hops: int,
    max_paths_per_pair: int,
    max_candidates: int,
    graph: JoinGraph | None = None,
) -> list[JoinPathCandidate]:
    """Enumerate every way the selected objects could be connected.

    Produces a connected candidate per simple path between each pair of selected
    objects, plus a standalone candidate for any selected object left with no
    path to the others -- whether because it sits in a different component of the
    schema or because the nearest path exceeds the hop limit. Standalone objects
    are candidates rather than errors: a question may legitimately be about one,
    and the selector is better placed than a heuristic to tell that from a table
    that was retrieved by mistake.
    """
    join_graph = graph if graph is not None else JoinGraph(schema)
    selected = list(dict.fromkeys(name for name in selected_object_names if name and name.strip()))

    raw: list[dict] = []
    for index, left in enumerate(selected):
        for right in selected[index + 1 :]:
            for tables, edges in join_graph.find_paths(
                left, right, max_hops=max_hops, max_paths=max_paths_per_pair
            ):
                raw.append(
                    {
                        "endpoints": [join_graph.canonical(left), join_graph.canonical(right)],
                        "tables": [join_graph.canonical(table) for table in tables],
                        "edges": edges,
                        "hops": len(edges),
                        "connected": True,
                        "component_id": join_graph.component_of(left),
                        "note": "",
                    }
                )

    # Bound the prompt on densely linked schemas, keeping the shortest paths.
    if len(raw) > max_candidates:
        raw.sort(key=lambda item: (item["hops"], item["tables"]))
        raw = raw[:max_candidates]

    # Whatever survived truncation defines which objects still have a path, so
    # standalone candidates are derived afterwards and never contradict the list.
    connected_objects = {name.lower() for item in raw for name in item["endpoints"]}
    for name in selected:
        if name.lower() in connected_objects:
            continue
        component = join_graph.component_of(name)
        shares_component = any(
            other.lower() != name.lower() and join_graph.component_of(other) == component
            for other in selected
        )
        note = (
            f"no foreign-key path to another selected object within {max_hops} hops"
            if shares_component
            else "no foreign-key path to any other selected object -- separate part of the schema"
        )
        raw.append(
            {
                "endpoints": [join_graph.canonical(name)],
                "tables": [join_graph.canonical(name)],
                "edges": [],
                "hops": 0,
                "connected": False,
                "component_id": component,
                "note": note,
            }
        )

    # Ids are assigned last so they stay contiguous after truncation.
    return [
        JoinPathCandidate(path_id=f"p{position}", **item)
        for position, item in enumerate(raw, start=1)
    ]
