"""Deterministic foreign-key graph over a schema, and the walk out from a seed set.

Embedding matching answers "which tables does this question look like". It does
not answer "which tables does a query over them have to touch": asking for
revenue per customer matches `customers` and maybe `orders`, while the revenue
lives on `order_items`, which the question never names and whose description
never mentions customers. A table nobody retrieved cannot be selected, and a
selection missing a bridge table leaves the generator to invent a join.

So the candidate space is not the embedding matches -- it is their foreign-key
neighborhood: every table reachable from a match within a few hops, discovered
breadth-first and collected into one unique set. Choosing from that set is the
table selector's job, and materializing the chosen tables into a RelevantSchema
remains the expander's.

Pure: no model calls, no I/O, same answer for the same schema every time.
"""

import logging
from collections import deque

from backend.src.schemas.pipeline import NeighborhoodTable, TableNeighborhood
from backend.src.data.models.schema import DatabaseSchema, Relationship

logger = logging.getLogger(__name__)


def _edge_sort_key(item: tuple[str, Relationship]) -> tuple[str, str, str, str, str]:
    """Total order over adjacency entries, so traversal is reproducible."""
    neighbor, rel = item
    return (neighbor, rel.from_object, rel.from_field, rel.to_object, rel.to_field)


class JoinGraph:
    """Undirected multigraph of schema objects linked by foreign-key relationships.

    Edges are undirected because a join reads the same from either side, and
    parallel edges are kept distinct rather than collapsed: two foreign keys
    from one table to the same target -- `flights.origin_airport_id` and
    `flights.destination_airport_id` both referencing `airports.id` -- are two
    different joins that mean different things, and deciding which the question
    wants is precisely what the table selector is for.

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

    def contains(self, name: str) -> bool:
        """Whether the schema has an object by this name."""
        return name.lower() in self._adjacency

    def component_of(self, name: str) -> int:
        """Connected-component id of an object, or 0 when it is not in the schema."""
        return self._components.get(name.lower(), 0)

    def has_edges(self) -> bool:
        """Whether the schema has any foreign-key relationship to walk at all."""
        return any(self._adjacency.values())

    def neighbors(self, name: str) -> list[str]:
        """Objects one foreign key away, ordered and de-duplicated.

        Parallel foreign keys to the same table yield that table once: the walk
        is collecting tables, and which of several keys to join on is decided
        later, against the full relationship list.
        """
        key = name.lower()
        if key not in self._adjacency:
            return []
        ordered = [
            neighbor
            for neighbor, _ in sorted(self._adjacency[key], key=_edge_sort_key)
            if neighbor != key
        ]
        return list(dict.fromkeys(ordered))

    def bfs_levels(self, source: str, max_levels: int) -> dict[str, int]:
        """Breadth-first walk from one object, mapping each table to its hop count.

        The source is included at level 0. A table reached by several routes is
        recorded at the shortest one, which is what breadth-first already gives.
        """
        start = source.lower()
        if start not in self._adjacency or max_levels < 0:
            return {}

        levels = {start: 0}
        queue: deque[str] = deque([start])
        while queue:
            current = queue.popleft()
            depth = levels[current]
            if depth >= max_levels:
                continue
            for neighbor in self.neighbors(current):
                if neighbor not in levels:
                    levels[neighbor] = depth + 1
                    queue.append(neighbor)
        return levels


def relationships_among(names: list[str], schema: DatabaseSchema) -> list[Relationship]:
    """Foreign keys internal to an object set, without building a graph."""
    wanted = {name.lower() for name in names}
    return [
        rel
        for rel in schema.relationships
        if rel.from_object.lower() in wanted and rel.to_object.lower() in wanted
    ]


def build_table_neighborhood(
    seed_tables: list[tuple[str, float | None, int | None]],
    schema: DatabaseSchema,
    max_levels: int,
    max_tables: int,
    graph: JoinGraph | None = None,
) -> TableNeighborhood:
    """Walk out from every embedding match and collect one unique table set.

    Each seed is a `(name, similarity, rank)` triple from embedding retrieval.
    Every seed is walked separately -- each with its own visited set, so one
    seed's traversal never blocks another's -- and the results are unioned, a
    table being recorded at the shortest hop count any seed reached it by and
    carrying the seeds that reached it.

    `max_tables` bounds the union, because three hops through a densely
    normalized schema can reach most of the database and the selector's prompt
    has to fit a context window. Seeds are never discarded; past them, nearer
    tables survive, so what is dropped is always the weakest-related end of the
    walk. Relationships are then taken among whatever survived, so the
    relationship block can never reference a table the selector was not shown.
    """
    join_graph = graph if graph is not None else JoinGraph(schema)

    seeds: list[tuple[str, float | None, int | None]] = []
    seen_seeds: set[str] = set()
    for name, similarity, rank in seed_tables:
        if not name or not name.strip():
            continue
        key = name.lower()
        if key in seen_seeds:
            continue
        if not join_graph.contains(name):
            logger.warning("Embedding match %r is not in the schema; skipped as a BFS seed.", name)
            continue
        seen_seeds.add(key)
        seeds.append((join_graph.canonical(name), similarity, rank))

    # level: shortest hop count to any seed; reached_from: the seeds that got there.
    level: dict[str, int] = {}
    reached_from: dict[str, list[str]] = {}
    for seed_name, _, _ in seeds:
        for found, depth in join_graph.bfs_levels(seed_name, max_levels).items():
            if found not in level or depth < level[found]:
                level[found] = depth
            if seed_name not in reached_from.setdefault(found, []):
                reached_from[found].append(seed_name)

    seed_keys = {name.lower() for name, _, _ in seeds}
    seed_meta = {name.lower(): (similarity, rank) for name, similarity, rank in seeds}

    # Seeds first in retrieval order, then discovered tables nearest-hop first.
    discovered = sorted(
        (key for key in level if key not in seed_keys),
        key=lambda key: (level[key], key),
    )
    ordered = [name.lower() for name, _, _ in seeds] + discovered

    truncated = False
    discarded: list[str] = []
    if max_tables > 0 and len(ordered) > max_tables:
        # Seeds are kept even when they alone exceed the cap: dropping a table
        # the question matched directly is worse than an oversized prompt, and
        # the selector degrades its own rendering when the prompt will not fit.
        keep = max(max_tables, len(seeds))
        discarded = [join_graph.canonical(key) for key in ordered[keep:]]
        ordered = ordered[:keep]
        truncated = bool(discarded)

    tables: list[NeighborhoodTable] = []
    for key in ordered:
        obj = schema.get_object(join_graph.canonical(key))
        similarity, rank = seed_meta.get(key, (None, None))
        tables.append(
            NeighborhoodTable(
                name=join_graph.canonical(key),
                description=(obj.description or "") if obj else "",
                kind=(obj.kind.value if obj else "table"),
                level=level.get(key, 0),
                similarity=similarity,
                rank=rank,
                reached_from=reached_from.get(key, []),
            )
        )

    names = [t.name for t in tables]
    return TableNeighborhood(
        seed_objects=[name for name, _, _ in seeds],
        tables=tables,
        relationships=relationships_among(names, schema),
        max_levels=max_levels,
        max_tables=max_tables,
        truncated=truncated,
        discarded_objects=discarded,
    )


def _selection_components(
    selected: list[str],
    schema: DatabaseSchema,
) -> list[list[str]]:
    """Group the selected objects by what their own foreign keys connect.

    One group means the selection is joinable as it stands. More than one means
    the query would have to cross a gap the schema does not bridge.
    """
    keys = [name.lower() for name in dict.fromkeys(selected)]
    members = set(keys)
    adjacency: dict[str, set[str]] = {key: set() for key in keys}
    for rel in schema.relationships:
        source, target = rel.from_object.lower(), rel.to_object.lower()
        if source in members and target in members and source != target:
            adjacency[source].add(target)
            adjacency[target].add(source)

    groups: list[list[str]] = []
    seen: set[str] = set()
    # Walked in selection order, so the first group holds the earliest-selected
    # object and the grouping is reproducible.
    for start in keys:
        if start in seen:
            continue
        group: list[str] = []
        queue = deque([start])
        seen.add(start)
        while queue:
            current = queue.popleft()
            group.append(current)
            for neighbor in sorted(adjacency[current]):
                if neighbor not in seen:
                    seen.add(neighbor)
                    queue.append(neighbor)
        groups.append(group)
    return groups


def _shortest_path_between(
    graph: JoinGraph,
    sources: set[str],
    targets: set[str],
    allowed: set[str],
) -> list[str]:
    """Shortest walk from any source to any target, through allowed tables only.

    Restricted to `allowed` -- the tables the selector was actually shown --
    because a bridge pulled from outside the neighborhood would be a table
    nobody, model or human, ever had the chance to reject.
    """
    queue: deque[str] = deque(sorted(sources))
    previous: dict[str, str | None] = {source: None for source in sorted(sources)}
    while queue:
        current = queue.popleft()
        for neighbor in graph.neighbors(current):
            if neighbor in previous or neighbor not in allowed:
                continue
            previous[neighbor] = current
            if neighbor in targets:
                path = [neighbor]
                while previous[path[-1]] is not None:
                    path.append(previous[path[-1]])
                return list(reversed(path))
            queue.append(neighbor)
    return []


def connect_selection(
    selected_objects: list[str],
    schema: DatabaseSchema,
    graph: JoinGraph | None = None,
    allowed: list[str] | None = None,
) -> tuple[list[str], list[str], list[str]]:
    """Repair a selection that cannot be joined, and report what could not be.

    The table selector is asked to include the intermediate tables its choice
    needs, and it does not reliably do so: on a FinOps question spanning
    discrepancies and customers, the model named the bridging table in its own
    reasoning and still left it out of the selection, four runs out of four.
    The result reaches the planner with the two endpoints and no edge between
    them, which is the exact situation where a generator invents a join.

    Whether two tables are joinable is a fact about the schema, so it is settled
    here rather than asked of a model: the shortest foreign-key path between
    disconnected parts of the selection is walked -- through the neighborhood
    only -- and the tables along it are added. What genuinely cannot be
    connected is returned separately, so the caller can drop it rather than
    hand the generator a table with no way in.

    Returns (resolved objects, tables added as bridges, objects left unjoinable).
    """
    join_graph = graph if graph is not None else JoinGraph(schema)
    # De-duplicated after canonicalizing, not before: "CUSTOMERS" and
    # "customers" are one table, and two spellings of it would otherwise both
    # survive into the component grouping.
    selected = list(
        dict.fromkeys(
            join_graph.canonical(name)
            for name in selected_objects
            if name and name.strip()
        )
    )
    if len(selected) < 2:
        return selected, [], []

    # Default to the whole schema only when no neighborhood was supplied.
    allowed_keys = (
        {name.lower() for name in allowed}
        if allowed is not None
        else {obj.name.lower() for obj in schema.objects}
    )
    allowed_keys.update(name.lower() for name in selected)

    resolved = list(selected)
    added: list[str] = []
    while True:
        groups = _selection_components(resolved, schema)
        if len(groups) <= 1:
            return resolved, added, []

        # Shortest bridge between any two groups, so the smallest repair wins;
        # ties break on the path's own names to stay reproducible.
        best: list[str] = []
        for index, left in enumerate(groups):
            for right in groups[index + 1 :]:
                path = _shortest_path_between(
                    join_graph, set(left), set(right), allowed_keys
                )
                if not path:
                    continue
                if not best or (len(path), path) < (len(best), best):
                    best = path
        if not best:
            # Nothing reaches anything else: keep the group holding the
            # earliest-selected object and report the remainder.
            core = max(groups, key=len)
            keep = {name.lower() for name in core}
            unjoinable = [name for name in resolved if name.lower() not in keep]
            return (
                [name for name in resolved if name.lower() in keep],
                added,
                unjoinable,
            )

        present = {name.lower() for name in resolved}
        for key in best:
            if key not in present:
                canonical = join_graph.canonical(key)
                resolved.append(canonical)
                added.append(canonical)
                present.add(key)
