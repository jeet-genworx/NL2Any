"""Generates a one-sentence SLM description for each table.

Reads a nodes/edges TOML file -- either the MST (mst.py) or the plain graph
(graph.py) -- and walks its nodes in whatever order that file stores them in,
batching them into groups of a few tables at a time so the SLM can see the
relationships between tables in the same batch and describe them with that
context in mind, rather than describing each table in total isolation.
Batches are processed one at a time (not in parallel). Results are stored as
a flat table_name -> description mapping in a JSON file.

Reading from the MST (use_mst=True) means tables arrive in tree-traversal
order with only the cycle-free MST edges as relationship context. Reading
from the plain graph (use_mst=False) means tables arrive in extraction order
with every edge as context -- there's no tree reduction, so batches are
grouped by the source file's own order rather than by connectivity. MongoDB
has no MST at all (no foreign-key concept), so it always uses the graph.
"""

import json
import tomllib
from pathlib import Path
from typing import Any

from query_processing.core.text_utils import extract_json_block
from query_processing.models.schema import DatabaseType
from query_processing.providers.model.base import ModelProvider
from query_processing.providers.model.koboldcpp import KoboldCppProvider

DEFAULT_DESCRIBE_PROMPT = Path("query_processing/prompts/table_description.txt")
DEFAULT_BATCH_SIZE = 5


def get_default_descriptions_path(database_type: DatabaseType) -> Path:
    """Return canonical descriptions JSON path for the database type."""
    filename = (
        "postgres_descriptions.json"
        if database_type == DatabaseType.POSTGRESQL
        else "mongo_descriptions.json"
    )
    return Path("ingestion/schemas") / filename


def _format_tables(nodes: list[dict[str, Any]]) -> str:
    blocks = []
    for node in nodes:
        columns = ", ".join(f"{col['name']} ({col['type']})" for col in node.get("columns", []))
        blocks.append(f"- {node['id']}: {columns}")
    return "\n".join(blocks)


def _format_relationships(batch_ids: set[str], edges: list[dict[str, Any]]) -> str:
    within_batch = [edge for edge in edges if edge["from"] in batch_ids and edge["to"] in batch_ids]
    if not within_batch:
        return "(none within this batch)"
    return "\n".join(
        f"- {edge['from']}.{edge['from_column']} -> {edge['to']}.{edge['to_column']} ({edge['type']})"
        for edge in within_batch
    )


def _chunk(items: list[Any], size: int) -> list[list[Any]]:
    return [items[i : i + size] for i in range(0, len(items), size)]


def _load_nodes_and_edges(path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    nodes = data.get("nodes", [])
    # MST files use "mst_edges"; plain graph files use "edges".
    edges = data.get("mst_edges", data.get("edges", []))
    return nodes, edges


async def describe_tables(
    source_path: Path | str,
    provider: ModelProvider | None = None,
    prompt_path: Path | str = DEFAULT_DESCRIBE_PROMPT,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> dict[str, str]:
    """Read a nodes/edges TOML file (MST or plain graph) and generate SLM
    descriptions for its tables.

    Tables are processed in batches of `batch_size`, in the exact node order
    the source file already stores them in. Each batch is described together,
    with the relationships between tables in that batch given as context, so
    related tables are described with an understanding of how they connect --
    then the next batch is processed the same way, one batch at a time.
    """
    path = Path(source_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Schema graph/MST file not found at: {path}. Run 'uv run init-schema' first."
        )

    nodes, edges = _load_nodes_and_edges(path)

    model_provider = provider or KoboldCppProvider()
    template = Path(prompt_path).read_text(encoding="utf-8")

    descriptions: dict[str, str] = {}
    for batch in _chunk(nodes, batch_size):
        batch_ids = {node["id"] for node in batch}
        prompt = template.format(
            tables_context=_format_tables(batch),
            relationships_context=_format_relationships(batch_ids, edges),
        )

        raw_response = await model_provider.generate(prompt)
        parsed = extract_json_block(raw_response)
        batch_descriptions = parsed.get("descriptions", {}) if isinstance(parsed, dict) else {}

        for node in batch:
            descriptions[node["id"]] = str(batch_descriptions.get(node["id"], "")).strip()

    return descriptions


def save_descriptions(descriptions: dict[str, str], file_path: Path | str) -> None:
    """Write the table_name -> description mapping to a JSON file on disk."""
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(descriptions, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
