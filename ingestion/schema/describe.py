"""Generates SLM descriptions for each table and for every one of its columns.

Reads a nodes/edges TOML file -- either the MST (mst.py) or the plain graph
(graph.py) -- and walks its nodes in whatever order that file stores them in,
batching them into groups of a few tables at a time so the SLM can see the
relationships between tables in the same batch and describe them with that
context in mind, rather than describing each table in total isolation.
Batches are processed one at a time (not in parallel).

Each batch yields, per table, one prose table description plus one short
description per column. Both are persisted in two places:

  * The descriptions JSON file is the full record of what was generated:
    {"customers": {"description": "...", "columns": {"id": "..."}}}.
  * The canonical schema TOML gets the same text folded into it by
    `apply_descriptions`, so the schema carries its own documentation.

Only the table description is ever *embedded*. The embedding stage reads the
descriptions JSON but narrows it through `table_descriptions()` first, so
column text never enters the retrieval vectors.

Reading from the MST (use_mst=True) means tables arrive in tree-traversal
order with only the cycle-free MST edges as relationship context. Reading
from the plain graph (use_mst=False) means tables arrive in extraction order
with every edge as context -- there's no tree reduction, so batches are
grouped by the source file's own order rather than by connectivity. MongoDB
has no MST at all (no foreign-key concept), so it always uses the graph.
"""

import json
import tomllib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field as PydanticField

from query_processing.core.text_utils import extract_json_block
from query_processing.models.schema import DatabaseSchema, DatabaseType
from query_processing.providers.model.base import ModelProvider
from query_processing.providers.model.koboldcpp import KoboldCppProvider

DEFAULT_DESCRIBE_PROMPT = Path("query_processing/prompts/table_description.txt")
DEFAULT_BATCH_SIZE = 5
# A batch asks for one verbose description per table (naming every column) *and*
# one short description per column, and a reasoning model spends part of its
# budget on <think> before emitting any JSON. The global MODEL_MAX_TOKENS default
# (1500) truncates that mid-JSON, so this stage asks for its own, larger budget.
# Sized for the per-column output: ~5 tables x ~10 columns of extra JSON on top
# of the table prose.
DESCRIBE_MAX_TOKENS = 8000


class TableDescription(BaseModel):
    """One table's generated documentation: prose for the table, a line per column.

    Only `description` is embedded; `columns` is written to the schema TOML.
    """

    description: str = ""
    columns: dict[str, str] = PydanticField(default_factory=dict)


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


def parse_table_entry(entry: Any) -> TableDescription:
    """Normalize one table's entry from the model's JSON into a TableDescription.

    The prompt asks for {"description": ..., "columns": {...}}, but a bare
    description string is also accepted: smaller models sometimes collapse the
    object, and that shouldn't cost us the table description too.
    """
    if isinstance(entry, str):
        return TableDescription(description=entry.strip())
    if not isinstance(entry, dict):
        return TableDescription()

    raw_columns = entry.get("columns")
    columns: dict[str, str] = {}
    if isinstance(raw_columns, dict):
        for column_name, column_text in raw_columns.items():
            text = str(column_text).strip()
            if text:
                columns[str(column_name)] = text

    return TableDescription(
        description=str(entry.get("description", "")).strip(),
        columns=columns,
    )


def table_descriptions(descriptions: dict[str, TableDescription]) -> dict[str, str]:
    """Flatten to the table_name -> table description mapping.

    This is the only part of the generated documentation that gets embedded;
    column descriptions are intentionally excluded from the retrieval vectors.
    """
    return {name: table.description for name, table in descriptions.items()}


def apply_descriptions(
    schema: DatabaseSchema,
    descriptions: dict[str, TableDescription],
) -> int:
    """Write generated table and column descriptions into a DatabaseSchema in place.

    Matches columns by name against the object's fields -- dotted paths resolve
    into nested fields, so a MongoDB `address.city` would land correctly. A
    column the model named but the schema doesn't have is skipped rather than
    created, so the SLM can never introduce a field into the canonical schema.

    Returns the number of column descriptions actually applied.
    """
    applied_columns = 0

    for obj in schema.objects:
        table = descriptions.get(obj.name)
        if table is None:
            continue

        obj.description = table.description
        for column_name, column_text in table.columns.items():
            field = obj.get_field(column_name)
            if field is None:
                continue
            field.description = column_text
            applied_columns += 1

    schema.description_generated_at = datetime.now(timezone.utc).isoformat()
    return applied_columns


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
) -> dict[str, TableDescription]:
    """Read a nodes/edges TOML file (MST or plain graph) and generate SLM
    descriptions for its tables and their columns.

    Tables are processed in batches of `batch_size`, in the exact node order
    the source file already stores them in. Each batch is described together,
    with the relationships between tables in that batch given as context, so
    related tables are described with an understanding of how they connect --
    then the next batch is processed the same way, one batch at a time.

    Returns table_name -> TableDescription, each carrying the table's prose
    description and a description per column.
    """
    path = Path(source_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Schema graph/MST file not found at: {path}. Run 'uv run init-schema' first."
        )

    nodes, edges = _load_nodes_and_edges(path)

    model_provider = provider or KoboldCppProvider()
    template = Path(prompt_path).read_text(encoding="utf-8")

    descriptions: dict[str, TableDescription] = {}
    for batch in _chunk(nodes, batch_size):
        batch_ids = {node["id"] for node in batch}
        prompt = template.format(
            tables_context=_format_tables(batch),
            relationships_context=_format_relationships(batch_ids, edges),
        )

        raw_response = await model_provider.generate(prompt, max_tokens=DESCRIBE_MAX_TOKENS)
        parsed = extract_json_block(raw_response)
        batch_descriptions = parsed.get("descriptions", {}) if isinstance(parsed, dict) else {}

        for node in batch:
            descriptions[node["id"]] = parse_table_entry(batch_descriptions.get(node["id"]))

    return descriptions


def save_descriptions(
    descriptions: dict[str, TableDescription],
    file_path: Path | str,
) -> None:
    """Write the full generated documentation to a JSON file on disk.

    This is the complete record of what the SLM produced -- each table's prose
    description and every column description:

        {"customers": {"description": "...", "columns": {"id": "..."}}}

    The embedding stage reads this file but embeds only the `description` of
    each table (see `table_descriptions`); the columns are here as the record,
    and in the schema TOML as documentation.
    """
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        name: table.model_dump() for name, table in descriptions.items()
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
