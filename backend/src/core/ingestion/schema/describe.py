"""Generates SLM descriptions for each table and every one of its columns.

Reads a nodes/edges TOML file -- either the MST or the plain graph -- and walks
its nodes in the order that file stores them, batching them a few tables at a
time so the model sees the relationships between tables in the same batch and
describes them with that context, rather than describing each in isolation.
Batches are processed one at a time, not in parallel.

Reading from the MST means tables arrive in tree-traversal order with only the
cycle-free edges as context; reading from the plain graph means extraction order
with every edge. MongoDB has no MST, so it always uses the graph.

Each batch yields, per table, one prose description plus one short description
per column. Only the table description is ever embedded -- `table_descriptions`
narrows to it before the embedding stage runs, so column text never enters the
retrieval vectors.
"""

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from backend.src.control.providers.model.base import ModelProvider
from backend.src.control.providers.model.koboldcpp import KoboldCppProvider
from backend.src.data.models.schema import DatabaseSchema
from backend.src.data.repositories import graph_repository, paths
from backend.src.schemas.ingestion import TableDescription
from backend.src.utils.text_utils import extract_json_block

logger = logging.getLogger(__name__)

DEFAULT_BATCH_SIZE = 5
# A batch asks for one verbose description per table (naming every column) *and*
# one short description per column, and a reasoning model spends part of its
# budget on <think> before emitting any JSON. The global MODEL_MAX_TOKENS default
# (1500) truncates that mid-JSON, so this stage asks for its own, larger budget.
DESCRIBE_MAX_TOKENS = 8000
# How many times to ask for one batch before giving up. A reasoning model's
# <think> block has no fixed length: for an identical prompt, completions
# measured 2175 and 2870 tokens on consecutive runs. When that reasoning
# exhausts the budget, generation is cut before any JSON is emitted and parsing
# fails -- roughly one batch in three. Re-asking resamples a shorter reasoning
# pass, so a retry, not a bigger budget or a terser prompt, is what makes this
# stage reliable.
DESCRIBE_MAX_ATTEMPTS = 3


def _format_tables(nodes: list[dict[str, Any]]) -> str:
    """Render one batch's tables and their columns for the prompt."""
    return "\n".join(
        f"- {node['id']}: "
        + ", ".join(f"{col['name']} ({col['type']})" for col in node.get("columns", []))
        for node in nodes
    )


def _format_relationships(batch_ids: set[str], edges: list[dict[str, Any]]) -> str:
    """Render the relationships that fall entirely within one batch."""
    within_batch = [
        edge for edge in edges if edge["from"] in batch_ids and edge["to"] in batch_ids
    ]
    if not within_batch:
        return "(none within this batch)"
    return "\n".join(
        f"- {edge['from']}.{edge['from_column']} -> {edge['to']}.{edge['to_column']} ({edge['type']})"
        for edge in within_batch
    )


def _chunk(items: list[Any], size: int) -> list[list[Any]]:
    """Split a list into consecutive chunks of at most `size` items."""
    return [items[i : i + size] for i in range(0, len(items), size)]


async def _describe_batch(
    provider: ModelProvider,
    prompt: str,
    batch_ids: set[str],
) -> dict[str, Any]:
    """Ask the model to describe one batch, retrying until the response parses.

    Returns the raw per-table entries keyed by table name; an empty mapping when
    the model answered with well-formed JSON that simply had no descriptions.
    """
    for attempt in range(1, DESCRIBE_MAX_ATTEMPTS + 1):
        raw_response = await provider.generate(prompt, max_tokens=DESCRIBE_MAX_TOKENS)
        try:
            parsed = extract_json_block(raw_response)
        except ValueError as err:
            if attempt == DESCRIBE_MAX_ATTEMPTS:
                raise ValueError(
                    f"Could not get parseable JSON for tables {sorted(batch_ids)} "
                    f"after {DESCRIBE_MAX_ATTEMPTS} attempts: {err}"
                ) from err
            logger.warning(
                "Describe attempt %d/%d for tables %s returned no parseable JSON (%s). Retrying.",
                attempt,
                DESCRIBE_MAX_ATTEMPTS,
                sorted(batch_ids),
                err,
            )
            continue
        return parsed.get("descriptions", {}) if isinstance(parsed, dict) else {}

    return {}


async def describe_tables(
    source_path: Path | str,
    provider: ModelProvider | None = None,
    prompt_path: Path | str | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> dict[str, TableDescription]:
    """Generate descriptions for the tables in a graph or MST TOML file.

    Returns table_name -> TableDescription, each carrying the table's prose
    description and a description per column.
    """
    nodes, edges = graph_repository.load_nodes_and_edges(source_path)

    model_provider = provider or KoboldCppProvider()
    template = Path(prompt_path or paths.describe_prompt_path()).read_text(encoding="utf-8")

    descriptions: dict[str, TableDescription] = {}
    for batch in _chunk(nodes, batch_size):
        batch_ids = {node["id"] for node in batch}
        prompt = template.format(
            tables_context=_format_tables(batch),
            relationships_context=_format_relationships(batch_ids, edges),
        )
        entries = await _describe_batch(model_provider, prompt, batch_ids)
        for node in batch:
            descriptions[node["id"]] = TableDescription.from_raw(entries.get(node["id"]))

    return descriptions


def table_descriptions(descriptions: dict[str, TableDescription]) -> dict[str, str]:
    """Narrow to the table_name -> table description mapping that gets embedded."""
    return {name: table.description for name, table in descriptions.items()}


def apply_descriptions(
    schema: DatabaseSchema,
    descriptions: dict[str, TableDescription],
) -> int:
    """Write generated table and column descriptions into a DatabaseSchema in place.

    Columns are matched by name against the object's fields, so dotted paths
    resolve into nested fields and a MongoDB `address.city` lands correctly. A
    column the model named but the schema does not have is skipped rather than
    created, so the model can never introduce a field into the canonical schema.

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
