"""Table selector stage: choose the query's tables out of the BFS neighborhood.

The selector is handed the embedding matches *and* every table the foreign-key
walk reached from them, together with the relationships among that whole set.
Its job is to return the tables the question actually needs -- including the
bridge tables nobody searched for -- and the relationships that connect them,
which is what the planner is then given.

Tables come from the neighborhood, relationships come from the schema, and the
model only chooses: a name it invents is discarded, and the relationships
attached to its answer are the real foreign keys among the tables it kept, not
anything it wrote.
"""

import logging
from pathlib import Path

from backend.src.config import settings
from backend.src.utils.text_utils import extract_json_block
from backend.src.schemas.pipeline import (
    NeighborhoodTable,
    QuestionAnalysis,
    TableNeighborhood,
    TableSelectionResult,
)
from backend.src.data.models.schema import DatabaseSchema, Relationship
from backend.src.control.providers.model.base import ModelProvider
from backend.src.control.providers.model.koboldcpp import KoboldCppProvider

logger = logging.getLogger(__name__)


def _get_selector_prompt_path() -> Path:
    p = Path("backend/src/core/query_processing/prompts/table_selection.txt")
    if p.exists():
        return p
    alt = Path(__file__).resolve().parents[1] / "prompts" / "table_selection.txt"
    if alt.exists():
        return alt
    return p


DEFAULT_SELECTOR_PROMPT = _get_selector_prompt_path()

# The prompt competes with the completion for one context window, and the server
# truncates the PROMPT to make room -- cutting the instruction block off the
# front and leaving the model nothing but a schema dump to summarize. These
# bounds keep the rendered neighborhood small enough that the instructions survive.
MAX_FIELDS_SHOWN = 35
MAX_TABLE_DESCRIPTION_CHARS = 320
MAX_FIELD_DESCRIPTION_CHARS = 70


def _clip(text: str, limit: int) -> str:
    """Collapse whitespace and cut to a character budget."""
    clean = " ".join((text or "").split())
    return clean if len(clean) <= limit else clean[:limit].rstrip() + "..."


def _origin(table: NeighborhoodTable) -> str:
    """How this table got onto the list, in a few words.

    A table two hops out is a candidate for a structural reason, not a semantic
    one; saying so is what stops the model treating a far-off table as if the
    question had matched it directly.
    """
    if table.level == 0:
        score = f", similarity: {table.similarity:.4f}" if table.similarity is not None else ""
        rank = f", rank: {table.rank}" if table.rank is not None else ""
        return f"query match{score}{rank}"
    hops = "1 hop" if table.level == 1 else f"{table.level} hops"
    if table.reached_from:
        return f"{hops} from {', '.join(table.reached_from)}"
    return hops


def _render_tables(
    neighborhood: TableNeighborhood,
    schema: DatabaseSchema | None,
    include_columns: bool = True,
    include_field_descriptions: bool = True,
    limit: int | None = None,
) -> str:
    """Render the table block: name, origin, description, and optionally columns.

    Columns are the largest single contributor to this prompt and are the first
    thing sacrificed when it will not fit. The stage's own output is names,
    descriptions and relationships, so those three are what must survive.
    """
    lines: list[str] = []
    for table in neighborhood.tables[: limit if limit is not None else len(neighborhood.tables)]:
        line = f"• {table.name} ({_origin(table)})"
        if table.description:
            line += f" - {_clip(table.description, MAX_TABLE_DESCRIPTION_CHARS)}"
        if include_columns and schema:
            obj = schema.get_object(table.name)
            if obj:
                fields_repr = []
                for f in obj.fields[:MAX_FIELDS_SHOWN]:
                    f_desc = ""
                    if include_field_descriptions and f.description:
                        f_desc = f": {_clip(f.description, MAX_FIELD_DESCRIPTION_CHARS)}"
                    fields_repr.append(f"{f.name} ({f.type}){f_desc}")
                if len(obj.fields) > MAX_FIELDS_SHOWN:
                    fields_repr.append(f"... +{len(obj.fields) - MAX_FIELDS_SHOWN} more columns")
                if fields_repr:
                    line += f"\n  Columns: {', '.join(fields_repr)}"
        lines.append(line)
    return "\n".join(lines)


def _render_relationships(relationships: list[Relationship], names: set[str] | None = None) -> str:
    """Render the foreign keys, optionally restricted to a surviving table set."""
    lines: list[str] = []
    for r in relationships:
        if names is not None and (
            r.from_object.lower() not in names or r.to_object.lower() not in names
        ):
            continue
        lines.append(
            f"• {r.from_object}.{r.from_field} -> {r.to_object}.{r.to_field} ({r.relationship_type})"
        )
    return "\n".join(lines)


def filter_relationships(
    selected_objects: list[str],
    relationships: list[Relationship],
) -> list[Relationship]:
    """Foreign keys whose both endpoints are among the selected objects."""
    wanted = {name.lower() for name in selected_objects}
    return [
        r
        for r in relationships
        if r.from_object.lower() in wanted and r.to_object.lower() in wanted
    ]


class TableSelector:
    """Selects the question's tables and their relationships from the neighborhood."""

    def __init__(
        self,
        provider: ModelProvider | None = None,
        prompt_path: Path | str | None = None,
    ) -> None:
        self.provider = provider or KoboldCppProvider()
        self.prompt_path = Path(prompt_path or DEFAULT_SELECTOR_PROMPT)

    def _load_prompt_template(self) -> str:
        if self.prompt_path.exists():
            return self.prompt_path.read_text(encoding="utf-8")
        resolved = Path.cwd() / self.prompt_path
        if resolved.exists():
            return resolved.read_text(encoding="utf-8")
        raise FileNotFoundError(f"Table selection prompt file not found at {self.prompt_path}")

    def _prompt_char_budget(self) -> int:
        """Characters the whole prompt may occupy before the server truncates it.

        The completion budget is subtracted from the context window because the
        server reserves it first; what remains is converted with a deliberately
        pessimistic chars-per-token ratio, and a safety margin is held back so a
        tokenization estimate that runs slightly long still fits.
        """
        usable_tokens = settings.model_context_tokens - settings.table_selector_max_tokens
        if usable_tokens <= 0:
            return 0
        return int(usable_tokens * settings.prompt_chars_per_token * 0.90)

    def _fit_neighborhood(
        self,
        neighborhood: TableNeighborhood,
        schema: DatabaseSchema | None,
    ) -> tuple[str, str]:
        """Render tables and relationships, degrading detail until they fit.

        An overflowing prompt is not a degraded prompt, it is a broken one: the
        instructions sit at the front and are what gets cut, so the model is left
        with a schema dump and no task. Column descriptions go first, then
        columns altogether, and only then tables -- nearest-hop first, since the
        neighborhood is already ordered that way -- so the question's own
        embedding matches are the last thing to be surrendered.
        """
        budget = self._prompt_char_budget()
        # The template and semantic context ride along in the same prompt.
        overhead = len(self._load_prompt_template()) + 512
        rel_block = _render_relationships(neighborhood.relationships)

        def fits(tables_block: str, relationships_block: str) -> bool:
            return (
                budget <= 0
                or len(tables_block) + len(relationships_block) + overhead <= budget
            )

        full = _render_tables(neighborhood, schema)
        if fits(full, rel_block):
            return full, rel_block

        lean = _render_tables(neighborhood, schema, include_field_descriptions=False)
        if fits(lean, rel_block):
            logger.warning(
                "Table selector prompt would exceed the context budget; dropped "
                "column descriptions (%d -> %d chars).",
                len(full),
                len(lean),
            )
            return lean, rel_block

        bare = _render_tables(neighborhood, schema, include_columns=False)
        if fits(bare, rel_block):
            logger.warning(
                "Table selector prompt still too large; dropped column lists "
                "entirely (%d -> %d chars).",
                len(lean),
                len(bare),
            )
            return bare, rel_block

        # Still too large: shed the furthest tables, never fewer than one, so the
        # model always has something to choose from.
        kept = len(neighborhood.tables)
        while kept > 1:
            kept -= 1
            names = {t.name.lower() for t in neighborhood.tables[:kept]}
            trimmed = _render_tables(neighborhood, schema, include_columns=False, limit=kept)
            trimmed_rels = _render_relationships(neighborhood.relationships, names)
            if fits(trimmed, trimmed_rels):
                logger.warning(
                    "Table selector prompt still too large; showing only the "
                    "nearest %d of %d tables.",
                    kept,
                    len(neighborhood.tables),
                )
                return trimmed, trimmed_rels
        names = {neighborhood.tables[0].name.lower()} if neighborhood.tables else set()
        return (
            _render_tables(neighborhood, schema, include_columns=False, limit=1),
            _render_relationships(neighborhood.relationships, names),
        )

    async def select(
        self,
        question_analysis: QuestionAnalysis,
        neighborhood: TableNeighborhood,
        schema: DatabaseSchema | None = None,
        feedback: str | None = None,
    ) -> TableSelectionResult:
        """Select the tables and relationships the planner should be given."""
        if not neighborhood or not neighborhood.tables:
            return TableSelectionResult(
                selected_objects=[],
                selected_relationships=[],
                sufficient=False,
                missing_objects=[],
                reason="No candidate tables provided.",
                retrieval_hint=None,
            )

        # The neighborhood defines what may be selected; the schema only supplies
        # canonical casing when the two disagree.
        allowed = {t.name.lower(): t.name for t in neighborhood.tables}
        schema_map = {obj.name.lower(): obj.name for obj in schema.objects} if schema else {}

        tables_context, relationships_context = self._fit_neighborhood(neighborhood, schema)
        if not relationships_context:
            relationships_context = (
                "None -- no foreign keys link these tables. Do not invent one."
            )

        feedback_context = ""
        if feedback:
            feedback_context = f"\nATTENTION - PREVIOUS SELECTION FEEDBACK (RETRY):\n{feedback}\n"

        template = self._load_prompt_template()
        prompt = template.format(
            question=question_analysis.question,
            subjective=", ".join(question_analysis.subjective) or "None",
            objective=", ".join(question_analysis.objective) or "None",
            linguistic=", ".join(question_analysis.nouns) or "None",
            tables_context=tables_context,
            relationships_context=relationships_context,
            feedback_context=feedback_context,
        )

        try:
            raw_response = await self.provider.generate(
                prompt=prompt,
                temperature=0.0,
                max_tokens=settings.table_selector_max_tokens,
            )
            data = extract_json_block(raw_response)
            parsed = TableSelectionResult.model_validate(data)

            # Strict validation: the model may only select tables it was shown.
            validated: list[str] = []
            for item in parsed.selected_objects:
                item_lower = item.strip().lower()
                if item_lower in allowed:
                    canonical_name = schema_map.get(item_lower, allowed[item_lower])
                    if canonical_name not in validated:
                        validated.append(canonical_name)
                else:
                    logger.warning("Discarded hallucinated/out-of-neighborhood table: %r", item)

            return TableSelectionResult(
                selected_objects=validated,
                selected_relationships=filter_relationships(
                    validated, neighborhood.relationships
                ),
                sufficient=parsed.sufficient,
                missing_objects=parsed.missing_objects,
                reason=parsed.reason,
                retrieval_hint=parsed.retrieval_hint,
            )

        except Exception as err:
            logger.warning("Table selection SLM call failed: %s. Using top candidate fallback.", err)
            fallback = [neighborhood.tables[0].name] if neighborhood.tables else []
            return TableSelectionResult(
                selected_objects=fallback,
                selected_relationships=filter_relationships(
                    fallback, neighborhood.relationships
                ),
                sufficient=True,
                missing_objects=[],
                reason=f"Fallback selection due to error: {err}",
                retrieval_hint=None,
            )
