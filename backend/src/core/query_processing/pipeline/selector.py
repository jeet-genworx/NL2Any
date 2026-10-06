"""V2 Table selector stage: selects tables from complete database schema description."""

import logging
from pathlib import Path
from typing import Any

from backend.src.config import settings
from backend.src.utils.text_utils import extract_json_block
from backend.src.schemas.pipeline import (
    CandidateTable,
    QuestionAnalysis,
    RelevantSchema,
    TableSelectionResult,
)
from backend.src.data.models.schema import DatabaseSchema
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


def format_schema_description(schema: DatabaseSchema) -> str:
    """Format complete database schema with descriptions, columns, types, and relationships."""
    lines: list[str] = [
        f"Database: {schema.database_name} ({schema.database_type.value})",
        f"Total Tables/Collections: {len(schema.objects)}\n",
    ]

    for obj in schema.objects:
        desc = f" - {obj.description}" if obj.description else ""
        lines.append(f"• {obj.name} ({obj.kind.value}){desc}")
        fields_repr = []
        for f in obj.fields:
            f_desc = f": {f.description}" if f.description else ""
            fields_repr.append(f"    - {f.name} ({f.type}){f_desc}")
            for nested_f in f.nested:
                n_desc = f": {nested_f.description}" if nested_f.description else ""
                fields_repr.append(f"      - {f.name}.{nested_f.name} ({nested_f.type}){n_desc}")
        if fields_repr:
            lines.append("  Columns:")
            lines.extend(fields_repr)

    if schema.relationships:
        lines.append("\nKnown Relationships:")
        for r in schema.relationships:
            lines.append(
                f"• {r.from_object}.{r.from_field} -> {r.to_object}.{r.to_field} ({r.relationship_type})"
            )

    return "\n".join(lines)


def format_relevant_schema(schema: RelevantSchema | DatabaseSchema) -> str:
    """Format selected relevant schema for consumption by Query Generator and Validator."""
    lines: list[str] = []
    for obj in schema.objects:
        desc = f": {obj.description}" if obj.description else ""
        lines.append(f"• {obj.name} ({obj.kind.value}){desc}")
        for f in obj.fields:
            f_desc = f" - {f.description}" if f.description else ""
            samples = f" (samples: {f.sample_values})" if f.sample_values else ""
            lines.append(f"    - {f.name} ({f.type}){f_desc}{samples}")
            for nested_f in f.nested:
                n_desc = f" - {nested_f.description}" if nested_f.description else ""
                lines.append(f"      - {f.name}.{nested_f.name} ({nested_f.type}){n_desc}")

    if schema.relationships:
        lines.append("\nKnown Relationships:")
        for r in schema.relationships:
            lines.append(
                f"• {r.from_object}.{r.from_field} -> {r.to_object}.{r.to_field} ({r.relationship_type})"
            )
    return "\n".join(lines)


# Backwards compatibility alias
_format_relevant_schema_for_planner = format_relevant_schema


def package_selected_schema(schema: DatabaseSchema, selected_tables: list[str]) -> RelevantSchema:
    """Ground and package selected tables with their full schema and relationships.

    Deterministic grounding only: verifies selected table names against the canonical schema,
    attaches their exact fields and descriptions, and includes relationships connecting them.
    Does NOT discover or add additional tables.
    """
    if not selected_tables or not schema:
        return RelevantSchema(objects=[], relationships=[])

    selected_set = {t.strip().lower() for t in selected_tables if t.strip()}
    matched_objects = []
    for obj in schema.objects:
        obj_lower = obj.name.lower()
        short_name = obj_lower.split(".")[-1]
        if obj_lower in selected_set or short_name in selected_set:
            matched_objects.append(obj)

    matched_names = {obj.name.lower() for obj in matched_objects}
    matched_short_names = {obj.name.lower().split(".")[-1] for obj in matched_objects}

    matched_relationships = []
    for rel in schema.relationships:
        from_low = rel.from_object.lower()
        to_low = rel.to_object.lower()
        from_short = from_low.split(".")[-1]
        to_short = to_low.split(".")[-1]

        from_matched = from_low in matched_names or from_short in matched_short_names
        to_matched = to_low in matched_names or to_short in matched_short_names

        if from_matched and to_matched:
            matched_relationships.append(rel)

    return RelevantSchema(
        objects=matched_objects,
        relationships=matched_relationships,
    )


class TableSelector:
    """Selects relevant tables from complete database schema description."""

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

    async def select(
        self,
        question: str | QuestionAnalysis,
        schema: DatabaseSchema,
        candidates: list[CandidateTable] | None = None,
        feedback: str | None = None,
    ) -> TableSelectionResult:
        """Select relevant tables from complete database description using SLM."""
        q_text = question.question if isinstance(question, QuestionAnalysis) else str(question)

        if not schema or not schema.objects:
            return TableSelectionResult(
                selected_objects=[],
                sufficient=False,
                missing_objects=[],
                reason="No schema objects available in database.",
                retrieval_hint=None,
            )

        schema_map = {obj.name.lower(): obj.name for obj in schema.objects}
        for obj in schema.objects:
            short = obj.name.lower().split(".")[-1]
            if short not in schema_map:
                schema_map[short] = obj.name

        schema_context = format_schema_description(schema)

        feedback_context = ""
        if feedback:
            feedback_context = (
                f"\nATTENTION - PREVIOUS ATTEMPT FEEDBACK (RETRY):\n{feedback}\n"
                "Please reconsider table selection to address this feedback."
            )

        template = self._load_prompt_template()
        prompt = template.format(
            question=q_text,
            schema_context=schema_context,
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

            # Grounding: model may only select tables that actually exist in the schema
            validated: list[str] = []
            for item in parsed.selected_objects:
                item_lower = item.strip().lower()
                if item_lower in schema_map:
                    canonical_name = schema_map[item_lower]
                    if canonical_name not in validated:
                        validated.append(canonical_name)
                else:
                    logger.warning("Discarded non-existent/hallucinated table: %r", item)

            return TableSelectionResult(
                selected_objects=validated,
                sufficient=parsed.sufficient,
                missing_objects=parsed.missing_objects,
                reason=parsed.reason,
                retrieval_hint=parsed.retrieval_hint,
            )

        except Exception as err:
            logger.warning("Table selection SLM call failed: %s.", err)
            return TableSelectionResult(
                selected_objects=[],
                sufficient=False,
                missing_objects=[],
                reason=f"Table selection failed due to error: {err}",
                retrieval_hint=None,
            )
