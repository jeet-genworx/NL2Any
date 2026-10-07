"""Table selector stage with sufficiency validation and hallucination filtering."""

import logging
from pathlib import Path

from backend.src.config import settings
from backend.src.utils.text_utils import extract_json_block
from backend.src.schemas.pipeline import (
    CandidateTable,
    QuestionAnalysis,
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

# The prompt competes with the completion for one context window, and the server
# truncates the PROMPT to make room -- cutting the instruction block off the
# front and leaving the model nothing but a schema dump to summarize. These
# bounds keep the rendered candidates small enough that the instructions survive.
MAX_FIELDS_SHOWN = 35
MAX_TABLE_DESCRIPTION_CHARS = 320
MAX_FIELD_DESCRIPTION_CHARS = 70


def _clip(text: str, limit: int) -> str:
    """Collapse whitespace and cut to a character budget."""
    clean = " ".join((text or "").split())
    return clean if len(clean) <= limit else clean[:limit].rstrip() + "..."


def _render_candidates(
    candidates: list[CandidateTable],
    schema: DatabaseSchema | None,
    include_field_descriptions: bool = True,
) -> str:
    """Render the candidate block, optionally dropping per-column descriptions.

    Column descriptions are the largest single contributor to this prompt and are
    the first thing sacrificed when it will not fit: a column's name and type are
    enough to judge whether a table is relevant, which is all this stage decides.
    """
    lines: list[str] = []
    for c in candidates:
        line = f"• {c.table_name} (similarity: {c.similarity:.4f}, rank: {c.rank})"
        if schema:
            obj = schema.get_object(c.table_name)
            if obj:
                if obj.description:
                    line += f" - {_clip(obj.description, MAX_TABLE_DESCRIPTION_CHARS)}"
                fields_repr = []
                for f in obj.fields[:MAX_FIELDS_SHOWN]:
                    f_desc = ""
                    if include_field_descriptions and f.description:
                        f_desc = f": {_clip(f.description, MAX_FIELD_DESCRIPTION_CHARS)}"
                    fields_repr.append(f"{f.name} ({f.type}){f_desc}")
                if len(obj.fields) > MAX_FIELDS_SHOWN:
                    fields_repr.append(f"... +{len(obj.fields) - MAX_FIELDS_SHOWN} more columns")
                line += f"\n  Columns: {', '.join(fields_repr)}"
        lines.append(line)
    return "\n".join(lines)


class TableSelector:
    """Selects relevant tables from candidates and determines sufficiency."""

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

    def _fit_candidates(
        self,
        candidates: list[CandidateTable],
        schema: DatabaseSchema | None,
    ) -> str:
        """Render candidates, dropping column descriptions if they will not fit.

        An overflowing prompt is not a degraded prompt, it is a broken one: the
        instructions sit at the front and are what gets cut, so the model is left
        with a schema dump and no task. Shedding column descriptions keeps the
        table names, types and the instructions, which is what this stage needs.
        """
        budget = self._prompt_char_budget()
        full = _render_candidates(candidates, schema, include_field_descriptions=True)
        # The template and semantic context ride along in the same prompt.
        overhead = len(self._load_prompt_template()) + 512
        if budget <= 0 or len(full) + overhead <= budget:
            return full

        lean = _render_candidates(candidates, schema, include_field_descriptions=False)
        logger.warning(
            "Table selector prompt would exceed the context budget "
            "(%d chars + %d overhead > %d); dropped column descriptions, now %d chars.",
            len(full),
            overhead,
            budget,
            len(lean),
        )
        if len(lean) + overhead <= budget:
            return lean

        # Still too large: keep the highest-ranked candidates that fit, never
        # fewer than one, so the model always has something to choose from.
        kept = list(candidates)
        while len(kept) > 1:
            kept.pop()
            trimmed = _render_candidates(kept, schema, include_field_descriptions=False)
            if len(trimmed) + overhead <= budget:
                logger.warning(
                    "Table selector prompt still too large; showing only the top "
                    "%d of %d candidates.",
                    len(kept),
                    len(candidates),
                )
                return trimmed
        return _render_candidates(kept, schema, include_field_descriptions=False)

    async def select(
        self,
        question_analysis: QuestionAnalysis,
        candidates: list[CandidateTable],
        schema: DatabaseSchema | None = None,
        feedback: str | None = None,
    ) -> TableSelectionResult:
        """Select relevant tables from candidates, validating sufficiency and eliminating hallucinations."""
        if not candidates:
            return TableSelectionResult(
                selected_objects=[],
                sufficient=False,
                missing_objects=[],
                reason="No candidate tables provided.",
                retrieval_hint=None,
            )

        candidate_map = {c.table_name.lower(): c.table_name for c in candidates}
        schema_map = {obj.name.lower(): obj.name for obj in schema.objects} if schema else {}

        # Format candidates context with similarity scores, column types/descriptions, and known links
        candidate_names = {c.table_name.lower(): c.table_name for c in candidates}

        # Include relationships connecting any candidate tables
        rel_lines: list[str] = []
        if schema and schema.relationships:
            for r in schema.relationships:
                if r.from_object.lower() in candidate_names and r.to_object.lower() in candidate_names:
                    rel_lines.append(
                        f"• {r.from_object}.{r.from_field} -> {r.to_object}.{r.to_field} ({r.relationship_type})"
                    )

        candidates_context = self._fit_candidates(candidates, schema)
        if rel_lines:
            candidates_context += "\n\nKnown Relationships Between Candidates:\n" + "\n".join(rel_lines)

        feedback_context = ""
        if feedback:
            feedback_context = f"\nATTENTION - PREVIOUS SELECTION FEEDBACK (RETRY):\n{feedback}\n"

        template = self._load_prompt_template()
        prompt = template.format(
            question=question_analysis.question,
            subjective=", ".join(question_analysis.subjective) or "None",
            objective=", ".join(question_analysis.objective) or "None",
            linguistic=", ".join(question_analysis.nouns) or "None",
            candidates_context=candidates_context,
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

            # Strict validation: model may only select tables from provided candidates
            validated: list[str] = []
            for item in parsed.selected_objects:
                item_lower = item.strip().lower()
                if item_lower in candidate_map:
                    canonical_name = candidate_map[item_lower]
                    if schema_map and item_lower in schema_map:
                        canonical_name = schema_map[item_lower]
                    if canonical_name not in validated:
                        validated.append(canonical_name)
                else:
                    logger.warning("Discarded hallucinated/non-candidate table: %r", item)

            return TableSelectionResult(
                selected_objects=validated,
                sufficient=parsed.sufficient,
                missing_objects=parsed.missing_objects,
                reason=parsed.reason,
                retrieval_hint=parsed.retrieval_hint,
            )

        except Exception as err:
            logger.warning("Table selection SLM call failed: %s. Using top candidate fallback.", err)
            fallback = [candidates[0].table_name] if candidates else []
            return TableSelectionResult(
                selected_objects=fallback,
                sufficient=True,
                missing_objects=[],
                reason=f"Fallback selection due to error: {err}",
                retrieval_hint=None,
            )
