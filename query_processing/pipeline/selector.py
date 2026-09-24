"""Table and collection selector stage."""

import logging
from pathlib import Path
from query_processing.core.text_utils import extract_json_block
from query_processing.models.pipeline import QuestionAnalysis, TableSelectionResult
from query_processing.models.schema import DatabaseSchema
from query_processing.providers.model.base import ModelProvider
from query_processing.providers.model.koboldcpp import KoboldCppProvider
from query_processing.retrieval.bm25 import RetrievalResult

logger = logging.getLogger(__name__)

DEFAULT_SELECTOR_PROMPT = Path("query_processing/prompts/table_selection.txt")


class TableSelector:
    """Selects relevant tables/collections from BM25 candidates using the SLM."""

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
        question_analysis: QuestionAnalysis,
        candidates: list[RetrievalResult],
        schema: DatabaseSchema,
    ) -> TableSelectionResult:
        """Select relevant objects from BM25 candidates with strict validation against TOML schema."""
        # Refinement 4: Graceful handling for zero BM25 candidates
        if not candidates:
            return TableSelectionResult(selected_objects=[])

        candidate_names = [c.object_name for c in candidates]
        valid_schema_names = {obj.name.lower(): obj.name for obj in schema.objects}

        # Format candidates description context
        lines: list[str] = []
        for c in candidates:
            obj = c.schema_object
            field_paths = obj.all_field_paths()[:8]  # brief sample of fields
            desc = f" - {obj.description}" if obj.description else ""
            lines.append(f"• {obj.name} ({obj.kind.value}){desc}\n  Fields: {', '.join(field_paths)}")
        candidates_context = "\n".join(lines)

        template = self._load_prompt_template()
        prompt = template.format(
            question=question_analysis.question,
            subjective=", ".join(question_analysis.subjective) or "None",
            objective=", ".join(question_analysis.objective) or "None",
            linguistic=", ".join(question_analysis.nouns) or "None",
            candidates_context=candidates_context,
        )

        try:
            raw_response = await self.provider.generate(
                prompt=prompt,
                temperature=0.0,
                max_tokens=800,
            )
            data = extract_json_block(raw_response)
            parsed = TableSelectionResult.model_validate(data)

            # Strict validation: must exist in candidate_names and in valid_schema_names
            validated: list[str] = []
            candidate_set = {name.lower() for name in candidate_names}

            for item in parsed.selected_objects:
                item_lower = item.strip().lower()
                if item_lower in candidate_set and item_lower in valid_schema_names:
                    canonical_name = valid_schema_names[item_lower]
                    if canonical_name not in validated:
                        validated.append(canonical_name)
                else:
                    logger.warning(
                        "Discarded hallucinated/invalid object from selector: %r", item
                    )

            # If model returned nothing valid but candidates exist and score was high, fallback
            if not validated and candidates and candidates[0].score > 2.0:
                validated = [candidates[0].object_name]

            return TableSelectionResult(selected_objects=validated)

        except Exception as err:
            logger.warning("Table selection SLM call failed: %s. Using top candidate fallback.", err)
            # Safe fallback: take top BM25 candidate if available
            fallback = [candidates[0].object_name] if candidates else []
            return TableSelectionResult(selected_objects=fallback)
