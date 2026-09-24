"""Database-independent semantic QueryPlanner stage."""

import logging
from pathlib import Path
from query_processing.core.config import settings
from query_processing.core.text_utils import extract_json_block
from query_processing.models.pipeline import (
    QueryPlan,
    QuestionAnalysis,
    RelevantSchema,
)
from query_processing.models.schema import DatabaseType
from query_processing.providers.model.base import ModelProvider
from query_processing.providers.model.koboldcpp import KoboldCppProvider

logger = logging.getLogger(__name__)

DEFAULT_PLANNER_PROMPT = Path("query_processing/prompts/planner.txt")


def _format_relevant_schema_for_planner(schema: RelevantSchema) -> str:
    lines: list[str] = []
    for obj in schema.objects:
        lines.append(f"• {obj.name} ({obj.kind.value}): {obj.description}")
        for f in obj.fields:
            desc = f" - {f.description}" if f.description else ""
            samples = f" (samples: {f.sample_values})" if f.sample_values else ""
            lines.append(f"    - {f.name} ({f.type}){desc}{samples}")
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


class QueryPlanner:
    """Produces a database-independent QueryPlan representing semantic intents."""

    def __init__(
        self,
        provider: ModelProvider | None = None,
        prompt_path: Path | str | None = None,
    ) -> None:
        self.provider = provider or KoboldCppProvider()
        self.prompt_path = Path(prompt_path or DEFAULT_PLANNER_PROMPT)

    def _load_prompt_template(self) -> str:
        if self.prompt_path.exists():
            return self.prompt_path.read_text(encoding="utf-8")
        resolved = Path.cwd() / self.prompt_path
        if resolved.exists():
            return resolved.read_text(encoding="utf-8")
        raise FileNotFoundError(f"Planner prompt file not found at {self.prompt_path}")

    async def plan(
        self,
        question_analysis: QuestionAnalysis,
        relevant_schema: RelevantSchema,
        database_type: DatabaseType,
    ) -> QueryPlan:
        """Generate a validated, syntax-free QueryPlan."""
        schema_context = _format_relevant_schema_for_planner(relevant_schema)
        template = self._load_prompt_template()

        prompt = template.format(
            question=question_analysis.question,
            database_type=database_type.value,
            subjective=", ".join(question_analysis.subjective) or "None",
            objective=", ".join(question_analysis.objective) or "None",
            schema_context=schema_context,
        )

        try:
            raw_response = await self.provider.generate(
                prompt=prompt,
                temperature=0.1,
                max_tokens=settings.model_max_tokens,
            )
            data = extract_json_block(raw_response)
            plan = QueryPlan.model_validate(data)

            # Validate that sources belong to relevant schema
            allowed_sources = relevant_schema.get_object_names()
            valid_sources = [s for s in plan.sources if s.lower() in allowed_sources]
            if not valid_sources and relevant_schema.objects:
                valid_sources = [relevant_schema.objects[0].name]
            plan.sources = valid_sources

            return plan

        except Exception as err:
            logger.warning("QueryPlanner SLM call failed: %s. Using basic fallback plan.", err)
            default_sources = [obj.name for obj in relevant_schema.objects[:1]]
            return QueryPlan(
                operation="select",
                sources=default_sources,
                projections=["*"],
                filters=[],
                aggregations=[],
                group_by=[],
                order_by=[],
                limit=100,
            )
