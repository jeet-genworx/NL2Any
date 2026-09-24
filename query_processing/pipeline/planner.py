"""Database-independent semantic QueryPlanner stage."""

import logging
from pathlib import Path
import re

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

SQL_KEYWORD_PATTERN = re.compile(
    r"\b(SELECT|WHERE|JOIN|FROM|ORDER\s+BY|GROUP\s+BY|HAVING|LIMIT|UNION|DROP|DELETE|UPDATE|INSERT)\b",
    re.IGNORECASE,
)


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


def _clean_sql_syntax(text: str) -> str:
    """Strip extraneous SQL syntax keywords that SLM might mistakenly include."""
    cleaned = SQL_KEYWORD_PATTERN.sub("", text).strip()
    return cleaned if cleaned else text


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
        feedback: str | None = None,
    ) -> QueryPlan:
        """Generate a validated, syntax-free QueryPlan."""
        schema_context = _format_relevant_schema_for_planner(relevant_schema)
        template = self._load_prompt_template()

        feedback_section = ""
        if feedback:
            feedback_section = (
                f"\nATTENTION - PREVIOUS PLAN FAILED VALIDATION (RETRY):\n{feedback}\n"
                "Please fix the above plan issues."
            )

        prompt = template.format(
            question=question_analysis.question,
            database_type=database_type.value,
            subjective=", ".join(question_analysis.subjective) or "None",
            objective=", ".join(question_analysis.objective) or "None",
            schema_context=schema_context,
            feedback_section=feedback_section,
        )

        try:
            raw_response = await self.provider.generate(
                prompt=prompt,
                temperature=0.1,
                max_tokens=settings.model_max_tokens,
            )
            data = extract_json_block(raw_response)
            plan = QueryPlan.model_validate(data)

            return self._validate_and_sanitize_plan(plan, relevant_schema)

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

    def _validate_and_sanitize_plan(
        self,
        plan: QueryPlan,
        relevant_schema: RelevantSchema,
    ) -> QueryPlan:
        """Validate and sanitize plan against schema to guarantee no invalid tables or raw SQL."""
        allowed_sources = relevant_schema.get_object_names()
        all_allowed_fields: set[str] = set()
        for obj in relevant_schema.objects:
            all_allowed_fields.update(f.lower() for f in obj.all_field_paths())
            for f in obj.fields:
                all_allowed_fields.add(f.name.lower())
                all_allowed_fields.add(f"{obj.name.lower()}.{f.name.lower()}")

        # 1. Validate sources
        valid_sources = [s for s in plan.sources if s.lower() in allowed_sources]
        if not valid_sources and relevant_schema.objects:
            valid_sources = [relevant_schema.objects[0].name]
        plan.sources = valid_sources

        # 2. Sanitize and validate filters (strip SQL keywords like WHERE)
        sanitized_filters = []
        for flt in plan.filters:
            clean_field = _clean_sql_syntax(flt.field)
            clean_op = _clean_sql_syntax(flt.operator).lower() or "equals"
            # Verify field exists in schema
            f_norm = clean_field.lower()
            leaf = f_norm.split(".")[-1]
            if f_norm in all_allowed_fields or leaf in all_allowed_fields:
                flt.field = clean_field
                flt.operator = clean_op
                sanitized_filters.append(flt)
            else:
                logger.warning("Dropped filter on unknown field: %r", flt.field)
        plan.filters = sanitized_filters

        # 3. Sanitize group_by and order_by
        clean_group = []
        for g in plan.group_by:
            cg = _clean_sql_syntax(g)
            g_norm = cg.lower()
            leaf = g_norm.split(".")[-1]
            if g_norm in all_allowed_fields or leaf in all_allowed_fields:
                clean_group.append(cg)
        plan.group_by = clean_group

        clean_order = []
        for o in plan.order_by:
            co = _clean_sql_syntax(o.field)
            o_norm = co.lower()
            leaf = o_norm.split(".")[-1]
            if o_norm in all_allowed_fields or leaf in all_allowed_fields:
                o.field = co
                clean_order.append(o)
        plan.order_by = clean_order

        return plan
