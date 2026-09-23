"""PostgreSQL query generator translating QueryPlan into SQL."""

import logging
from pathlib import Path
import re
from nl2anyquery.core.text_utils import strip_think_tags
from nl2anyquery.models.pipeline import GeneratedQuery, QueryPlan, RelevantSchema
from nl2anyquery.models.schema import DatabaseType
from nl2anyquery.pipeline.generators.base import QueryGenerator
from nl2anyquery.pipeline.planner import _format_relevant_schema_for_planner
from nl2anyquery.providers.model.base import ModelProvider
from nl2anyquery.providers.model.koboldcpp import KoboldCppProvider

logger = logging.getLogger(__name__)

DEFAULT_POSTGRES_PROMPT = Path("prompts/postgres_query_generator.txt")


def _extract_sql(text: str) -> str:
    cleaned = strip_think_tags(text).strip()

    # 1. Match ```sql ... ```
    fence_match = re.search(r"```(?:sql)?\s*(.*?)\s*```", cleaned, re.DOTALL | re.IGNORECASE)
    if fence_match:
        sql = fence_match.group(1).strip()
    else:
        # 2. Extract starting from SELECT or WITH
        select_match = re.search(r"\b(SELECT|WITH)\b.*", cleaned, re.DOTALL | re.IGNORECASE)
        if select_match:
            sql = select_match.group(0).strip()
        else:
            sql = cleaned

    # Strip trailing markdown / extra notes after semicolon
    if ";" in sql:
        sql = sql[: sql.index(";") + 1].strip()
    elif not sql.endswith(";"):
        sql = sql + ";"

    return sql


class PostgresQueryGenerator(QueryGenerator):
    """Generates read-only PostgreSQL queries."""

    def __init__(
        self,
        provider: ModelProvider | None = None,
        prompt_path: Path | str | None = None,
    ) -> None:
        self.provider = provider or KoboldCppProvider()
        self.prompt_path = Path(prompt_path or DEFAULT_POSTGRES_PROMPT)

    def _load_prompt_template(self) -> str:
        if self.prompt_path.exists():
            return self.prompt_path.read_text(encoding="utf-8")
        resolved = Path.cwd() / self.prompt_path
        if resolved.exists():
            return resolved.read_text(encoding="utf-8")
        raise FileNotFoundError(f"Postgres generator prompt not found at {self.prompt_path}")

    async def generate_query(
        self,
        question: str,
        plan: QueryPlan,
        schema: RelevantSchema,
        feedback: str | None = None,
    ) -> GeneratedQuery:
        """Generate PostgreSQL query from QueryPlan and RelevantSchema."""
        template = self._load_prompt_template()
        schema_context = _format_relevant_schema_for_planner(schema)
        plan_context = plan.model_dump_json(indent=2)

        feedback_section = ""
        if feedback:
            feedback_section = (
                f"ATTENTION - PREVIOUS ATTEMPT FAILED WITH FEEDBACK:\n{feedback}\n"
                "Please fix the above issue in the generated query."
            )

        prompt = template.format(
            question=question,
            plan_context=plan_context,
            schema_context=schema_context,
            feedback_section=feedback_section,
        )

        raw_response = await self.provider.generate(
            prompt=prompt,
            temperature=0.1,
            max_tokens=600,
        )

        sql = _extract_sql(raw_response)
        return GeneratedQuery(
            database_type=DatabaseType.POSTGRESQL,
            raw_query=sql,
            formatted_query=sql,
        )
