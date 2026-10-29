"""PostgreSQL query generator translating user question and Selected Schema directly into SQL."""

import logging
import re
from pathlib import Path
from typing import Any

import sqlglot
from sqlglot import exp

from backend.src.config import settings
from backend.src.control.providers.model.base import ModelProvider
from backend.src.control.providers.model.koboldcpp import KoboldCppProvider
from backend.src.core.query_processing.pipeline.generators.base import QueryGenerator
from backend.src.core.query_processing.pipeline.selector import format_relevant_schema
from backend.src.data.models.schema import DatabaseType
from backend.src.schemas.pipeline import (
    GeneratedQuery,
    GeneratorMissingInfo,
    RelevantSchema,
)
from backend.src.utils.text_utils import (
    clean_unreadable_characters,
    estimate_token_count,
    extract_json_block,
    strip_ansi_escapes,
    strip_think_tags,
)

logger = logging.getLogger(__name__)


def _get_postgres_prompt_path() -> Path:
    p = Path("backend/src/core/query_processing/prompts/postgres_query_generator.txt")
    if p.exists():
        return p
    alt = Path(__file__).resolve().parents[2] / "prompts" / "postgres_query_generator.txt"
    if alt.exists():
        return alt
    return p


DEFAULT_POSTGRES_PROMPT = _get_postgres_prompt_path()


def _extract_sql(text: str) -> str:
    cleaned = clean_unreadable_characters(strip_ansi_escapes(strip_think_tags(text))).strip()

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

    if not sql or sql.strip() == "":
        return ""

    # Strip trailing markdown / extra notes after semicolon
    if ";" in sql:
        sql = sql[: sql.index(";") + 1].strip()
    elif not sql.endswith(";"):
        sql = sql + ";"

    return clean_unreadable_characters(strip_ansi_escapes(sql))


class PostgresQueryGenerator(QueryGenerator):
    """Generates read-only PostgreSQL queries directly from question and selected schema."""

    def __init__(
        self,
        provider: ModelProvider | None = None,
        prompt_path: Path | str | None = None,
    ) -> None:
        self.provider = provider or KoboldCppProvider()
        self.prompt_path = Path(prompt_path or DEFAULT_POSTGRES_PROMPT)
        self.last_prompt_tokens: int = 0

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
        schema: Any = None,
        feedback: str | None = None,
        plan: Any = None,
        **kwargs: Any,
    ) -> GeneratedQuery | GeneratorMissingInfo:
        """Generate PostgreSQL query or return missing-information feedback."""
        self.last_prompt_tokens = 0
        # Detect whether arguments were passed as (question, schema) or (question, plan, schema)
        if isinstance(schema, RelevantSchema):
            active_schema = schema
        elif isinstance(plan, RelevantSchema):
            active_schema = plan
        elif isinstance(kwargs.get("schema"), RelevantSchema):
            active_schema = kwargs["schema"]
        else:
            active_schema = schema if isinstance(schema, RelevantSchema) else RelevantSchema()

        active_feedback = feedback if isinstance(feedback, str) else kwargs.get("feedback")

        template = self._load_prompt_template()
        schema_context = format_relevant_schema(active_schema)

        feedback_section = ""
        if active_feedback:
            cleaned_feedback = clean_unreadable_characters(strip_ansi_escapes(active_feedback)).strip()
            if cleaned_feedback.startswith("["):
                feedback_section = cleaned_feedback
            else:
                feedback_section = (
                    f"ATTENTION - PREVIOUS ATTEMPT FAILED WITH FEEDBACK:\n{cleaned_feedback}\n"
                    "Please fix the above issue in the generated query."
                )

        prompt = template.format(
            question=question,
            schema_context=schema_context,
            feedback_section=feedback_section,
        )
        self.last_prompt_tokens = estimate_token_count(prompt)

        raw_response = await self.provider.generate(
            prompt=prompt,
            temperature=0.1,
            max_tokens=settings.model_max_tokens,
        )

        # Check if model returned a structured missing information response
        try:
            data = extract_json_block(raw_response)
            if isinstance(data, dict) and data.get("status") == "missing_information":
                missing = data.get("missing_tables") or data.get("missing_objects") or []
                if isinstance(missing, str):
                    missing = [missing]
                missing_cols = data.get("missing_columns") or []
                if isinstance(missing_cols, str):
                    missing_cols = [missing_cols]
                return GeneratorMissingInfo(
                    status="missing_information",
                    missing_tables=missing,
                    missing_columns=missing_cols,
                    reason=str(data.get("reason", "Required schema information is missing.")),
                )
        except (ValueError, TypeError, KeyError) as json_err:
            logger.debug("Response is not missing_information JSON: %s", json_err)

        sql = _extract_sql(raw_response)

        # Check if extracted SQL references tables outside the Selected Schema
        if sql and active_schema and active_schema.objects:
            allowed_tables = set()
            for obj in active_schema.objects:
                nl = obj.name.lower()
                allowed_tables.add(nl)
                if "." in nl:
                    allowed_tables.add(nl.split(".", 1)[-1])

            try:
                parsed = sqlglot.parse_one(sql, read="postgres")
                cte_names = {
                    cte.alias_or_name.lower()
                    for cte in parsed.find_all(exp.CTE)
                    if cte.alias_or_name
                }
                subquery_aliases = {
                    s.alias.lower()
                    for s in parsed.find_all(exp.Subquery)
                    if s.alias
                }

                unselected_tables: list[str] = []
                for table_exp in parsed.find_all(exp.Table):
                    t_name = table_exp.name.lower()
                    full_name = f"{table_exp.db.lower()}.{t_name}" if table_exp.db else t_name
                    if not t_name or t_name in cte_names or t_name in subquery_aliases:
                        continue
                    if full_name not in allowed_tables and t_name not in allowed_tables:
                        candidate = full_name if table_exp.db else t_name
                        if candidate not in unselected_tables:
                            unselected_tables.append(candidate)

                if unselected_tables:
                    logger.info(
                        "Generated SQL referenced unselected tables %s not in schema %s; emitting GeneratorMissingInfo",
                        unselected_tables,
                        sorted(allowed_tables),
                    )
                    return GeneratorMissingInfo(
                        status="missing_information",
                        missing_tables=unselected_tables,
                        reason=f"Generated query requires table(s) {', '.join(unselected_tables)} which are not in the Selected Schema.",
                    )
            except (sqlglot.errors.SqlglotError, ValueError, AttributeError) as parse_err:
                logger.debug("sqlglot schema containment check skipped: %s", parse_err)

        return GeneratedQuery(
            database_type=DatabaseType.POSTGRESQL,
            raw_query=sql,
            formatted_query=sql,
        )
