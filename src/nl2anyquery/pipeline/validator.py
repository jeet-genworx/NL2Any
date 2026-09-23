"""Query validator combining deterministic schema AST checks and SLM validation."""

import logging
from pathlib import Path
import sqlglot
from sqlglot import exp

from nl2anyquery.core.text_utils import extract_json_block
from nl2anyquery.models.pipeline import (
    GeneratedQuery,
    MongoQuery,
    QueryPlan,
    RelevantSchema,
    ValidationResult,
)
from nl2anyquery.models.schema import DatabaseType
from nl2anyquery.pipeline.planner import _format_relevant_schema_for_planner
from nl2anyquery.providers.model.base import ModelProvider
from nl2anyquery.providers.model.koboldcpp import KoboldCppProvider

logger = logging.getLogger(__name__)

DEFAULT_VALIDATOR_PROMPT = Path("prompts/validator.txt")


def validate_sql_schema_references(sql: str, schema: RelevantSchema) -> list[str]:
    """Deterministically inspect SQL AST to ensure all referenced tables and columns exist in schema."""
    issues: list[str] = []
    try:
        parsed = sqlglot.parse_one(sql, read="postgres")
    except Exception as err:
        return [f"SQL syntax error: {err}"]

    allowed_tables = schema.get_object_names()
    all_allowed_columns: set[str] = set()
    table_columns_map: dict[str, set[str]] = {}

    for obj in schema.objects:
        tbl_name = obj.name.lower()
        cols = {f.name.lower() for f in obj.fields}
        table_columns_map[tbl_name] = cols
        all_allowed_columns.update(cols)

    # 1. Check referenced tables
    referenced_tables: set[str] = set()
    for table_exp in parsed.find_all(exp.Table):
        t_name = table_exp.name.lower()
        if t_name:
            referenced_tables.add(t_name)
            if t_name not in allowed_tables:
                issues.append(f"Table '{t_name}' does not exist in relevant schema {sorted(allowed_tables)}")

    # 2. Check referenced columns
    for col_exp in parsed.find_all(exp.Column):
        col_name = col_exp.name.lower()
        # Skip star wildcard or expressions
        if col_name in ("*", ""):
            continue

        table_qualifier = col_exp.table.lower() if col_exp.table else None
        if table_qualifier and table_qualifier in table_columns_map:
            if col_name not in table_columns_map[table_qualifier]:
                issues.append(
                    f"Column '{col_name}' does not exist in table '{table_qualifier}'"
                )
        elif table_qualifier and table_qualifier not in allowed_tables:
            # Qualifier might be an alias; check if col exists anywhere in allowed columns
            if col_name not in all_allowed_columns:
                issues.append(
                    f"Column '{col_name}' does not exist in any relevant table"
                )
        else:
            if col_name not in all_allowed_columns:
                issues.append(
                    f"Column '{col_name}' does not exist in relevant schema"
                )

    return issues


def validate_mongo_schema_references(query: MongoQuery, schema: RelevantSchema) -> list[str]:
    """Deterministically verify collection exists in schema for MongoDB."""
    issues: list[str] = []
    allowed_collections = schema.get_object_names()
    if query.collection.lower() not in allowed_collections:
        issues.append(
            f"Collection '{query.collection}' does not exist in relevant schema {sorted(allowed_collections)}"
        )
    return issues


class QueryValidator:
    """Validates generated queries using deterministic AST checks and SLM verification."""

    def __init__(
        self,
        provider: ModelProvider | None = None,
        prompt_path: Path | str | None = None,
    ) -> None:
        self.provider = provider or KoboldCppProvider()
        self.prompt_path = Path(prompt_path or DEFAULT_VALIDATOR_PROMPT)

    def _load_prompt_template(self) -> str:
        if self.prompt_path.exists():
            return self.prompt_path.read_text(encoding="utf-8")
        resolved = Path.cwd() / self.prompt_path
        if resolved.exists():
            return resolved.read_text(encoding="utf-8")
        raise FileNotFoundError(f"Validator prompt not found at {self.prompt_path}")

    async def validate(
        self,
        question: str,
        plan: QueryPlan,
        generated_query: GeneratedQuery,
        schema: RelevantSchema,
    ) -> ValidationResult:
        """Validate generated query against schema and question."""
        # 1. Deterministic AST/Schema verification (Refinement 3)
        deterministic_issues: list[str] = []
        if generated_query.database_type == DatabaseType.POSTGRESQL:
            sql_str = str(generated_query.raw_query)
            deterministic_issues = validate_sql_schema_references(sql_str, schema)
        elif isinstance(generated_query.raw_query, MongoQuery):
            deterministic_issues = validate_mongo_schema_references(generated_query.raw_query, schema)

        if deterministic_issues:
            return ValidationResult(
                valid=False,
                issues=deterministic_issues,
                suggestion=f"Correct schema references: {'; '.join(deterministic_issues)}",
            )

        # 2. SLM Semantic Validation
        template = self._load_prompt_template()
        schema_context = _format_relevant_schema_for_planner(schema)
        plan_context = plan.model_dump_json(indent=2)

        prompt = template.format(
            question=question,
            database_type=generated_query.database_type.value,
            plan_context=plan_context,
            schema_context=schema_context,
            query_context=generated_query.formatted_query,
        )

        try:
            raw_response = await self.provider.generate(
                prompt=prompt,
                temperature=0.0,
                max_tokens=800,
            )
            data = extract_json_block(raw_response)
            return ValidationResult.model_validate(data)
        except Exception as err:
            logger.warning("Validation SLM call failed: %s. Relying on deterministic check.", err)
            # If deterministic check passed and SLM call failed, accept with empty issues
            return ValidationResult(valid=True, issues=[], suggestion=None)
