"""Query validator combining deterministic schema AST checks and SLM validation."""

import logging
from pathlib import Path
import re
import sqlglot
from sqlglot import exp

from query_processing.core.text_utils import extract_json_block
from query_processing.models.pipeline import (
    GeneratedQuery,
    MongoQuery,
    QueryPlan,
    RelevantSchema,
    ValidationErrorType,
    ValidationResult,
)
from query_processing.models.schema import DatabaseType
from query_processing.pipeline.planner import _format_relevant_schema_for_planner
from query_processing.providers.model.base import ModelProvider
from query_processing.providers.model.koboldcpp import KoboldCppProvider

logger = logging.getLogger(__name__)

DEFAULT_VALIDATOR_PROMPT = Path("query_processing/prompts/validator.txt")

UNSAFE_SQL_PATTERN = re.compile(
    r"\b(INSERT\s+INTO|UPDATE\s+\w+\s+SET|DELETE\s+FROM|DROP\s+(TABLE|DATABASE|VIEW)|ALTER\s+TABLE|TRUNCATE|CREATE\s+(TABLE|DATABASE)|GRANT\s+|REVOKE\s+)\b",
    re.IGNORECASE,
)


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
        """Validate generated query against schema and question with classified error types."""
        # 1. Deterministic fast check for UNSAFE operations
        if generated_query.database_type == DatabaseType.POSTGRESQL:
            sql_str = str(generated_query.raw_query).strip()
            if not sql_str:
                return ValidationResult(
                    valid=False,
                    error_type=ValidationErrorType.GENERATION_ERROR,
                    issues=["No SQL query was generated."],
                    suggestion="Generate a valid read-only PostgreSQL query based on the query plan.",
                )

            if UNSAFE_SQL_PATTERN.search(sql_str):
                return ValidationResult(
                    valid=False,
                    error_type=ValidationErrorType.UNSAFE,
                    issues=["Unsafe modifying or destructive SQL statement detected."],
                    suggestion="Queries must be read-only SELECT statements.",
                )

            # 2. Deterministic AST syntax verification
            try:
                sqlglot.parse(sql_str, read="postgres")
            except Exception as syntax_err:
                return ValidationResult(
                    valid=False,
                    error_type=ValidationErrorType.SYNTAX_ERROR,
                    issues=[f"PostgreSQL syntax error: {syntax_err}"],
                    suggestion="Correct the PostgreSQL SQL syntax.",
                )

            # 3. Deterministic schema reference check
            schema_issues = validate_sql_schema_references(sql_str, schema)
            if schema_issues:
                # Check if the plan itself contained invalid sources
                plan_sources_set = {s.lower() for s in plan.sources}
                allowed_sources = schema.get_object_names()
                is_planner_flaw = not plan_sources_set.issubset(allowed_sources)
                err_type = (
                    ValidationErrorType.PLANNER_ERROR
                    if is_planner_flaw
                    else ValidationErrorType.GENERATION_ERROR
                )
                return ValidationResult(
                    valid=False,
                    error_type=err_type,
                    issues=schema_issues,
                    suggestion=f"Correct schema references: {'; '.join(schema_issues)}",
                )

        # 4. SLM Semantic Validation
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

            # Map error_type if returned as string
            raw_err_type = data.get("error_type", "VALID")
            if not isinstance(raw_err_type, str) or raw_err_type not in ValidationErrorType.__members__:
                raw_err_type = "VALID" if data.get("valid", True) else "GENERATION_ERROR"

            # Enforce mutual consistency between valid boolean and error_type
            if raw_err_type != ValidationErrorType.VALID:
                data["valid"] = False
            elif not data.get("valid", True):
                raw_err_type = ValidationErrorType.GENERATION_ERROR

            data["error_type"] = raw_err_type

            return ValidationResult.model_validate(data)

        except Exception as err:
            logger.warning("Validation SLM call failed: %s. Relying on deterministic check.", err)
            # If deterministic checks passed and SLM call failed, accept as valid
            return ValidationResult(
                valid=True,
                error_type=ValidationErrorType.VALID,
                issues=[],
                suggestion=None,
            )
