"""Query validator combining deterministic schema AST checks and SLM validation."""

import logging
from pathlib import Path
import re
from typing import Any
import sqlglot
from sqlglot import exp

from backend.src.config import settings
from backend.src.utils.text_utils import (
    clean_unreadable_characters,
    extract_json_block,
    strip_ansi_escapes,
)
from backend.src.schemas.pipeline import (
    GeneratedQuery,
    MongoQuery,
    QueryPlan,
    RelevantSchema,
    ValidationErrorType,
    ValidationResult,
)
from backend.src.data.models.schema import DatabaseType
from backend.src.core.query_processing.pipeline.selector import format_relevant_schema
from backend.src.control.providers.model.base import ModelProvider
from backend.src.control.providers.model.koboldcpp import KoboldCppProvider

logger = logging.getLogger(__name__)


def _get_validator_prompt_path() -> Path:
    p = Path("backend/src/core/query_processing/prompts/validator.txt")
    if p.exists():
        return p
    alt = Path(__file__).resolve().parents[1] / "prompts" / "validator.txt"
    if alt.exists():
        return alt
    return p


DEFAULT_VALIDATOR_PROMPT = _get_validator_prompt_path()

UNSAFE_SQL_PATTERN = re.compile(
    r"\b(INSERT\s+INTO|UPDATE\s+\w+\s+SET|DELETE\s+FROM|DROP\s+(TABLE|DATABASE|VIEW)|ALTER\s+TABLE|TRUNCATE|CREATE\s+(TABLE|DATABASE)|GRANT\s+|REVOKE\s+)\b",
    re.IGNORECASE,
)


def _clean_sqlglot_error(err: Exception | str) -> str:
    """Format and clean sqlglot exception message of ANSI escapes and internal class reprs."""
    text = clean_unreadable_characters(strip_ansi_escapes(str(err)))
    text = re.sub(r"<class 'sqlglot\.expressions\.[^.']+\.([A-Za-z0-9_]+)'>", r"'\1'", text)
    text = re.sub(r"<class '[^']+'>", "expression", text)
    return text.strip()


def validate_sql_schema_references(sql: str, schema: RelevantSchema) -> list[str]:
    """Deterministically inspect SQL AST to ensure all referenced tables and columns exist in schema."""
    issues: list[str] = []
    try:
        parsed = sqlglot.parse_one(sql, read="postgres")
    except Exception as err:
        return [_clean_sqlglot_error(f"SQL syntax error: {err}")]

    allowed_tables = schema.get_object_names()
    all_allowed_columns: set[str] = set()
    table_columns_map: dict[str, set[str]] = {}

    for obj in schema.objects:
        tbl_name = obj.name.lower()
        cols = {f.name.lower() for f in obj.fields}
        table_columns_map[tbl_name] = cols
        if "." in tbl_name:
            short_name = tbl_name.split(".", 1)[1]
            if short_name not in table_columns_map:
                table_columns_map[short_name] = cols
            else:
                table_columns_map[short_name] = table_columns_map[short_name].union(cols)
        all_allowed_columns.update(cols)

    # Collect CTE names, subquery aliases, and projected column aliases defined in the query
    cte_names = {cte.alias_or_name.lower() for cte in parsed.find_all(exp.CTE) if cte.alias_or_name}
    subquery_aliases = {s.alias.lower() for s in parsed.find_all(exp.Subquery) if s.alias}
    query_column_aliases = {alias_exp.alias.lower() for alias_exp in parsed.find_all(exp.Alias) if alias_exp.alias}

    # 1. Check referenced tables
    table_alias_map: dict[str, str] = {}
    referenced_tables: set[str] = set()
    for table_exp in parsed.find_all(exp.Table):
        t_name = table_exp.name.lower()
        full_name = f"{table_exp.db.lower()}.{t_name}" if table_exp.db else t_name
        if t_name:
            referenced_tables.add(full_name)
            is_allowed = (
                full_name in allowed_tables
                or t_name in allowed_tables
                or any(a.split(".", 1)[-1] == t_name for a in allowed_tables)
                or t_name in cte_names
                or t_name in subquery_aliases
                or full_name in cte_names
                or full_name in subquery_aliases
            )
            if not is_allowed:
                issues.append(f"Table '{full_name}' does not exist in relevant schema {sorted(allowed_tables)}")

        # Map alias and table name to canonical schema table name
        canonical = full_name if full_name in table_columns_map else (t_name if t_name in table_columns_map else full_name)
        if table_exp.alias:
            table_alias_map[table_exp.alias.lower()] = canonical
        if t_name:
            table_alias_map[t_name] = canonical
        if table_exp.db:
            table_alias_map[full_name] = canonical

    # 2. Check referenced columns
    for col_exp in parsed.find_all(exp.Column):
        col_name = col_exp.name.lower()
        # Skip star wildcard or projected column aliases (e.g. AS total_sales in ORDER BY / HAVING)
        if col_name in ("*", "") or col_name in query_column_aliases:
            continue

        raw_qualifier = col_exp.table.lower() if col_exp.table else None
        table_qualifier = table_alias_map.get(raw_qualifier, raw_qualifier)

        if table_qualifier and table_qualifier in table_columns_map:
            if col_name not in table_columns_map[table_qualifier]:
                issues.append(
                    f"Column '{col_name}' does not exist in table '{table_qualifier}'"
                )
        elif table_qualifier and (table_qualifier in cte_names or table_qualifier in subquery_aliases):
            continue
        elif table_qualifier and table_qualifier not in allowed_tables and table_qualifier not in table_columns_map:
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

    return [clean_unreadable_characters(strip_ansi_escapes(i)) for i in issues]


from backend.src.core.query_processing.pipeline.validators.base import QueryValidator as BaseQueryValidator
from backend.src.core.query_processing.pipeline.validators.postgres import PostgresQueryValidator
from backend.src.core.query_processing.pipeline.validators.mongo import MongoQueryValidator


class QueryValidator(BaseQueryValidator):
    """Composite query validator delegating to PostgresQueryValidator or MongoQueryValidator."""

    def __init__(
        self,
        provider: ModelProvider | None = None,
        prompt_path: Path | str | None = None,
        postgres_validator: PostgresQueryValidator | None = None,
        mongo_validator: MongoQueryValidator | None = None,
    ) -> None:
        self.provider = provider or KoboldCppProvider()
        self.postgres_validator = postgres_validator or PostgresQueryValidator(
            provider=self.provider, prompt_path=prompt_path
        )
        self.mongo_validator = mongo_validator or MongoQueryValidator(
            provider=self.provider, prompt_path=prompt_path
        )

    @property
    def last_prompt_tokens(self) -> int:
        return max(
            getattr(self.postgres_validator, "last_prompt_tokens", 0),
            getattr(self.mongo_validator, "last_prompt_tokens", 0),
        )

    async def validate(
        self,
        question: str,
        generated_query: Any = None,
        schema: Any = None,
        plan: Any = None,
        **kwargs: Any,
    ) -> ValidationResult:
        """Route query validation to specialized validator based on database type."""
        if isinstance(generated_query, GeneratedQuery):
            active_query = generated_query
        elif isinstance(schema, GeneratedQuery):
            active_query = schema
        elif isinstance(kwargs.get("generated_query"), GeneratedQuery):
            active_query = kwargs["generated_query"]
        else:
            return ValidationResult(
                valid=False,
                error_type=ValidationErrorType.GENERATION_ERROR,
                issues=["Invalid arguments passed to query validator: missing GeneratedQuery."],
                suggestion="Provide a valid GeneratedQuery.",
            )

        if active_query.database_type == DatabaseType.POSTGRESQL:
            return await self.postgres_validator.validate(
                question=question,
                generated_query=generated_query,
                schema=schema,
                plan=plan,
                **kwargs,
            )
        elif active_query.database_type == DatabaseType.MONGODB:
            return await self.mongo_validator.validate(
                question=question,
                generated_query=generated_query,
                schema=schema,
                plan=plan,
                **kwargs,
            )
        else:
            return ValidationResult(
                valid=False,
                error_type=ValidationErrorType.GENERATION_ERROR,
                issues=[f"Unsupported database type: {active_query.database_type}"],
                suggestion="Provide PostgreSQL or MongoDB query.",
            )

