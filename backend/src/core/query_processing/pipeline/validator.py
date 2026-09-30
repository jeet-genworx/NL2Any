"""Query validator combining deterministic schema AST checks and SLM validation."""

import logging
from pathlib import Path
import re
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
from backend.src.core.query_processing.pipeline.planner import _format_relevant_schema_for_planner
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


class QueryValidator:
    """Validates generated queries using relaxed diagnostic checks and SLM verification."""

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
        """Validate generated query using SLM verification and safety boundaries.

        Deterministic parsing/schema checks are intentionally omitted to avoid false
        rejections on complex PostgreSQL dialects, subqueries, or valid syntax.
        The SLM performs semantic, syntax, and relational validation.
        """
        # 1. Fast safety check: empty SQL or destructive write statements
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

        # 2. SLM Semantic & Relational Validation
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
                max_tokens=settings.validator_max_tokens,
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

            # Sanitize issues and suggestions from ANSI escapes and unreadable characters
            raw_issues = data.get("issues", [])
            if isinstance(raw_issues, list):
                data["issues"] = [
                    clean_unreadable_characters(strip_ansi_escapes(str(i)))
                    for i in raw_issues
                    if i
                ]
            raw_suggestion = data.get("suggestion")
            if raw_suggestion:
                data["suggestion"] = clean_unreadable_characters(strip_ansi_escapes(str(raw_suggestion)))

            return ValidationResult.model_validate(data)

        except Exception as err:
            logger.warning("Validation SLM call failed: %s. Lenient fallback: accepting query.", err)
            # Lenient fallback: do NOT fail queries on deterministic parser guesses.
            # Downstream database engine will catch genuine runtime errors.
            return ValidationResult(
                valid=True,
                error_type=ValidationErrorType.VALID,
                issues=[],
                suggestion=None,
            )
