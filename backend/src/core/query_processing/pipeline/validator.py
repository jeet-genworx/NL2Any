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
from backend.src.core.query_processing.pipeline.planner import (
    _format_relevant_schema_for_planner,
    format_column_manifest,
)
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


# Existence claims the model makes after sqlglot has already verified every
# table and column in the query. Instructing it not to make them is not enough:
# told to stop saying "does not exist", it says "is not present in" instead and
# rejects a valid query anyway. Existence is a fact Python has settled, so these
# are discarded rather than argued with.
EXISTENCE_CLAIM_PATTERN = re.compile(
    r"(does\s+not\s+exist"
    r"|do\s+not\s+exist"
    r"|is\s+not\s+(present|available|defined|a\s+column|a\s+field)"
    r"|are\s+not\s+(present|available|defined)"
    r"|not\s+found\s+in"
    r"|no\s+such\s+(column|field|table)"
    r"|does\s+not\s+(have|contain)\s+(a\s+)?(column|field)"
    r"|missing\s+from\s+the\s+(relevant\s+)?schema"
    r"|not\s+part\s+of\s+the\s+(relevant\s+)?schema)",
    re.IGNORECASE,
)


# Claims that the query leaves out a column the plan asked for. These are
# checkable: the column either appears in the SELECT list or it does not. The
# model gets this wrong when the plan spells a column `finopsiq.products.x` and
# the query spells it `products.x` -- it compares the two lists as strings,
# finds a difference, and reports a column missing that is in plain sight.
OMISSION_CLAIM_PATTERN = re.compile(
    r"(does\s+not\s+(include|select|project|return|contain|have)"
    r"|do\s+not\s+(include|select|project|return|appear)"
    r"|(is|are)\s+not\s+(included|selected|projected|returned|present\s+in\s+the)"
    r"|missing\s+from\s+the\s+(projections|select|SELECT\s+list|result|query|output)"
    r"|omits?\s+the"
    r"|fails?\s+to\s+(include|select|project|return)"
    r"|should\s+(also\s+)?(be\s+)?(include|select|project))",
    re.IGNORECASE,
)

# Identifiers the model quotes when naming a column, e.g. 'products.product_id'.
_QUOTED_IDENTIFIER_PATTERN = re.compile(r"['\"`]([A-Za-z_][\w.]*)['\"`]")

# Claims about a specific clause are NEVER dropped by the check below.
#
# "...does not include these in a GROUP BY clause" matches the omission wording
# above while naming columns that are present as projections, so the presence
# check would call it false and silence it. But that claim is about WHERE the
# columns appear, not whether they appear, and "is this column in the query"
# cannot answer it -- a genuine grouping bug would be discarded on irrelevant
# evidence. Grouping, ordering and windowing complaints are kept and left for
# the generator to answer.
_CLAUSE_SCOPED_PATTERN = re.compile(
    r"\b(GROUP\s+BY|ORDER\s+BY|HAVING|DISTINCT|PARTITION\s+BY|WINDOW|group\s+by)\b",
    re.IGNORECASE,
)


def _columns_present_in_sql(sql: str) -> set[str]:
    """Every column name the query references, bare and qualified, lowercased.

    Used to check a claim about the query against the query itself rather than
    against the model's recollection of it.
    """
    present: set[str] = set()
    try:
        parsed = sqlglot.parse_one(sql, read="postgres")
    except Exception:
        return present
    for col in parsed.find_all(exp.Column):
        name = col.name.lower()
        if not name:
            continue
        present.add(name)
        if col.table:
            present.add(f"{col.table.lower()}.{name}")
    for alias in parsed.find_all(exp.Alias):
        if alias.alias:
            present.add(alias.alias.lower())
    return present


def _drop_false_omission_claims(
    issues: list[str],
    sql: str,
) -> tuple[list[str], list[str]]:
    """Discard "the query omits column X" claims when X is in the query.

    Only claims whose every named column is demonstrably present are dropped; a
    claim naming a column the query really does lack is left for the generator.
    """
    present = _columns_present_in_sql(sql)
    if not present:
        return issues, []

    kept: list[str] = []
    discarded: list[str] = []
    for issue in issues:
        if not OMISSION_CLAIM_PATTERN.search(issue):
            kept.append(issue)
            continue
        # A grouping or ordering complaint is about clause placement, which the
        # presence check cannot speak to. Keep it.
        if _CLAUSE_SCOPED_PATTERN.search(issue):
            kept.append(issue)
            continue
        named = _QUOTED_IDENTIFIER_PATTERN.findall(issue)
        # Compare on the leaf name: the qualifier is exactly what the model gets
        # wrong, and the question here is only whether the column is selected.
        leaves = {n.split(".")[-1].lower() for n in named}
        if leaves and leaves <= present:
            discarded.append(issue)
        else:
            kept.append(issue)
    return kept, discarded


def _drop_existence_claims(issues: list[str]) -> tuple[list[str], list[str]]:
    """Split model issues into ones worth acting on and false existence claims."""
    kept: list[str] = []
    discarded: list[str] = []
    for issue in issues:
        (discarded if EXISTENCE_CLAIM_PATTERN.search(issue) else kept).append(issue)
    return kept, discarded


def _column_location_hint(
    col_name: str,
    qualifier: str | None,
    table_columns_map: dict[str, set[str]],
) -> str:
    """Say where a column actually lives, and what the table has instead.

    Reporting only "column does not exist" leaves the generator to guess again,
    and it guesses the same way every retry -- `customer_email` sounds like it
    belongs to `customers`, so it keeps going back there. The slice already
    knows that the column is on `invoices` and that `customers` offers `email`,
    so saying both turns three wasted retries into one corrected query.
    """
    elsewhere = sorted(
        table for table, cols in table_columns_map.items() if col_name in cols
    )
    # The map holds both qualified and bare keys for one table; keep the longest
    # spelling of each so the hint names tables the way the query must.
    deduped: list[str] = []
    for table in sorted(elsewhere, key=len, reverse=True):
        if not any(existing.endswith(f".{table}") for existing in deduped):
            deduped.append(table)
    parts: list[str] = []
    if deduped:
        parts.append(f"it exists on {', '.join(sorted(deduped))}")
    if qualifier:
        from difflib import get_close_matches

        near = get_close_matches(col_name, sorted(table_columns_map.get(qualifier, ())), n=2, cutoff=0.6)
        if near:
            parts.append(f"'{qualifier}' has {', '.join(repr(n) for n in near)}")
    return f" ({'; '.join(parts)})" if parts else ""


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
                    + _column_location_hint(col_name, table_qualifier, table_columns_map)
                )
        elif table_qualifier and (table_qualifier in cte_names or table_qualifier in subquery_aliases):
            continue
        elif table_qualifier and table_qualifier not in allowed_tables and table_qualifier not in table_columns_map:
            # Qualifier might be an alias; check if col exists anywhere in allowed columns
            if col_name not in all_allowed_columns:
                issues.append(
                    f"Column '{col_name}' does not exist in any relevant table"
                    + _column_location_hint(col_name, None, table_columns_map)
                )
        else:
            if col_name not in all_allowed_columns:
                issues.append(
                    f"Column '{col_name}' does not exist in relevant schema"
                    + _column_location_hint(col_name, None, table_columns_map)
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
        """Validate a generated query against safety, the schema, then meaning.

        Three layers, cheapest and most certain first: a regex guard against
        empty or destructive SQL, a deterministic sqlglot check that every table
        and column referenced actually exists, and only then the SLM, for the
        judgements that need judgement -- whether the query answers the question
        and joins the tables the way the question means.
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

        # 2. Deterministic schema reference check.
        #
        # This used to be left to the SLM on the theory that an AST check would
        # false-reject complex PostgreSQL. In practice the opposite happened: a
        # 4B model asked to find six columns among a few hundred spread over
        # 25k characters of schema text reports the ones it cannot locate as
        # missing, so real queries were rejected with issues that were simply
        # untrue -- and the generator then "fixed" columns that were never
        # wrong. sqlglot resolves aliases, CTEs and subqueries properly, so the
        # question "does this column exist" is answered here, exactly, and the
        # model is left to judge only semantics.
        if generated_query.database_type == DatabaseType.POSTGRESQL:
            ref_issues = validate_sql_schema_references(sql_str, schema)
            if ref_issues:
                # The same bad column appears once per occurrence (SELECT,
                # WHERE, ORDER BY); the generator only needs telling once.
                deduped = list(dict.fromkeys(ref_issues))
                is_syntax = any(i.startswith("SQL syntax error") for i in deduped)
                return ValidationResult(
                    valid=False,
                    error_type=(
                        ValidationErrorType.SYNTAX_ERROR
                        if is_syntax
                        else ValidationErrorType.GENERATION_ERROR
                    ),
                    issues=deduped,
                    suggestion=(
                        "Fix the SQL syntax so the query parses."
                        if is_syntax
                        else "Use only columns that exist on the table they are "
                        "qualified with. Check the relevant schema above for the "
                        "correct table for each column."
                    ),
                )

        # 3. SLM Semantic & Relational Validation
        template = self._load_prompt_template()
        schema_context = _format_relevant_schema_for_planner(schema)
        plan_context = plan.model_dump_json(indent=2)

        prompt = template.format(
            question=question,
            database_type=generated_query.database_type.value,
            plan_context=plan_context,
            schema_context=schema_context,
            column_manifest=format_column_manifest(schema),
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

            # sqlglot already proved every reference resolves, so an existence
            # complaint here is false by construction. Dropping the last issue
            # this way means the query was only ever failing on that claim.
            if generated_query.database_type == DatabaseType.POSTGRESQL:
                kept, discarded = _drop_existence_claims(data.get("issues", []))
                kept, omission_discarded = _drop_false_omission_claims(kept, sql_str)
                discarded = discarded + omission_discarded
                if discarded:
                    logger.warning(
                        "Discarded %d false existence claim(s) from the validator "
                        "(schema references were verified deterministically): %s",
                        len(discarded),
                        discarded,
                    )
                    data["issues"] = kept
                    if not kept and data.get("error_type") in (
                        ValidationErrorType.GENERATION_ERROR,
                        ValidationErrorType.GENERATION_ERROR.value,
                        ValidationErrorType.PLANNER_ERROR,
                        ValidationErrorType.PLANNER_ERROR.value,
                    ):
                        data["valid"] = True
                        data["error_type"] = ValidationErrorType.VALID.value
                        data["suggestion"] = None

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
