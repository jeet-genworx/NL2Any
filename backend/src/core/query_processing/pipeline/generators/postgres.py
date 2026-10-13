"""PostgreSQL query generator translating QueryPlan into SQL."""

import logging
from pathlib import Path
import re
from backend.src.config import settings
from backend.src.utils.text_utils import (
    clean_unreadable_characters,
    strip_ansi_escapes,
    strip_think_tags,
)
from backend.src.schemas.pipeline import GeneratedQuery, QueryPlan, RelevantSchema
from backend.src.data.models.schema import DatabaseType
from backend.src.core.query_processing.pipeline.generators.base import QueryGenerator
from backend.src.core.query_processing.pipeline.planner import (
    _format_relevant_schema_for_planner,
    format_column_manifest,
)
from backend.src.control.providers.model.base import ModelProvider
from backend.src.control.providers.model.koboldcpp import KoboldCppProvider

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


# The JOIN PLAN the generator prompt asks for sits ahead of the SQL as working
# notes. It is prose, and prose is where a naive "find the first SELECT or WITH"
# scan goes wrong: a fanout line reading "handled ... with line items" matches
# the bare keyword WITH, and the extracted query then begins mid-sentence.
_JOIN_PLAN_BLOCK = re.compile(
    r"^\s*JOIN\s+PLAN\b.*?(?:^\s*fanout\s*:[^\n]*$|\Z)",
    re.IGNORECASE | re.DOTALL | re.MULTILINE,
)

# A statement start, not a word: SELECT, or WITH introducing a real CTE
# ("WITH name AS ("). A bare "with" in a sentence matches neither.
_STATEMENT_START = re.compile(
    r"\bSELECT\b|\bWITH\s+(?:RECURSIVE\s+)?[A-Za-z_\"][\w\"]*\s+AS\s*\(",
    re.IGNORECASE,
)


def _extract_sql(text: str) -> str:
    cleaned = clean_unreadable_characters(strip_ansi_escapes(strip_think_tags(text))).strip()

    # 1. Match ```sql ... ```
    fence_match = re.search(r"```(?:sql)?\s*(.*?)\s*```", cleaned, re.DOTALL | re.IGNORECASE)
    if fence_match:
        sql = fence_match.group(1).strip()
    else:
        # 2. No fence: drop the JOIN PLAN notes, then start at the statement.
        without_plan = _JOIN_PLAN_BLOCK.sub("", cleaned).strip()
        candidate = without_plan or cleaned
        start = _STATEMENT_START.search(candidate)
        if start:
            sql = candidate[start.start() :].strip()
        else:
            sql = candidate

    if not sql or sql.strip() == "":
        return ""

    # Strip trailing markdown / extra notes after semicolon
    if ";" in sql:
        sql = sql[: sql.index(";") + 1].strip()
    elif not sql.endswith(";"):
        sql = sql + ";"

    return clean_unreadable_characters(strip_ansi_escapes(sql))


def _delete_predicate(column_name: str, column_type: str) -> str:
    """How to exclude deleted rows for one marker column.

    A boolean flag and a timestamp mark deletion differently -- is_deleted is
    FALSE for a live row, deleted_at is NULL for one -- and writing the wrong
    predicate either errors or silently returns nothing.

    The declared type decides when it is informative. When it is not -- an
    ingested schema may record a flag as "text" or leave the type vague -- the
    name does: an is_/has_ prefix is a boolean by convention everywhere.
    """
    type_text = (column_type or "").lower()
    if "bool" in type_text:
        return f"{column_name} IS FALSE"
    if column_name.lower().startswith(("is_", "has_")):
        return f"{column_name} IS FALSE"
    return f"{column_name} IS NULL"


def format_row_scope(schema: RelevantSchema) -> str:
    """State which deletion-marker and scope columns THIS slice actually has.

    Which column marks a row deleted, and which partitions rows between
    tenants, is a per-database convention: is_deleted, deleted_at, is_archived,
    tenant_id, org_id, account_id are all in use, and many schemas have none of
    them. The names looked for are configurable (SOFT_DELETE_COLUMNS,
    ROW_SCOPE_COLUMNS), and what is reported here is measured from the slice.

    Asking the model which tables are soft-deletable makes it guess, and it
    guesses wrong: told to exclude deleted rows it added "is_deleted IS FALSE"
    to two tables that have no such column, and given only a positive
    tenant-column list it joined on a tenant column another table lacked. So
    every column gets a positive AND a negative list -- the negative list is
    what actually stops the invention.
    """
    table_names = [obj.name for obj in schema.objects]
    if not table_names:
        return "No tables in this slice."

    columns_by_table = {
        obj.name: {f.name.lower(): (f.name, f.type) for f in obj.fields}
        for obj in schema.objects
    }

    lines: list[str] = []

    # Deletion markers: reported per predicate, since a table may use a flag
    # while another uses a timestamp.
    marker_hits: dict[str, list[str]] = {}
    for name in table_names:
        for candidate in settings.soft_delete_column_names:
            found = columns_by_table[name].get(candidate)
            if found:
                marker_hits.setdefault(_delete_predicate(found[0], found[1]), []).append(name)
                break
    if marker_hits:
        for predicate, owners in marker_hits.items():
            lines.append(
                f"Tables that mark deleted rows -- exclude them with \"{predicate}\" "
                f"on these tables and ONLY these: " + ", ".join(owners)
            )
        clean = [n for n in table_names if not any(n in o for o in marker_hits.values())]
        if clean:
            lines.append(
                "Tables with NO deletion marker -- adding one is a hallucinated "
                "column and the query will be rejected: " + ", ".join(clean)
            )
    else:
        lines.append(
            "No table in this slice marks rows as deleted. Do NOT add any "
            "deleted/archived filter; no such column exists here."
        )

    # Scope columns: grouped by column name, since the join condition names it.
    scope_hits: dict[str, list[str]] = {}
    for name in table_names:
        for candidate in settings.row_scope_column_names:
            found = columns_by_table[name].get(candidate)
            if found:
                scope_hits.setdefault(found[0], []).append(name)
                break
    if scope_hits:
        for column, owners in scope_hits.items():
            lines.append(
                f"Tables carrying the scope column \"{column}\" -- include it in the "
                f"ON clause only when BOTH sides of that join are listed here: "
                + ", ".join(owners)
            )
        unscoped = [n for n in table_names if not any(n in o for o in scope_hits.values())]
        if unscoped:
            lines.append(
                "Tables with NO scope column -- referencing one on these is a "
                "hallucinated column and the query will be rejected: "
                + ", ".join(unscoped)
            )
    else:
        lines.append(
            "No table in this slice carries a tenant or organisation scope "
            "column. Do NOT invent one."
        )

    return "\n".join(lines)


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
            cleaned_feedback = clean_unreadable_characters(strip_ansi_escapes(feedback))
            feedback_section = (
                f"ATTENTION - PREVIOUS ATTEMPT FAILED WITH FEEDBACK:\n{cleaned_feedback}\n"
                "Please fix the above issue in the generated query."
            )

        prompt = template.format(
            question=question,
            plan_context=plan_context,
            schema_context=schema_context,
            column_manifest=format_column_manifest(schema),
            row_scope=format_row_scope(schema),
            feedback_section=feedback_section,
        )

        raw_response = await self.provider.generate(
            prompt=prompt,
            temperature=0.1,
            max_tokens=settings.model_max_tokens,
        )

        sql = _extract_sql(raw_response)
        return GeneratedQuery(
            database_type=DatabaseType.POSTGRESQL,
            raw_query=sql,
            formatted_query=sql,
        )
