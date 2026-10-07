"""Database-independent semantic QueryPlanner stage."""

import logging
from pathlib import Path
import re

from backend.src.config import settings
from backend.src.utils.text_utils import (
    clean_unreadable_characters,
    extract_json_block,
    strip_ansi_escapes,
)
from backend.src.schemas.pipeline import (
    QueryPlan,
    QuestionAnalysis,
    RelevantSchema,
)
from backend.src.data.models.schema import DatabaseType
from backend.src.control.providers.model.base import ModelProvider
from backend.src.control.providers.model.koboldcpp import KoboldCppProvider

logger = logging.getLogger(__name__)


def _get_planner_prompt_path() -> Path:
    p = Path("backend/src/core/query_processing/prompts/planner.txt")
    if p.exists():
        return p
    alt = Path(__file__).resolve().parents[1] / "prompts" / "planner.txt"
    if alt.exists():
        return alt
    return p


DEFAULT_PLANNER_PROMPT = _get_planner_prompt_path()

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


def format_column_manifest(schema: RelevantSchema) -> str:
    """List every column per table, tersely, for lookup rather than reading.

    The verbose schema block above is the right shape for deciding what a column
    MEANS -- it carries descriptions and sample values -- and the wrong shape for
    answering "does this table have this column". A six-table slice renders as
    253 lines and 25k characters with one column per line, and a small model
    asked to find `customer_email` in there reports it missing, then qualifies it
    against whichever table its name resembles. The same facts on one line per
    table fit in a tenth of the space and can be scanned, so this is included
    alongside the verbose block, not instead of it.
    """
    lines: list[str] = []
    for obj in schema.objects:
        names = [f.name for f in obj.fields]
        # Dotted paths for subdocument fields, so a Mongo collection's nested
        # keys are listed the way a query has to spell them.
        for f in obj.fields:
            names.extend(f"{f.name}.{nested.name}" for nested in f.nested)
        lines.append(f"{obj.name}: {', '.join(names)}")
    return "\n".join(lines)


def _clean_sql_syntax(text: str) -> str:
    """Strip extraneous SQL syntax keywords that SLM might mistakenly include."""
    cleaned = SQL_KEYWORD_PATTERN.sub("", text).strip()
    return cleaned if cleaned else text


def _build_field_index(
    relevant_schema: RelevantSchema,
) -> tuple[set[str], dict[str, set[str]]]:
    """Index the slice's fields globally and per table.

    The per-table map is keyed by both the full object name and its bare name,
    because the planner refers to `discrepancies_details.customer_email` while
    the schema object is `finopsiq.discrepancies_details`.
    """
    all_fields: set[str] = set()
    by_table: dict[str, set[str]] = {}
    for obj in relevant_schema.objects:
        columns = {f.lower() for f in obj.all_field_paths()}
        columns.update(f.name.lower() for f in obj.fields)
        full = obj.name.lower()
        by_table.setdefault(full, set()).update(columns)
        by_table.setdefault(full.split(".")[-1], set()).update(columns)
        all_fields.update(columns)
        all_fields.update(f"{full}.{c}" for c in columns)
    return all_fields, by_table


def _field_is_valid(
    field: str,
    all_fields: set[str],
    by_table: dict[str, set[str]],
) -> bool:
    """Whether a plan field exists, honouring its table qualifier.

    Matching on the bare leaf name alone is what let
    `discrepancies_details.customer_email` through: `customer_email` does exist
    -- on `invoices` -- so a filter was planned against a table that has no such
    column, the generator faithfully emitted it, and validation rejected the
    query on every retry because the plan kept demanding it. When a qualifier is
    present it must be the qualifier that decides.
    """
    norm = field.lower().strip()
    if not norm:
        return False
    if "." in norm:
        qualifier, column = norm.rsplit(".", 1)
        known = by_table.get(qualifier)
        if known is not None:
            return column in known
        # Unknown qualifier: it may be a query alias rather than a table, so
        # fall back to the global check rather than dropping a valid field.
    return norm in all_fields or norm.split(".")[-1] in all_fields


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
            cleaned_feedback = clean_unreadable_characters(strip_ansi_escapes(feedback))
            feedback_section = (
                f"\nATTENTION - PREVIOUS PLAN FAILED VALIDATION (RETRY):\n{cleaned_feedback}\n"
                "Please fix the above plan issues."
            )

        prompt = template.format(
            question=question_analysis.question,
            database_type=database_type.value,
            subjective=", ".join(question_analysis.subjective) or "None",
            objective=", ".join(question_analysis.objective) or "None",
            schema_context=schema_context,
            column_manifest=format_column_manifest(relevant_schema),
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
        all_allowed_fields, fields_by_table = _build_field_index(relevant_schema)

        # 1. Validate sources and detect any missing tables whose schema is not retrieved
        def _is_table_in_allowed(tbl_name: str) -> bool:
            t = tbl_name.lower().strip()
            if not t:
                return True
            if t in allowed_sources:
                return True
            short_t = t.split(".")[-1]
            return any(a.split(".")[-1] == short_t for a in allowed_sources)

        missing_tables: list[str] = []
        for m in plan.missing_tables:
            m_clean = m.strip()
            if m_clean and not _is_table_in_allowed(m_clean) and m_clean not in missing_tables:
                missing_tables.append(m_clean)

        valid_sources = []
        for s in plan.sources:
            s_clean = s.strip()
            if _is_table_in_allowed(s_clean):
                valid_sources.append(s_clean)
            else:
                if s_clean and s_clean not in missing_tables:
                    missing_tables.append(s_clean)

        for rel in plan.relationships_used:
            for part in rel.split("->"):
                tokens = part.strip().split(".")
                if len(tokens) >= 3:
                    tbl = f"{tokens[0]}.{tokens[1]}"
                elif len(tokens) >= 2:
                    tbl = tokens[0]
                else:
                    tbl = part.strip()
                if tbl and not _is_table_in_allowed(tbl) and tbl not in missing_tables:
                    missing_tables.append(tbl)

        plan.missing_tables = missing_tables
        if not valid_sources and relevant_schema.objects:
            valid_sources = [relevant_schema.objects[0].name]
        plan.sources = valid_sources

        # 1b. Qualify bare projection columns against the slice.
        #
        # The plan names projections without a table, so the generator has to
        # guess which one owns each column -- and it guesses by name, putting
        # `customer_email` on `customers` when the column actually lives on
        # `invoices`. Validation then rejects the query, and the generator
        # reproduces it verbatim on every retry, because the feedback does not
        # change the plan it is working from. Resolving the owner here removes
        # the guess. Only unambiguous columns are qualified; one that several
        # slice tables share is left alone, since the generator picking any of
        # them is correct.
        # Qualified with the BARE table name, not the schema-qualified one: the
        # generator aliases tables by their bare name, and the validator compares
        # the plan's projections to the query's columns as strings. Writing
        # `finopsiq.products.product_id` where the SQL says `products.product_id`
        # reads as a difference to it, and it reports a column as missing that is
        # plainly in the SELECT list.
        owners: dict[str, list[str]] = {}
        for obj in relevant_schema.objects:
            bare = obj.name.split(".")[-1]
            for field in obj.fields:
                owners.setdefault(field.name.lower(), []).append(bare)
        qualified_projections: list[str] = []
        for projection in plan.projections:
            candidate = projection.strip()
            if candidate and candidate != "*" and "." not in candidate:
                tables = owners.get(candidate.lower(), [])
                if len(tables) == 1:
                    qualified = f"{tables[0]}.{candidate}"
                    if qualified != candidate:
                        logger.info(
                            "Qualified plan projection %r as %r", candidate, qualified
                        )
                    candidate = qualified
            qualified_projections.append(candidate)
        plan.projections = qualified_projections

        # 2. Sanitize and validate filters (strip SQL keywords like WHERE)
        sanitized_filters = []
        for flt in plan.filters:
            clean_field = _clean_sql_syntax(flt.field)
            clean_op = _clean_sql_syntax(flt.operator).lower() or "equals"
            # Verify the field exists on the table it is qualified with
            if _field_is_valid(clean_field, all_allowed_fields, fields_by_table):
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
            if _field_is_valid(cg, all_allowed_fields, fields_by_table):
                clean_group.append(cg)
            else:
                logger.warning("Dropped group_by on unknown field: %r", g)
        plan.group_by = clean_group

        clean_order = []
        for o in plan.order_by:
            co = _clean_sql_syntax(o.field)
            if _field_is_valid(co, all_allowed_fields, fields_by_table):
                o.field = co
                clean_order.append(o)
            else:
                logger.warning("Dropped order_by on unknown field: %r", o.field)
        plan.order_by = clean_order

        # 4. Drop an incoherent GROUP BY.
        #
        # Grouping with nothing aggregated is only meaningful when the grouped
        # fields cover every projection, which is a DISTINCT by another name and
        # is left alone. Grouping a SUBSET of the projections with no aggregate
        # cannot be executed at all -- PostgreSQL rejects it with "column must
        # appear in the GROUP BY clause or be used in an aggregate function" --
        # and the planner emits it readily, three grouped fields against twelve
        # projections. The generator then either obeys the plan and produces
        # invalid SQL, or ignores it and gets failed for not implementing the
        # plan, so the incoherent clause is removed here instead.
        if plan.group_by and not plan.aggregations:
            grouped = {g.lower().split(".")[-1] for g in plan.group_by}
            projected = {
                p.lower().split(".")[-1]
                for p in plan.projections
                if p.strip() and p.strip() != "*"
            }
            if projected and not projected <= grouped:
                logger.warning(
                    "Dropped GROUP BY %s: no aggregations, and it covers only "
                    "%d of %d projected fields, which PostgreSQL cannot execute.",
                    plan.group_by,
                    len(projected & grouped),
                    len(projected),
                )
                plan.group_by = []

        return plan
