"""V2 Table selector stage: selects tables from complete database schema description."""

import logging
from pathlib import Path
import time
from typing import Any

from backend.src.config import settings
from backend.src.utils.text_utils import estimate_token_count, extract_json_block
from backend.src.schemas.pipeline import (
    CandidateTable,
    QuestionAnalysis,
    RelevantSchema,
    TableSelectionResult,
)
from backend.src.data.models.schema import DatabaseSchema
from backend.src.control.providers.model.base import ModelProvider
from backend.src.control.providers.model.koboldcpp import KoboldCppProvider
from backend.src.control.providers.model.koboldcpp_cache import (
    KoboldCppCacheManager,
    get_cache_manager,
)

logger = logging.getLogger(__name__)


def _get_selector_prompt_path() -> Path:
    p = Path("backend/src/core/query_processing/prompts/table_selection.txt")
    if p.exists():
        return p
    alt = Path(__file__).resolve().parents[1] / "prompts" / "table_selection.txt"
    if alt.exists():
        return alt
    return p


DEFAULT_SELECTOR_PROMPT = _get_selector_prompt_path()


def format_schema_description(schema: DatabaseSchema) -> str:
    """Format complete database schema with descriptions, columns, types, and relationships."""
    lines: list[str] = [
        f"Database: {schema.database_name} ({schema.database_type.value})",
        f"Total Tables/Collections: {len(schema.objects)}\n",
    ]

    for obj in schema.objects:
        desc = f" - {obj.description}" if obj.description else ""
        lines.append(f"• {obj.name} ({obj.kind.value}){desc}")
        fields_repr = []
        for f in obj.fields:
            f_desc = f": {f.description}" if f.description else ""
            fields_repr.append(f"    - {f.name} ({f.type}){f_desc}")
            for nested_f in f.nested:
                n_desc = f": {nested_f.description}" if nested_f.description else ""
                fields_repr.append(f"      - {f.name}.{nested_f.name} ({nested_f.type}){n_desc}")
        if fields_repr:
            lines.append("  Columns:")
            lines.extend(fields_repr)

    if schema.relationships:
        lines.append("\nKnown Relationships:")
        for r in schema.relationships:
            lines.append(
                f"• {r.from_object}.{r.from_field} -> {r.to_object}.{r.to_field} ({r.relationship_type})"
            )

    return "\n".join(lines)


def format_relevant_schema(schema: RelevantSchema | DatabaseSchema) -> str:
    """Format selected relevant schema for consumption by Query Generator and Validator."""
    lines: list[str] = []
    for obj in schema.objects:
        desc = f": {obj.description}" if obj.description else ""
        lines.append(f"• {obj.name} ({obj.kind.value}){desc}")
        for f in obj.fields:
            f_desc = f" - {f.description}" if f.description else ""
            samples = f" (samples: {f.sample_values})" if f.sample_values else ""
            lines.append(f"    - {f.name} ({f.type}){f_desc}{samples}")
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


# Backwards compatibility alias
_format_relevant_schema_for_planner = format_relevant_schema


def package_selected_schema(schema: DatabaseSchema, selected_tables: list[str]) -> RelevantSchema:
    """Ground and package selected tables with their full schema and relationships.

    Deterministic grounding only: verifies selected table names against the canonical schema,
    attaches their exact fields and descriptions, and includes relationships connecting them.
    Does NOT discover or add additional tables.
    """
    if not selected_tables or not schema:
        return RelevantSchema(objects=[], relationships=[])

    selected_set = {t.strip().lower() for t in selected_tables if t.strip()}
    matched_objects = []
    for obj in schema.objects:
        obj_lower = obj.name.lower()
        short_name = obj_lower.split(".")[-1]
        if obj_lower in selected_set or short_name in selected_set:
            matched_objects.append(obj)

    matched_names = {obj.name.lower() for obj in matched_objects}
    matched_short_names = {obj.name.lower().split(".")[-1] for obj in matched_objects}

    matched_relationships = []
    for rel in schema.relationships:
        from_low = rel.from_object.lower()
        to_low = rel.to_object.lower()
        from_short = from_low.split(".")[-1]
        to_short = to_low.split(".")[-1]

        from_matched = from_low in matched_names or from_short in matched_short_names
        to_matched = to_low in matched_names or to_short in matched_short_names

        if from_matched and to_matched:
            matched_relationships.append(rel)

    return RelevantSchema(
        objects=matched_objects,
        relationships=matched_relationships,
    )


class TableSelector:
    """Selects relevant tables from complete database schema description."""

    def __init__(
        self,
        provider: ModelProvider | None = None,
        prompt_path: Path | str | None = None,
        cache_manager: KoboldCppCacheManager | None = None,
    ) -> None:
        self.provider = provider or KoboldCppProvider()
        self.prompt_path = Path(prompt_path or DEFAULT_SELECTOR_PROMPT)
        self.cache_manager = cache_manager if cache_manager is not None else (
            get_cache_manager() if settings.koboldcpp_admin_enabled else None
        )
        self.last_prompt_tokens: int = 0

    def _load_prompt_template(self) -> str:
        if self.prompt_path.exists():
            return self.prompt_path.read_text(encoding="utf-8")
        resolved = Path.cwd() / self.prompt_path
        if resolved.exists():
            return resolved.read_text(encoding="utf-8")
        raise FileNotFoundError(f"Table selection prompt file not found at {self.prompt_path}")

    def build_static_prefix(self, schema: DatabaseSchema) -> str:
        """Format the byte-identical static prompt prefix (rules + schema context) ending on a newline boundary."""
        template = self._load_prompt_template()
        schema_context = format_schema_description(schema)
        if "{schema_context}" in template:
            idx = template.index("{schema_context}") + len("{schema_context}")
            prefix_tmpl = template[:idx]
            prefix = prefix_tmpl.format(schema_context=schema_context)
            if not prefix.endswith("\n\n"):
                prefix = prefix.rstrip("\n") + "\n\n"
            return prefix
        return format_schema_description(schema) + "\n\n"

    async def _generate_with_cache(
        self,
        db_key: str,
        schema: DatabaseSchema,
        prompt: str,
        prefix: str,
        max_toks: int,
    ) -> str:
        """Execute Stage 1 SLM generate under lock with KV cache restoration and fallback."""
        assert self.cache_manager is not None

        # 1. If background warm-up is in flight, await it
        warmup_task = self.cache_manager._warming_tasks.get(db_key)
        if warmup_task and not warmup_task.done():
            try:
                await warmup_task
            except Exception as w_err:
                logger.warning("Background warm-up failed for '%s': %s", db_key, w_err)

        slot = self.cache_manager.get_or_allocate_slot(db_key)

        # 2. Acquire concurrency lock
        async with self.cache_manager._slot_lock:
            # Check if warmed in slot; if not, lazy warm-up directly
            if not self.cache_manager.is_warmed(db_key):
                logger.info("Database '%s' not warmed in slot %d; performing lazy warm-up...", db_key, slot)
                warmup_prompt = f"{prefix}User Question:\n{settings.koboldcpp_warmup_dummy_question}\n"
                await self.provider.generate(
                    prompt=warmup_prompt,
                    temperature=0.0,
                    max_tokens=1,
                )
                await self.cache_manager.save_state(slot)

            # Load state
            t_load_start = time.perf_counter()
            loaded = await self.cache_manager.load_state(slot)
            t_load = time.perf_counter() - t_load_start

            if not loaded:
                logger.warning("load_state failed for '%s' (slot %d); re-warming once...", db_key, slot)
                warmup_prompt = f"{prefix}User Question:\n{settings.koboldcpp_warmup_dummy_question}\n"
                await self.provider.generate(
                    prompt=warmup_prompt,
                    temperature=0.0,
                    max_tokens=1,
                )
                await self.cache_manager.save_state(slot)
                loaded = await self.cache_manager.load_state(slot)
                if not loaded:
                    logger.warning("Re-warmed load_state still failed for slot %d; continuing with uncached generation", slot)

            t0 = time.perf_counter()
            resp = await self.provider.generate(
                prompt=prompt,
                temperature=0.0,
                max_tokens=max_toks,
            )
            gen_duration = time.perf_counter() - t0

            perf = await self.cache_manager.get_perf()
            logger.info(
                "Stage 1 SLM call '%s' (slot %d) | load: %.1fms | gen: %.2fs | speed: %.1f t/s",
                db_key,
                slot,
                t_load * 1000,
                gen_duration,
                perf.get("last_process_speed", 0.0),
            )
            return resp

    async def select(
        self,
        question: str | QuestionAnalysis,
        schema: DatabaseSchema,
        candidates: list[CandidateTable] | None = None,
        feedback: str | None = None,
        target_key: str | None = None,
        max_tokens: int | None = None,
    ) -> TableSelectionResult:
        """Select relevant tables from complete database description using SLM."""
        q_text = question.question if isinstance(question, QuestionAnalysis) else str(question)

        if not schema or not schema.objects:
            return TableSelectionResult(
                selected_objects=[],
                sufficient=False,
                missing_objects=[],
                reason="No schema objects available in database.",
                retrieval_hint=None,
            )

        schema_map = {obj.name.lower(): obj.name for obj in schema.objects}
        for obj in schema.objects:
            short = obj.name.lower().split(".")[-1]
            if short not in schema_map:
                schema_map[short] = obj.name

        feedback_context = ""
        if feedback:
            cleaned_fb = feedback.strip()
            if cleaned_fb.startswith("["):
                feedback_context = f"\n{cleaned_fb}\n\n"
            else:
                feedback_context = (
                    f"\nATTENTION - PREVIOUS ATTEMPT FEEDBACK (RETRY):\n{cleaned_fb}\n"
                    "Please reconsider table selection to address this feedback.\n"
                )

        db_key = target_key or getattr(schema, "database_name", "default")
        use_cache = (
            self.cache_manager is not None
            and isinstance(self.provider, KoboldCppProvider)
            and settings.koboldcpp_admin_enabled
            and db_key not in getattr(self.cache_manager, "_disabled_databases", set())
        )

        if use_cache and self.cache_manager is not None:
            prefix = self.cache_manager.get_cached_prefix(db_key)
            if not prefix:
                prefix = self.build_static_prefix(schema)
                self.cache_manager.cache_prefix(db_key, prefix)
        else:
            prefix = self.build_static_prefix(schema)

        self.last_prompt_tokens = 0
        prompt = f"{prefix}{feedback_context}User Question:\n{q_text}\n"
        self.last_prompt_tokens = estimate_token_count(prompt)
        max_toks = max_tokens if max_tokens is not None else settings.table_selector_max_tokens

        try:
            if use_cache and self.cache_manager is not None:
                raw_response = await self._generate_with_cache(
                    db_key=db_key,
                    schema=schema,
                    prompt=prompt,
                    prefix=prefix,
                    max_toks=max_toks,
                )
            else:
                raw_response = await self.provider.generate(
                    prompt=prompt,
                    temperature=0.0,
                    max_tokens=max_toks,
                )

            data = extract_json_block(raw_response)
            parsed = TableSelectionResult.model_validate(data)

            # Grounding: model may only select tables that actually exist in the schema
            validated: list[str] = []
            for item in parsed.selected_objects:
                item_lower = item.strip().lower()
                if item_lower in schema_map:
                    canonical_name = schema_map[item_lower]
                    if canonical_name not in validated:
                        validated.append(canonical_name)
                else:
                    logger.warning("Discarded non-existent/hallucinated table: %r", item)

            return TableSelectionResult(
                selected_objects=validated,
                sufficient=parsed.sufficient,
                missing_objects=parsed.missing_objects,
                reason=parsed.reason,
                retrieval_hint=parsed.retrieval_hint,
            )

        except Exception as err:
            logger.warning("Table selection SLM call failed: %s.", err)
            return TableSelectionResult(
                selected_objects=[],
                sufficient=False,
                missing_objects=[],
                reason=f"Table selection failed due to error: {err}",
                retrieval_hint=None,
            )

