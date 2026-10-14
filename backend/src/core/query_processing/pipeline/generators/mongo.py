"""MongoDB query generator translating user question and Selected Schema directly into typed MongoQuery."""

import json
import logging
from pathlib import Path
from typing import Any

from backend.src.config import settings
from backend.src.control.providers.model.base import ModelProvider
from backend.src.control.providers.model.koboldcpp import KoboldCppProvider
from backend.src.core.query_processing.pipeline.generators.base import QueryGenerator
from backend.src.core.query_processing.pipeline.selector import format_relevant_schema
from backend.src.data.models.schema import DatabaseType
from backend.src.schemas.pipeline import (
    GeneratedQuery,
    GeneratorMissingInfo,
    MongoQuery,
    RelevantSchema,
)
from backend.src.utils.text_utils import (
    clean_unreadable_characters,
    estimate_token_count,
    extract_json_block,
    strip_ansi_escapes,
)

logger = logging.getLogger(__name__)


def _get_mongo_prompt_path() -> Path:
    p = Path("backend/src/core/query_processing/prompts/mongo_query_generator.txt")
    if p.exists():
        return p
    alt = Path(__file__).resolve().parents[2] / "prompts" / "mongo_query_generator.txt"
    if alt.exists():
        return alt
    return p


DEFAULT_MONGO_PROMPT = _get_mongo_prompt_path()


class MongoQueryGenerator(QueryGenerator):
    """Generates structured, typed MongoQuery objects directly from question and selected schema."""

    def __init__(
        self,
        provider: ModelProvider | None = None,
        prompt_path: Path | str | None = None,
    ) -> None:
        self.provider = provider or KoboldCppProvider()
        self.prompt_path = Path(prompt_path or DEFAULT_MONGO_PROMPT)
        self.last_prompt_tokens: int = 0

    def _load_prompt_template(self) -> str:
        if self.prompt_path.exists():
            return self.prompt_path.read_text(encoding="utf-8")
        resolved = Path.cwd() / self.prompt_path
        if resolved.exists():
            return resolved.read_text(encoding="utf-8")
        raise FileNotFoundError(f"MongoDB generator prompt not found at {self.prompt_path}")

    async def generate_query(
        self,
        question: str,
        schema: Any = None,
        feedback: str | None = None,
        plan: Any = None,
        **kwargs: Any,
    ) -> GeneratedQuery | GeneratorMissingInfo:
        """Generate typed MongoQuery or return missing-information feedback."""
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
            cleaned_fb = clean_unreadable_characters(strip_ansi_escapes(active_feedback)).strip()
            if cleaned_fb.startswith("["):
                feedback_section = cleaned_fb
            else:
                feedback_section = (
                    f"ATTENTION - PREVIOUS ATTEMPT FAILED WITH FEEDBACK:\n{cleaned_fb}\n"
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

        data = extract_json_block(raw_response)
        if not isinstance(data, dict):
            raise ValueError(f"MongoDB generator expected a JSON object, got {type(data)}")

        if data.get("status") == "missing_information":
            missing = data.get("missing_tables") or data.get("missing_objects") or []
            if isinstance(missing, str):
                missing = [missing]
            return GeneratorMissingInfo(
                status="missing_information",
                missing_tables=missing,
                reason=str(data.get("reason", "Required schema information is missing.")),
            )
        mongo_query = MongoQuery.model_validate(data)

        # Check if MongoQuery references collections outside the Selected Schema
        if active_schema and active_schema.objects:
            allowed_collections = set()
            for obj in active_schema.objects:
                nl = obj.name.lower()
                allowed_collections.add(nl)
                if "." in nl:
                    allowed_collections.add(nl.split(".", 1)[-1])

            unselected_collections: list[str] = []
            base_col = mongo_query.collection.lower() if mongo_query.collection else ""
            if base_col and base_col not in allowed_collections:
                unselected_collections.append(mongo_query.collection)

            for stage in (mongo_query.pipeline or []):
                if isinstance(stage, dict) and "$lookup" in stage and isinstance(stage["$lookup"], dict):
                    from_col = stage["$lookup"].get("from", "")
                    if from_col and from_col.lower() not in allowed_collections and from_col not in unselected_collections:
                        unselected_collections.append(from_col)

            if unselected_collections:
                logger.info(
                    "Generated MongoDB query referenced unselected collections %s not in schema %s; emitting GeneratorMissingInfo",
                    unselected_collections,
                    sorted(allowed_collections),
                )
                return GeneratorMissingInfo(
                    status="missing_information",
                    missing_tables=unselected_collections,
                    reason=f"Generated query requires collection(s) {', '.join(unselected_collections)} which are not in the Selected Schema.",
                )

        formatted_json = json.dumps(mongo_query.model_dump(exclude_none=True), indent=2)

        return GeneratedQuery(
            database_type=DatabaseType.MONGODB,
            raw_query=mongo_query,
            formatted_query=formatted_json,
        )
