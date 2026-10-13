"""MongoDB query generator translating user question and Selected Schema directly into typed MongoQuery."""

import json
import logging
from pathlib import Path
from typing import Any

from backend.src.config import settings
from backend.src.utils.text_utils import (
    clean_unreadable_characters,
    estimate_token_count,
    extract_json_block,
    strip_ansi_escapes,
)
from backend.src.schemas.pipeline import GeneratedQuery, GeneratorMissingInfo, MongoQuery, RelevantSchema
from backend.src.data.models.schema import DatabaseType
from backend.src.core.query_processing.pipeline.generators.base import QueryGenerator
from backend.src.core.query_processing.pipeline.selector import format_relevant_schema
from backend.src.control.providers.model.base import ModelProvider
from backend.src.control.providers.model.koboldcpp import KoboldCppProvider

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
        formatted_json = json.dumps(mongo_query.model_dump(exclude_none=True), indent=2)

        return GeneratedQuery(
            database_type=DatabaseType.MONGODB,
            raw_query=mongo_query,
            formatted_query=formatted_json,
        )
