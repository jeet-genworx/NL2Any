"""MongoDB query generator translating QueryPlan into typed MongoQuery."""

import json
import logging
from pathlib import Path
from nl2anyquery.core.text_utils import extract_json_block
from nl2anyquery.models.pipeline import GeneratedQuery, MongoQuery, QueryPlan, RelevantSchema
from nl2anyquery.models.schema import DatabaseType
from nl2anyquery.pipeline.generators.base import QueryGenerator
from nl2anyquery.pipeline.planner import _format_relevant_schema_for_planner
from nl2anyquery.providers.model.base import ModelProvider
from nl2anyquery.providers.model.koboldcpp import KoboldCppProvider

logger = logging.getLogger(__name__)

DEFAULT_MONGO_PROMPT = Path("prompts/mongo_query_generator.txt")


class MongoQueryGenerator(QueryGenerator):
    """Generates structured, typed MongoQuery objects."""

    def __init__(
        self,
        provider: ModelProvider | None = None,
        prompt_path: Path | str | None = None,
    ) -> None:
        self.provider = provider or KoboldCppProvider()
        self.prompt_path = Path(prompt_path or DEFAULT_MONGO_PROMPT)

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
        plan: QueryPlan,
        schema: RelevantSchema,
        feedback: str | None = None,
    ) -> GeneratedQuery:
        """Generate typed MongoQuery from QueryPlan and RelevantSchema."""
        template = self._load_prompt_template()
        schema_context = _format_relevant_schema_for_planner(schema)
        plan_context = plan.model_dump_json(indent=2)

        feedback_section = ""
        if feedback:
            feedback_section = (
                f"ATTENTION - PREVIOUS ATTEMPT FAILED WITH FEEDBACK:\n{feedback}\n"
                "Please fix the above issue in the generated query."
            )

        prompt = template.format(
            question=question,
            plan_context=plan_context,
            schema_context=schema_context,
            feedback_section=feedback_section,
        )

        raw_response = await self.provider.generate(
            prompt=prompt,
            temperature=0.1,
            max_tokens=600,
        )

        data = extract_json_block(raw_response)
        if not isinstance(data, dict):
            raise ValueError(f"MongoDB generator expected a JSON object, got {type(data)}")

        mongo_query = MongoQuery.model_validate(data)
        formatted_json = json.dumps(mongo_query.model_dump(exclude_none=True), indent=2)

        return GeneratedQuery(
            database_type=DatabaseType.MONGODB,
            raw_query=mongo_query,
            formatted_query=formatted_json,
        )
