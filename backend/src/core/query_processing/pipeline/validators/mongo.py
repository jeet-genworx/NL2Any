"""MongoDB query validator with safety checks and dedicated SLM verification."""

import logging
from pathlib import Path
from typing import Any

from backend.src.config import settings
from backend.src.utils.text_utils import (
    clean_unreadable_characters,
    extract_json_block,
    strip_ansi_escapes,
)
from backend.src.schemas.pipeline import (
    GeneratedQuery,
    MongoQuery,
    RelevantSchema,
    ValidationErrorType,
    ValidationResult,
)
from backend.src.core.query_processing.pipeline.selector import format_relevant_schema
from backend.src.core.query_processing.pipeline.validators.base import QueryValidator
from backend.src.control.providers.model.base import ModelProvider
from backend.src.control.providers.model.koboldcpp import KoboldCppProvider

logger = logging.getLogger(__name__)


def _get_mongo_validator_prompt_path() -> Path:
    p = Path("backend/src/core/query_processing/prompts/mongo_validator.txt")
    if p.exists():
        return p
    alt = Path(__file__).resolve().parents[2] / "prompts" / "mongo_validator.txt"
    if alt.exists():
        return alt
    return p


DEFAULT_MONGO_VALIDATOR_PROMPT = _get_mongo_validator_prompt_path()


class MongoQueryValidator(QueryValidator):
    """Specialized validator for MongoDB queries."""

    def __init__(
        self,
        provider: ModelProvider | None = None,
        prompt_path: Path | str | None = None,
    ) -> None:
        self.provider = provider or KoboldCppProvider()
        self.prompt_path = Path(prompt_path or DEFAULT_MONGO_VALIDATOR_PROMPT)

    def _load_prompt_template(self) -> str:
        if self.prompt_path.exists():
            return self.prompt_path.read_text(encoding="utf-8")
        resolved = Path.cwd() / self.prompt_path
        if resolved.exists():
            return resolved.read_text(encoding="utf-8")
        raise FileNotFoundError(f"Mongo validator prompt not found at {self.prompt_path}")

    async def validate(
        self,
        question: str,
        generated_query: Any = None,
        schema: Any = None,
        plan: Any = None,
        **kwargs: Any,
    ) -> ValidationResult:
        """Validate generated MongoDB query using safety checks and SLM verification."""
        if isinstance(generated_query, GeneratedQuery):
            active_query = generated_query
            active_schema = schema if isinstance(schema, RelevantSchema) else kwargs.get("schema", RelevantSchema())
        elif isinstance(schema, GeneratedQuery):
            active_query = schema
            active_schema = plan if isinstance(plan, RelevantSchema) else kwargs.get("schema", RelevantSchema())
        elif isinstance(kwargs.get("generated_query"), GeneratedQuery):
            active_query = kwargs["generated_query"]
            active_schema = kwargs.get("schema", RelevantSchema())
        else:
            return ValidationResult(
                valid=False,
                error_type=ValidationErrorType.GENERATION_ERROR,
                issues=["Invalid arguments passed to Mongo validator: missing GeneratedQuery."],
                suggestion="Provide a valid GeneratedQuery.",
            )

        raw = active_query.raw_query
        if not isinstance(raw, MongoQuery):
            return ValidationResult(
                valid=False,
                error_type=ValidationErrorType.SYNTAX_ERROR,
                issues=["Raw query is not a valid typed MongoQuery."],
                suggestion="Generate a valid MongoQuery JSON object.",
            )

        # 1. Fast read-only and operation safety check
        if raw.operation not in ("find", "aggregate"):
            return ValidationResult(
                valid=False,
                error_type=ValidationErrorType.UNSAFE,
                issues=[f"Prohibited MongoDB operation '{raw.operation}'. Only 'find' and 'aggregate' are allowed."],
                suggestion="Use 'find' or 'aggregate' operation.",
            )

        if raw.operation == "aggregate" and raw.pipeline:
            disallowed_stages = {"$out", "$merge", "$writeconcern", "$collmod", "$plancachestats"}
            known_bare_stages = {"match", "lookup", "unwind", "group", "sort", "limit", "project", "addfields", "count", "skip"}
            for idx, stage in enumerate(raw.pipeline):
                if isinstance(stage, dict):
                    for k in stage.keys():
                        k_clean = str(k).strip().lower()
                        if k_clean in disallowed_stages:
                            return ValidationResult(
                                valid=False,
                                error_type=ValidationErrorType.UNSAFE,
                                issues=[f"Stage {idx}: prohibited write stage '{k}'."],
                                suggestion="Aggregation pipeline must be read-only.",
                            )
                        if k_clean in known_bare_stages:
                            return ValidationResult(
                                valid=False,
                                error_type=ValidationErrorType.SYNTAX_ERROR,
                                issues=[f"Stage {idx}: stage operator '{k}' is missing '$' prefix; must be '${k_clean}'."],
                                suggestion=f"Prefix stage operator with '$': change '{k}' to '${k_clean}'.",
                            )
                        if k_clean in ("$group", "group") and isinstance(stage[k], dict):
                            for group_field, accum in stage[k].items():
                                if group_field != "_id" and isinstance(accum, dict):
                                    for op in accum.keys():
                                        if str(op).strip().lower() in ("$size", "size"):
                                            return ValidationResult(
                                                valid=False,
                                                error_type=ValidationErrorType.SYNTAX_ERROR,
                                                issues=[
                                                    f"Stage {idx} ($group): '$size' is not a valid accumulator operator in $group; use {{'$sum': 1}} to count documents in a group."
                                                ],
                                                suggestion="In a $group stage, replace '$size' with {'$sum': 1} to count documents.",
                                            )

        # 2. SLM Semantic & Relational Validation
        template = self._load_prompt_template()
        schema_context = format_relevant_schema(active_schema)

        prompt = template.format(
            question=question,
            schema_context=schema_context,
            query_context=active_query.formatted_query,
        )

        try:
            raw_response = await self.provider.generate(
                prompt=prompt,
                temperature=0.0,
                max_tokens=settings.validator_max_tokens,
            )
            data = extract_json_block(raw_response)

            raw_err_type = data.get("error_type", "VALID")
            if not isinstance(raw_err_type, str) or raw_err_type not in ValidationErrorType.__members__:
                raw_err_type = "VALID" if data.get("valid", True) else "GENERATION_ERROR"

            raw_issues = data.get("issues", [])
            if isinstance(raw_issues, list):
                cleaned_issues = [
                    clean_unreadable_characters(strip_ansi_escapes(str(i)))
                    for i in raw_issues
                    if i
                ]
                data["issues"] = cleaned_issues

            if raw_err_type != ValidationErrorType.VALID:
                data["valid"] = False
            elif not data.get("valid", True):
                raw_err_type = ValidationErrorType.GENERATION_ERROR

            data["error_type"] = raw_err_type

            raw_suggestion = data.get("suggestion")
            if raw_suggestion:
                data["suggestion"] = clean_unreadable_characters(strip_ansi_escapes(str(raw_suggestion)))

            return ValidationResult.model_validate(data)

        except Exception as err:
            logger.warning("Mongo validation SLM call failed: %s. Lenient fallback: accepting query.", err)
            return ValidationResult(
                valid=True,
                error_type=ValidationErrorType.VALID,
                issues=[],
                suggestion=None,
            )
