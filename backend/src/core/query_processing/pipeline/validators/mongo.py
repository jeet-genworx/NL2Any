"""MongoDB query validator with safety checks and dedicated SLM verification."""

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
        self.last_prompt_tokens: int = 0

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
        self.last_prompt_tokens = 0
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

            # Fast-path check for projecting fields that only exist inside a non-unwound lookup array
            lookup_sub_fields: dict[str, set[str]] = {}
            unwound_fields: set[str] = set()
            for idx, stage in enumerate(raw.pipeline):
                if isinstance(stage, dict):
                    if "$lookup" in stage and isinstance(stage["$lookup"], dict):
                        l_info = stage["$lookup"]
                        as_name = l_info.get("as")
                        sub_pipe = l_info.get("pipeline")
                        if as_name and isinstance(sub_pipe, list):
                            sub_fields = set()
                            for sub_stg in sub_pipe:
                                if isinstance(sub_stg, dict) and "$group" in sub_stg and isinstance(sub_stg["$group"], dict):
                                    for k_grp in sub_stg["$group"].keys():
                                        if k_grp != "_id":
                                            sub_fields.add(k_grp)
                            if sub_fields:
                                lookup_sub_fields[as_name] = sub_fields
                    elif "$unwind" in stage:
                        u_path = stage["$unwind"]
                        if isinstance(u_path, dict):
                            u_path = u_path.get("path", "")
                        if isinstance(u_path, str):
                            unwound_fields.add(u_path.lstrip("$"))
                    elif "$project" in stage and isinstance(stage["$project"], dict):
                        for as_name, sub_fields in lookup_sub_fields.items():
                            if as_name not in unwound_fields:
                                for proj_k, proj_v in stage["$project"].items():
                                    if proj_k in sub_fields and (proj_v == 1 or proj_v is True):
                                        return ValidationResult(
                                            valid=False,
                                            error_type=ValidationErrorType.GENERATION_ERROR,
                                            issues=[
                                                f"Stage {idx} ($project): field '{proj_k}' is nested inside lookup array '{as_name}' and does not exist at root level; it will be silently dropped by MongoDB. Use $size with $filter over '${as_name}' in $project, or unwind '${as_name}' first."
                                            ],
                                            suggestion=f"In $project, compute multiple counts using {{'$size': {{'$filter': {{'input': '${as_name}', 'as': 'item', 'cond': ...}}}}}} or unwind '${as_name}' before projecting.",
                                        )

        # 2. SLM Semantic & Relational Validation
        template = self._load_prompt_template()
        schema_context = format_relevant_schema(active_schema)

        prompt = template.format(
            question=question,
            schema_context=schema_context,
            query_context=active_query.formatted_query,
        )
        self.last_prompt_tokens = estimate_token_count(prompt)

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
                # Filter out obvious SLM hallucinations
                query_text = active_query.formatted_query or ""
                filtered_issues = []
                for i in raw_issues:
                    issue_str = str(i)
                    # 1) Whitespace / empty string treated as a stage operator
                    if "stage operator ' '" in issue_str or "stage operator ''" in issue_str or 'stage operator ""' in issue_str:
                        continue
                    # 2) $$ROOT hallucination when $$ROOT is not in the query at all
                    if "$$root" in issue_str.lower() and "$$root" not in query_text.lower():
                        continue
                    filtered_issues.append(issue_str)

                cleaned_issues = [
                    clean_unreadable_characters(strip_ansi_escapes(str(i)))
                    for i in filtered_issues
                    if i
                ]
                data["issues"] = cleaned_issues

                # If all issues were discarded hallucinations, mark query valid
                if not cleaned_issues and raw_err_type in (ValidationErrorType.SYNTAX_ERROR, ValidationErrorType.GENERATION_ERROR):
                    raw_err_type = ValidationErrorType.VALID
                    data["valid"] = True
                    data["suggestion"] = None

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
            if kwargs.get("fail_fast", False):
                raise
            logger.warning("Mongo validation SLM call failed: %s. Lenient fallback: accepting query.", err)
            return ValidationResult(
                valid=True,
                error_type=ValidationErrorType.VALID,
                issues=[],
                suggestion=None,
            )
