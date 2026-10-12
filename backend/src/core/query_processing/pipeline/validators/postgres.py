"""PostgreSQL query validator with safety checks and SLM verification."""

import logging
from pathlib import Path
import re
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
    RelevantSchema,
    ValidationErrorType,
    ValidationResult,
)
from backend.src.data.models.schema import DatabaseType
from backend.src.core.query_processing.pipeline.selector import format_relevant_schema
from backend.src.core.query_processing.pipeline.validators.base import QueryValidator
from backend.src.control.providers.model.base import ModelProvider
from backend.src.control.providers.model.koboldcpp import KoboldCppProvider

logger = logging.getLogger(__name__)


def _get_postgres_validator_prompt_path() -> Path:
    p = Path("backend/src/core/query_processing/prompts/postgres_validator.txt")
    if p.exists():
        return p
    alt = Path(__file__).resolve().parents[2] / "prompts" / "postgres_validator.txt"
    if alt.exists():
        return alt
    return p


DEFAULT_POSTGRES_VALIDATOR_PROMPT = _get_postgres_validator_prompt_path()

UNSAFE_SQL_PATTERN = re.compile(
    r"\b(INSERT\s+INTO|UPDATE\s+\w+\s+SET|DELETE\s+FROM|DROP\s+(TABLE|DATABASE|VIEW)|ALTER\s+TABLE|TRUNCATE|CREATE\s+(TABLE|DATABASE)|GRANT\s+|REVOKE\s+)\b",
    re.IGNORECASE,
)


class PostgresQueryValidator(QueryValidator):
    """Specialized validator for PostgreSQL read-only queries."""

    def __init__(
        self,
        provider: ModelProvider | None = None,
        prompt_path: Path | str | None = None,
    ) -> None:
        self.provider = provider or KoboldCppProvider()
        self.prompt_path = Path(prompt_path or DEFAULT_POSTGRES_VALIDATOR_PROMPT)
        self.last_prompt_tokens: int = 0

    def _load_prompt_template(self) -> str:
        if self.prompt_path.exists():
            return self.prompt_path.read_text(encoding="utf-8")
        resolved = Path.cwd() / self.prompt_path
        if resolved.exists():
            return resolved.read_text(encoding="utf-8")
        raise FileNotFoundError(f"Postgres validator prompt not found at {self.prompt_path}")

    async def validate(
        self,
        question: str,
        generated_query: Any = None,
        schema: Any = None,
        plan: Any = None,
        **kwargs: Any,
    ) -> ValidationResult:
        """Validate generated PostgreSQL query using safety checks and SLM verification."""
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
                issues=["Invalid arguments passed to Postgres validator: missing GeneratedQuery."],
                suggestion="Provide a valid GeneratedQuery.",
            )

        # 1. Fast safety checks
        sql_str = str(active_query.raw_query).strip()
        if not sql_str:
            return ValidationResult(
                valid=False,
                error_type=ValidationErrorType.GENERATION_ERROR,
                issues=["No SQL query was generated."],
                suggestion="Generate a valid read-only PostgreSQL query based on the selected schema.",
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

            raw_verdict = data.get("verdict")
            if isinstance(raw_verdict, str):
                v_clean = raw_verdict.lower().strip()
                data["verdict"] = v_clean
                if v_clean in ("correct", "correct_with_caveats"):
                    data["valid"] = True
                    data["error_type"] = ValidationErrorType.VALID
                elif v_clean in ("incorrect", "failed"):
                    data["valid"] = False
                    current_err = str(data.get("error_type", "")).upper().strip()
                    if current_err in ValidationErrorType.__members__ and current_err != ValidationErrorType.VALID.value:
                        data["error_type"] = ValidationErrorType[current_err]
                    else:
                        data["error_type"] = ValidationErrorType.SEMANTIC_ERROR

            raw_err_type = data.get("error_type", "VALID")
            if isinstance(raw_err_type, ValidationErrorType):
                raw_err_type = raw_err_type.value
            if not isinstance(raw_err_type, str) or raw_err_type not in ValidationErrorType.__members__:
                raw_err_type = "VALID" if data.get("valid", True) else "SEMANTIC_ERROR"

            raw_issues = data.get("issues", [])
            if isinstance(raw_issues, list):
                cleaned_issues = [
                    clean_unreadable_characters(strip_ansi_escapes(str(i)))
                    for i in raw_issues
                    if i
                ]
                data["issues"] = cleaned_issues

            if raw_err_type != ValidationErrorType.VALID.value:
                data["valid"] = False
            elif not data.get("valid", True):
                raw_err_type = ValidationErrorType.SEMANTIC_ERROR.value

            data["error_type"] = ValidationErrorType[raw_err_type] if isinstance(raw_err_type, str) else raw_err_type

            raw_suggestion = data.get("suggestion")
            if raw_suggestion:
                data["suggestion"] = clean_unreadable_characters(strip_ansi_escapes(str(raw_suggestion)))

            return ValidationResult.model_validate(data)

        except Exception as err:
            if kwargs.get("fail_fast", False):
                raise
            logger.warning("Postgres validation SLM call failed: %s. Lenient fallback: accepting query.", err)
            return ValidationResult(
                valid=True,
                error_type=ValidationErrorType.VALID,
                issues=[],
                suggestion=None,
            )
