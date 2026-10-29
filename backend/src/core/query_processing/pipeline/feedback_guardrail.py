"""Feedback safety guardrail for human feedback before partial pipeline reruns."""

import logging
from pathlib import Path
from typing import Literal
from pydantic import BaseModel, Field

from backend.src.control.providers.model.base import ModelProvider
from backend.src.utils.text_utils import (
    clean_unreadable_characters,
    estimate_token_count,
    extract_json_block,
    strip_ansi_escapes,
)

logger = logging.getLogger(__name__)

MAX_FEEDBACK_LENGTH = 1000


def _get_feedback_guardrail_prompt_path() -> Path:
    p = Path("backend/src/core/query_processing/prompts/feedback_guardrail.txt")
    if p.exists():
        return p
    alt = Path(__file__).resolve().parents[1] / "prompts" / "feedback_guardrail.txt"
    if alt.exists():
        return alt
    return p


DEFAULT_FEEDBACK_GUARDRAIL_PROMPT = _get_feedback_guardrail_prompt_path()


class FeedbackGuardrailResult(BaseModel):
    decision: Literal["ACCEPT", "REJECT"]
    reason: str = ""
    prompt_tokens: int = 0


class FeedbackGuardrailClassifier:
    """Evaluates human feedback text for a pipeline stage against safety guidelines."""

    def __init__(
        self,
        provider: ModelProvider,
        prompt_path: Path | str | None = None,
        provider_name: str | None = None,
    ) -> None:
        self.provider = provider
        self.provider_name = provider_name or getattr(provider, "model", "custom")
        self.prompt_path = Path(prompt_path or DEFAULT_FEEDBACK_GUARDRAIL_PROMPT)
        self.last_prompt_tokens: int = 0

    def _load_prompt_template(self) -> str:
        if self.prompt_path.exists():
            return self.prompt_path.read_text(encoding="utf-8")
        resolved = Path.cwd() / self.prompt_path
        if resolved.exists():
            return resolved.read_text(encoding="utf-8")
        raise FileNotFoundError(f"Feedback guardrail prompt file not found at {self.prompt_path}")

    async def classify(self, feedback: str, stage: str = "general") -> FeedbackGuardrailResult:
        """Classify user feedback on a stage into ACCEPT or REJECT."""
        self.last_prompt_tokens = 0
        cleaned = clean_unreadable_characters(strip_ansi_escapes(feedback)).strip()

        if not cleaned:
            return FeedbackGuardrailResult(decision="ACCEPT", reason="Empty feedback.")

        # Structural check 1: Maximum length
        if len(cleaned) > MAX_FEEDBACK_LENGTH:
            return FeedbackGuardrailResult(
                decision="REJECT",
                reason=f"Feedback text exceeds maximum length of {MAX_FEEDBACK_LENGTH} characters (length: {len(cleaned)}).",
            )

        template = self._load_prompt_template()
        prompt = template.format(stage=stage, feedback=cleaned)
        self.last_prompt_tokens = estimate_token_count(prompt)

        try:
            raw_response = await self.provider.generate(
                prompt=prompt,
                temperature=0.0,
                max_tokens=300,
            )
            data = extract_json_block(raw_response)
            decision = str(data.get("decision", "ACCEPT")).upper().strip()
            if decision not in ("ACCEPT", "REJECT"):
                decision = "ACCEPT"
            reason = str(data.get("reason", ""))
            return FeedbackGuardrailResult(
                decision=decision,  # type: ignore[arg-type]
                reason=reason,
                prompt_tokens=self.last_prompt_tokens,
            )
        except Exception as err:
            err_msg = f"Feedback guardrail failed using provider '{self.provider_name}': {err}"
            logger.error(err_msg)
            raise RuntimeError(err_msg) from err
