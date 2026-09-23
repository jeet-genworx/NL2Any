"""Guardrail classification and narrow deterministic BASIC question handling."""

import logging
from pathlib import Path
from pydantic import BaseModel
from nl2anyquery.core.text_utils import extract_json_block
from nl2anyquery.models.pipeline import GuardrailDecision, GuardrailResult
from nl2anyquery.providers.model.base import ModelProvider
from nl2anyquery.providers.model.koboldcpp import KoboldCppProvider

logger = logging.getLogger(__name__)

DEFAULT_GUARDRAIL_PROMPT = Path("prompts/guardrail.txt")

DESTRUCTIVE_KEYWORDS = {
    "drop", "delete", "truncate", "update", "insert", "alter", "create",
    "grant", "revoke", "replace", "remove", "kill", "shutdown"
}


class GuardrailOutput(BaseModel):
    decision: GuardrailDecision
    reason: str = ""


class GuardrailClassifier:
    """Classifies user questions into READ_QUERY, BASIC, or REJECT."""

    def __init__(
        self,
        provider: ModelProvider | None = None,
        prompt_path: Path | str | None = None,
    ) -> None:
        self.provider = provider or KoboldCppProvider()
        self.prompt_path = Path(prompt_path or DEFAULT_GUARDRAIL_PROMPT)

    def _load_prompt_template(self) -> str:
        if self.prompt_path.exists():
            return self.prompt_path.read_text(encoding="utf-8")
        # Fallback if running from a subdirectory
        resolved = Path.cwd() / self.prompt_path
        if resolved.exists():
            return resolved.read_text(encoding="utf-8")
        raise FileNotFoundError(f"Guardrail prompt file not found at {self.prompt_path}")

    async def classify(self, question: str) -> GuardrailResult:
        """Classify input query into READ_QUERY, BASIC, or REJECT."""
        cleaned_question = question.strip().lower()

        # Deterministic fast check for obvious destructive keywords
        words = set(cleaned_question.split())
        if words.intersection(DESTRUCTIVE_KEYWORDS):
            return GuardrailResult(
                decision=GuardrailDecision.REJECT,
                reason="Question contains prohibited data modification or administrative keywords.",
            )

        template = self._load_prompt_template()
        prompt = template.format(question=question)

        try:
            raw_response = await self.provider.generate(
                prompt=prompt,
                temperature=0.0,
                max_tokens=600,
            )
            data = extract_json_block(raw_response)
            parsed = GuardrailOutput.model_validate(data)
            return GuardrailResult(decision=parsed.decision, reason=parsed.reason)
        except Exception as err:
            logger.warning("Guardrail SLM parsing failed: %s. Applying fallback.", err)
            # Safe fallback: if question asks for info/select/show, allow as READ_QUERY
            if any(w in cleaned_question for w in ("who", "what can you do", "what database")):
                return GuardrailResult(
                    decision=GuardrailDecision.BASIC,
                    reason="Matched basic query pattern in fallback.",
                )
            return GuardrailResult(
                decision=GuardrailDecision.READ_QUERY,
                reason="Defaulted to read-only query in fallback.",
            )

    def handle_basic_question(
        self,
        question: str,
        database_type: str,
        database_name: str,
    ) -> str:
        """Narrow, deterministic responses for meta and system queries."""
        q = question.lower()
        if "who are you" in q:
            return (
                "I am NL2AnyQuery, an AI assistant that translates natural language "
                "questions into safe, read-only database queries."
            )
        elif "what can you do" in q:
            return (
                f"I can analyze your {database_type} database ('{database_name}') schema, "
                "translate your natural language analytical questions into safe read-only queries, "
                "execute them, and display formatted results and previews."
            )
        elif "what database" in q or "connected" in q:
            return (
                f"Currently connected to {database_type.upper()} database: '{database_name}'."
            )
        else:
            return (
                "I am an NL2AnyQuery database assistant. Please ask an analytical, "
                f"read-only question about your {database_type} database."
            )
