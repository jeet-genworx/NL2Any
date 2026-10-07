"""Tests for Guardrail classifier and deterministic basic handling."""

import pytest
from backend.src.schemas.pipeline import GuardrailDecision
from backend.src.core.query_processing.pipeline.guardrail import GuardrailClassifier


class MockModelProvider:
    def __init__(self, response: str) -> None:
        self.response = response

    async def generate(self, prompt: str, **kwargs) -> str:
        return self.response


@pytest.mark.asyncio
async def test_guardrail_read_query():
    provider = MockModelProvider('{"decision": "READ_QUERY", "reason": "Analytical question"}')
    guardrail = GuardrailClassifier(provider=provider)
    result = await guardrail.classify("Show me all customers from Chicago")
    assert result.decision == GuardrailDecision.READ_QUERY


@pytest.mark.asyncio
async def test_guardrail_basic_query():
    provider = MockModelProvider('{"decision": "BASIC", "reason": "Meta question"}')
    guardrail = GuardrailClassifier(provider=provider)
    result = await guardrail.classify("Who are you?")
    assert result.decision == GuardrailDecision.BASIC


@pytest.mark.asyncio
async def test_guardrail_reject_fast_check():
    # Destructive keywords should be rejected without even querying the model
    guardrail = GuardrailClassifier(provider=None)
    result = await guardrail.classify("Delete all customers from the database")
    assert result.decision == GuardrailDecision.REJECT

    result2 = await guardrail.classify("Drop table orders")
    assert result2.decision == GuardrailDecision.REJECT


def test_guardrail_deterministic_basic_handler():
    guardrail = GuardrailClassifier(provider=None)

    ans1 = guardrail.handle_basic_question("Who are you?", "postgresql", "shop_db")
    assert "NL2AnyQuery" in ans1

    ans2 = guardrail.handle_basic_question("What can you do?", "postgresql", "shop_db")
    assert "postgresql" in ans2.lower()

    ans3 = guardrail.handle_basic_question("What database are you connected to?", "mongodb", "shop_mongo")
    assert "MONGODB" in ans3
    assert "shop_mongo" in ans3


@pytest.mark.asyncio
async def test_misspelled_read_question_is_not_rejected():
    """A typo must never cost a user their query.

    The guardrail runs before the spelling corrector, so it sees raw, misspelled
    text. It once stretched "perform actions outside read-only analytics" into
    rejecting anything it found ambiguous, which killed read questions whose
    only fault was a typo.
    """
    provider = MockModelProvider(
        '{"decision": "READ_QUERY", "reason": "Read-only question with typos"}'
    )
    guardrail = GuardrailClassifier(provider=provider)
    result = await guardrail.classify("Show me desperencies with custmer emails")
    assert result.decision == GuardrailDecision.READ_QUERY


def test_guardrail_prompt_forbids_rejecting_on_spelling_or_ambiguity():
    """Pins the instructions that keep typos and unknown jargon out of REJECT."""
    guardrail = GuardrailClassifier(provider=None)
    prompt = guardrail._load_prompt_template().lower()

    assert "not grounds for rejection" in prompt
    for topic in ("misspelling", "typo", "jargon", "ambiguous"):
        assert topic in prompt, f"guardrail prompt no longer mentions {topic!r}"

    # The template must still render, with its JSON block intact.
    rendered = guardrail._load_prompt_template().format(question="show me desperencies")
    assert '"decision"' in rendered
    assert "show me desperencies" in rendered
