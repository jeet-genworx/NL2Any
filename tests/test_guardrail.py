"""Tests for Guardrail classifier and deterministic basic handling."""

import pytest
from query_processing.models.pipeline import GuardrailDecision
from query_processing.pipeline.guardrail import GuardrailClassifier


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
