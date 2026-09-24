"""Tests for SemanticAnalyzer stage."""

import pytest
from query_processing.pipeline.semantic import SemanticAnalyzer


class MockSemanticProvider:
    async def generate(self, prompt: str, **kwargs) -> str:
        return """
        <think>Analyzing question</think>
        ```json
        {
          "subjective": ["customers", "orders"],
          "objective": ["number", "placed in last month"]
        }
        ```
        """


@pytest.mark.asyncio
async def test_semantic_analyzer_success():
    analyzer = SemanticAnalyzer(provider=MockSemanticProvider())
    res = await analyzer.analyze("How many orders did each customer place in the last month?")
    assert "customers" in res.subjective
    assert "orders" in res.subjective
    assert "number" in res.objective
