"""Tests for ModelProvider abstraction and KoboldCppProvider with mocking."""

import json
import httpx
import pytest
from query_processing.models.schema import (
    DatabaseSchema,
    DatabaseType,
    Field,
    SchemaObject,
    SchemaObjectKind,
)
from query_processing.providers.model.koboldcpp import KoboldCppProvider
from ingestion.schema.profiler import SchemaProfiler


@pytest.mark.asyncio
async def test_koboldcpp_provider_generate_success():
    def handler(request: httpx.Request) -> httpx.Response:
        data = json.loads(request.content)
        assert data["model"] == "test-model"
        assert len(data["messages"]) == 1
        return httpx.Response(
            status_code=200,
            json={
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "<think>thinking...</think> Hello from model",
                        }
                    }
                ]
            },
        )

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        provider = KoboldCppProvider(
            base_url="http://mock-kobold/v1",
            model="test-model",
            http_client=client,
        )
        response = await provider.generate("Hi")
        # KoboldCppProvider returns raw response as per requirement 2
        assert response == "<think>thinking...</think> Hello from model"


@pytest.mark.asyncio
async def test_koboldcpp_provider_connection_error():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("Connection refused")

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        provider = KoboldCppProvider(
            base_url="http://mock-kobold/v1",
            model="test-model",
            http_client=client,
        )
        with pytest.raises(ConnectionError, match="Failed to connect to KoboldCpp"):
            await provider.generate("Hi")


@pytest.mark.asyncio
async def test_schema_profiler_enrichment_with_mock_provider():
    class MockProvider:
        async def generate(self, prompt: str, **kwargs) -> str:
            return """
            <think>I should describe customers and orders</think>
            ```json
            {
              "objects": [
                {
                  "name": "customers",
                  "description": "User profiles and contact records.",
                  "fields": {
                    "id": "Unique customer ID",
                    "city": "Primary residence city"
                  }
                }
              ]
            }
            ```
            """

    schema = DatabaseSchema(
        database_type=DatabaseType.POSTGRESQL,
        database_name="test_db",
        objects=[
            SchemaObject(
                name="customers",
                kind=SchemaObjectKind.TABLE,
                description="",
                fields=[
                    Field(name="id", type="integer"),
                    Field(name="city", type="varchar"),
                ],
            )
        ],
    )

    profiler = SchemaProfiler(provider=MockProvider())
    enriched = await profiler.enrich_schema(schema)

    cust = enriched.get_object("customers")
    assert cust is not None
    assert cust.description == "User profiles and contact records."
    assert cust.get_field("id").description == "Unique customer ID"
    assert cust.get_field("city").description == "Primary residence city"
    # Authoritative types must not change
    assert cust.get_field("id").type == "integer"
