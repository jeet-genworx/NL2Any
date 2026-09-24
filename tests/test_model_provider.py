"""Tests for ModelProvider abstraction and KoboldCppProvider with mocking."""

import json
import httpx
import pytest
from query_processing.providers.model.koboldcpp import KoboldCppProvider


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

