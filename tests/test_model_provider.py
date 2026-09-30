"""Tests for ModelProvider abstraction and KoboldCppProvider with mocking."""

import json
import httpx
import pytest
from backend.src.control.providers.model.koboldcpp import KoboldCppProvider
from backend.src.control.providers.model.embedding import KoboldCppEmbeddingProvider


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
async def test_koboldcpp_embedding_provider_embed_success():
    def handler(request: httpx.Request) -> httpx.Response:
        data = json.loads(request.content)
        assert data["model"] == "all-MiniLM-L6-v2-Q8_0"
        assert data["input"] == ["customers table", "orders table"]
        return httpx.Response(
            status_code=200,
            json={
                "data": [
                    {"embedding": [0.1, 0.2, 0.3], "index": 0},
                    {"embedding": [0.4, 0.5, 0.6], "index": 1},
                ]
            },
        )

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        provider = KoboldCppEmbeddingProvider(
            base_url="http://mock-kobold/v1",
            model="all-MiniLM-L6-v2-Q8_0",
            http_client=client,
        )
        vectors = await provider.embed(["customers table", "orders table"])
        assert vectors == [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]


@pytest.mark.asyncio
async def test_koboldcpp_embedding_provider_orders_by_index():
    def handler(request: httpx.Request) -> httpx.Response:
        # Server returns them out of order; provider must sort by index.
        return httpx.Response(
            status_code=200,
            json={
                "data": [
                    {"embedding": [1.0], "index": 1},
                    {"embedding": [0.0], "index": 0},
                ]
            },
        )

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        provider = KoboldCppEmbeddingProvider(
            base_url="http://mock-kobold/v1", model="test-embed", http_client=client
        )
        vectors = await provider.embed(["a", "b"])
        assert vectors == [[0.0], [1.0]]


@pytest.mark.asyncio
async def test_koboldcpp_embedding_provider_connection_error():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("Connection refused")

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        provider = KoboldCppEmbeddingProvider(
            base_url="http://mock-kobold/v1", model="test-embed", http_client=client
        )
        with pytest.raises(ConnectionError, match="Failed to connect to KoboldCpp"):
            await provider.embed(["a"])


def test_koboldcpp_provider_timeout_disabled(monkeypatch):
    from backend.src.config import settings
    monkeypatch.setattr(settings, "model_timeout_seconds", 0)

    provider = KoboldCppProvider()
    assert provider.timeout is None

    monkeypatch.setattr(settings, "model_timeout_seconds", None)
    provider_none = KoboldCppProvider()
    assert provider_none.timeout is None

    embed_provider = KoboldCppEmbeddingProvider()
    assert embed_provider.timeout is None



@pytest.mark.asyncio
async def test_generate_appends_no_think_directive_by_default():
    """The reasoning switch rides on the user turn so the model answers directly.

    Qwen3-4B decides per request whether to open a <think> block, and the
    reasoning is charged to the same completion budget as the answer; when it
    fires on a long structured task the JSON is cut off mid-string.
    """
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content))
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = KoboldCppProvider(model="m", http_client=client)
        await provider.generate("Describe the invoices table.")

    assert captured["messages"][-1]["content"] == "Describe the invoices table.\n\n/no_think"


@pytest.mark.asyncio
async def test_generate_leaves_prompt_untouched_when_thinking_allowed():
    """Opting out restores the bare prompt, for a model with no thinking mode."""
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content))
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = KoboldCppProvider(model="m", http_client=client, disable_thinking=False)
        await provider.generate("Describe the invoices table.")

    assert captured["messages"][-1]["content"] == "Describe the invoices table."


@pytest.mark.asyncio
async def test_no_think_directive_does_not_displace_the_system_prompt():
    """The directive attaches to the user turn, leaving any system turn intact."""
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content))
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = KoboldCppProvider(model="m", http_client=client)
        await provider.generate("Question?", system_prompt="You are terse.")

    assert captured["messages"][0] == {"role": "system", "content": "You are terse."}
    assert captured["messages"][1]["content"] == "Question?\n\n/no_think"
