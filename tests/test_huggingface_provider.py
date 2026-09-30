"""Unit tests for HuggingFaceProvider."""

from unittest.mock import AsyncMock, MagicMock
import httpx
import pytest

from backend.src.config import settings
from backend.src.control.providers.model.huggingface import HuggingFaceProvider


def test_huggingface_provider_missing_credentials(monkeypatch):
    monkeypatch.setattr(settings, "hf_api_key", "")
    monkeypatch.setattr(settings, "hf_model", "")

    with pytest.raises(ValueError, match="HF_API_KEY must be configured"):
        HuggingFaceProvider(api_key="", model="meta-llama/Meta-Llama-3-8B-Instruct")

    with pytest.raises(ValueError, match="HF_MODEL must be configured"):
        HuggingFaceProvider(api_key="valid-key", model="")


@pytest.mark.asyncio
async def test_huggingface_provider_generate_success():
    mock_choice = MagicMock()
    mock_choice.message.content = '{"status": "ok", "query": "SELECT * FROM customers;"}'
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]

    mock_client = MagicMock()
    mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

    provider = HuggingFaceProvider(
        api_key="test-hf-key",
        model="meta-llama/Meta-Llama-3-8B-Instruct",
        client=mock_client,
    )

    result = await provider.generate(
        "Generate SQL",
        temperature=0.2,
        max_tokens=500,
        system_prompt="You are a SQL generator.",
    )

    assert result == '{"status": "ok", "query": "SELECT * FROM customers;"}'

    # Verify parameters forwarded to client
    mock_client.chat.completions.create.assert_awaited_once_with(
        model="meta-llama/Meta-Llama-3-8B-Instruct",
        messages=[
            {"role": "system", "content": "You are a SQL generator."},
            {"role": "user", "content": "Generate SQL"},
        ],
        temperature=0.2,
        max_tokens=500,
    )


@pytest.mark.asyncio
async def test_huggingface_provider_empty_choices():
    mock_response = MagicMock()
    mock_response.choices = []
    mock_client = MagicMock()
    mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

    provider = HuggingFaceProvider(
        api_key="test-key",
        model="meta-llama/Meta-Llama-3-8B-Instruct",
        client=mock_client,
    )

    with pytest.raises(ValueError, match="no choices"):
        await provider.generate("Hi")


@pytest.mark.asyncio
async def test_huggingface_provider_connection_error():
    mock_client = MagicMock()
    mock_client.chat.completions.create = AsyncMock(
        side_effect=httpx.ConnectError("Connection refused")
    )

    provider = HuggingFaceProvider(
        api_key="test-key",
        model="meta-llama/Meta-Llama-3-8B-Instruct",
        client=mock_client,
    )

    with pytest.raises(ConnectionError, match="Failed to connect to Hugging Face API"):
        await provider.generate("Hi")


@pytest.mark.asyncio
async def test_huggingface_provider_timeout_error():
    mock_client = MagicMock()
    mock_client.chat.completions.create = AsyncMock(
        side_effect=httpx.TimeoutException("Timed out")
    )

    provider = HuggingFaceProvider(
        api_key="test-key",
        model="meta-llama/Meta-Llama-3-8B-Instruct",
        client=mock_client,
    )

    with pytest.raises(TimeoutError, match="timed out"):
        await provider.generate("Hi")
