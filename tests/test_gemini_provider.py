"""Unit tests for GeminiProvider."""

from unittest.mock import AsyncMock, MagicMock
from google.genai.errors import APIError
import pytest

from backend.src.config import settings
from backend.src.control.providers.model.gemini import GeminiProvider


def test_gemini_provider_missing_credentials(monkeypatch):
    monkeypatch.setattr(settings, "gemini_api_key", "")
    monkeypatch.setattr(settings, "gemini_model", "")

    with pytest.raises(ValueError, match="GEMINI_API_KEY must be configured"):
        GeminiProvider(api_key="", model="gemini-2.5-flash")

    with pytest.raises(ValueError, match="GEMINI_MODEL must be configured"):
        GeminiProvider(api_key="valid-key", model="")


@pytest.mark.asyncio
async def test_gemini_provider_generate_success():
    mock_response = MagicMock()
    mock_response.text = '{"decision": "READ_QUERY", "reason": "Analytical question"}'

    mock_client = MagicMock()
    mock_client.aio.models.generate_content = AsyncMock(return_value=mock_response)

    provider = GeminiProvider(
        api_key="test-gemini-key",
        model="gemini-2.5-flash",
        client=mock_client,
    )

    result = await provider.generate(
        "Who are customers?",
        temperature=0.0,
        max_tokens=300,
        system_prompt="You are a classifier.",
    )

    assert result == '{"decision": "READ_QUERY", "reason": "Analytical question"}'

    # Verify call to aio.models.generate_content
    mock_client.aio.models.generate_content.assert_awaited_once()
    call_kwargs = mock_client.aio.models.generate_content.await_args.kwargs
    assert call_kwargs["model"] == "gemini-2.5-flash"
    assert call_kwargs["contents"] == "Who are customers?"
    assert call_kwargs["config"].temperature == 0.0
    assert call_kwargs["config"].max_output_tokens == 300
    assert call_kwargs["config"].system_instruction == "You are a classifier."


@pytest.mark.asyncio
async def test_gemini_provider_api_error():
    mock_client = MagicMock()
    mock_client.aio.models.generate_content = AsyncMock(
        side_effect=APIError(code=400, response_json={"error": "Bad Request"})
    )

    provider = GeminiProvider(
        api_key="test-gemini-key",
        model="gemini-2.5-flash",
        client=mock_client,
    )

    with pytest.raises(RuntimeError, match="Gemini API error"):
        await provider.generate("Hi")


@pytest.mark.asyncio
async def test_gemini_provider_timeout_error():
    mock_client = MagicMock()
    mock_client.aio.models.generate_content = AsyncMock(
        side_effect=TimeoutError("Request timed out")
    )

    provider = GeminiProvider(
        api_key="test-gemini-key",
        model="gemini-2.5-flash",
        client=mock_client,
    )

    with pytest.raises(TimeoutError, match="timed out"):
        await provider.generate("Hi")
