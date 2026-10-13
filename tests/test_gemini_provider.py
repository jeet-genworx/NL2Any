"""Unit tests for GeminiProvider."""

from unittest.mock import AsyncMock, MagicMock
from google.genai import types
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


def test_direct_mode_builds_a_plain_google_client(monkeypatch):
    """No base URL means Google, reached the way the SDK reaches it by default."""
    monkeypatch.setattr(settings, "gemini_api_key", "AIza-test-key")
    monkeypatch.setattr(settings, "gemini_model", "gemini-2.5-flash")
    monkeypatch.setattr(settings, "gemini_base_url", None)

    provider = GeminiProvider()
    assert provider.base_url is None
    client = provider._get_client()
    # The SDK fills in Google's own endpoint when nothing overrides it.
    assert "generativelanguage.googleapis.com" in client._api_client._http_options.base_url


def test_a_base_url_points_the_sdk_at_the_gateway(monkeypatch):
    monkeypatch.setattr(settings, "gemini_api_key", "sk-litellm-virtual-key")
    monkeypatch.setattr(settings, "gemini_model", "gemini/gemini-3.8-flash")
    monkeypatch.setattr(settings, "gemini_base_url", "https://litellm.example.com")

    provider = GeminiProvider()
    assert provider.base_url == "https://litellm.example.com"
    client = provider._get_client()
    assert client._api_client._http_options.base_url == "https://litellm.example.com"


def test_a_trailing_slash_is_dropped(monkeypatch):
    """The SDK joins its own paths on, and a doubled separator 404s on some gateways."""
    monkeypatch.setattr(settings, "gemini_api_key", "sk-key")
    monkeypatch.setattr(settings, "gemini_model", "gemini/gemini-3.8-flash")
    monkeypatch.setattr(settings, "gemini_base_url", "https://litellm.example.com/")

    assert GeminiProvider().base_url == "https://litellm.example.com"


def test_a_blank_base_url_is_treated_as_unset(monkeypatch):
    """An env var left empty must not be read as a gateway address."""
    monkeypatch.setattr(settings, "gemini_api_key", "AIza-test-key")
    monkeypatch.setattr(settings, "gemini_model", "gemini-2.5-flash")
    monkeypatch.setattr(settings, "gemini_base_url", "   ")

    assert GeminiProvider().base_url is None


def test_the_model_name_is_passed_through_untouched(monkeypatch):
    """A gateway publishes its own aliases, so nothing is prefixed on its behalf."""
    monkeypatch.setattr(settings, "gemini_api_key", "sk-key")
    monkeypatch.setattr(settings, "gemini_base_url", "https://litellm.example.com")

    monkeypatch.setattr(settings, "gemini_model", "gemini/gemini-3.8-flash")
    assert GeminiProvider().model == "gemini/gemini-3.8-flash"
    # An alias with no provider prefix is just as valid and stays as written.
    monkeypatch.setattr(settings, "gemini_model", "house-flash")
    assert GeminiProvider().model == "house-flash"


def test_the_missing_key_error_names_the_gateway(monkeypatch):
    """Which credential is missing differs by mode: a virtual key, not a Google one."""
    monkeypatch.setattr(settings, "gemini_api_key", "")
    monkeypatch.setattr(settings, "gemini_model", "gemini/gemini-3.8-flash")
    monkeypatch.setattr(settings, "gemini_base_url", "https://litellm.example.com")

    with pytest.raises(ValueError, match="gateway at https://litellm.example.com"):
        GeminiProvider()


def test_an_explicit_base_url_argument_beats_the_setting(monkeypatch):
    monkeypatch.setattr(settings, "gemini_api_key", "sk-key")
    monkeypatch.setattr(settings, "gemini_model", "gemini/gemini-3.8-flash")
    monkeypatch.setattr(settings, "gemini_base_url", "https://configured.example.com")

    provider = GeminiProvider(base_url="https://explicit.example.com")
    assert provider.base_url == "https://explicit.example.com"


def _reply(text, *, finish_reason=None, thoughts=0, answer=0):
    """A stand-in generate_content response, with the fields the provider reads."""
    response = MagicMock()
    response.text = text
    if finish_reason is None:
        response.candidates = []
    else:
        candidate = MagicMock()
        candidate.finish_reason = finish_reason
        response.candidates = [candidate]
    response.usage_metadata = MagicMock(
        thoughts_token_count=thoughts, candidates_token_count=answer
    )
    return response


def _provider(mock_client, **kwargs):
    return GeminiProvider(
        api_key="test-key", model="gemini/gemini-3.5-flash", client=mock_client, **kwargs
    )


@pytest.mark.asyncio
async def test_thinking_is_disabled_by_default_so_the_budget_buys_answer():
    """Reasoning tokens come out of max_output_tokens, which is sized for an answer."""
    mock_client = MagicMock()
    mock_client.aio.models.generate_content = AsyncMock(return_value=_reply("SELECT 1;"))

    provider = _provider(mock_client)
    assert provider.disable_thinking is True
    await provider.generate("Write SQL", max_tokens=1500)

    config = mock_client.aio.models.generate_content.await_args.kwargs["config"]
    assert config.thinking_config.thinking_budget == 0


@pytest.mark.asyncio
async def test_thinking_can_be_left_on():
    mock_client = MagicMock()
    mock_client.aio.models.generate_content = AsyncMock(return_value=_reply("SELECT 1;"))

    await _provider(mock_client, disable_thinking=False).generate("Write SQL")

    config = mock_client.aio.models.generate_content.await_args.kwargs["config"]
    assert config.thinking_config is None


@pytest.mark.asyncio
async def test_a_model_that_cannot_stop_thinking_is_retried_with_it_on():
    """Pro models reject a zero budget outright; the answer still has to arrive."""
    refusal = APIError(
        code=400, response_json={"error": {"message": "Budget 0 is invalid for this model"}}
    )
    mock_client = MagicMock()
    mock_client.aio.models.generate_content = AsyncMock(
        side_effect=[refusal, _reply("SELECT 1;")]
    )

    provider = _provider(mock_client)
    assert await provider.generate("Write SQL") == "SELECT 1;"

    calls = mock_client.aio.models.generate_content.await_args_list
    assert len(calls) == 2
    assert calls[0].kwargs["config"].thinking_config.thinking_budget == 0
    assert calls[1].kwargs["config"].thinking_config is None
    # Remembered, so the whole pipeline pays for the discovery once.
    assert provider._thinking_unsupported is True

    mock_client.aio.models.generate_content = AsyncMock(return_value=_reply("SELECT 2;"))
    await provider.generate("Write more SQL")
    assert mock_client.aio.models.generate_content.await_args.kwargs["config"].thinking_config is None


@pytest.mark.asyncio
async def test_an_unrelated_400_is_not_retried():
    """Only the thinking refusal earns a second call; other errors surface as they are."""
    mock_client = MagicMock()
    mock_client.aio.models.generate_content = AsyncMock(
        side_effect=APIError(code=400, response_json={"error": {"message": "API key not valid"}})
    )

    with pytest.raises(RuntimeError, match="Gemini API error"):
        await _provider(mock_client).generate("Write SQL")

    assert mock_client.aio.models.generate_content.await_count == 1


@pytest.mark.asyncio
async def test_a_truncated_answer_is_an_error_not_a_query():
    """Half-written SQL survives extraction and reaches validation looking like an answer."""
    mock_client = MagicMock()
    mock_client.aio.models.generate_content = AsyncMock(
        return_value=_reply(
            "SELECT c.name, SUM(o.total) FROM customers c JOIN",
            finish_reason=types.FinishReason.MAX_TOKENS,
            thoughts=1441,
            answer=55,
        )
    )

    with pytest.raises(RuntimeError, match="1441 tokens went to reasoning"):
        await _provider(mock_client).generate("Write SQL", max_tokens=1500)


@pytest.mark.asyncio
async def test_truncation_without_thinking_says_to_raise_the_budget():
    mock_client = MagicMock()
    mock_client.aio.models.generate_content = AsyncMock(
        return_value=_reply(
            "SELECT", finish_reason=types.FinishReason.MAX_TOKENS, thoughts=0, answer=60
        )
    )

    with pytest.raises(RuntimeError, match="Raise MODEL_MAX_TOKENS"):
        await _provider(mock_client).generate("Write SQL", max_tokens=60)


@pytest.mark.asyncio
async def test_a_complete_answer_passes_the_truncation_check():
    mock_client = MagicMock()
    mock_client.aio.models.generate_content = AsyncMock(
        return_value=_reply(
            "SELECT 1;", finish_reason=types.FinishReason.STOP, thoughts=0, answer=12
        )
    )

    assert await _provider(mock_client).generate("Write SQL") == "SELECT 1;"
