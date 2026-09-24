"""Tests for KoboldCppEmbeddingProvider."""

import httpx
import pytest
from unittest.mock import AsyncMock, MagicMock

from query_processing.providers.embedding.koboldcpp import KoboldCppEmbeddingProvider


@pytest.mark.asyncio
async def test_embedding_provider_success():
    mock_vec = [0.1] * 384
    mock_response = MagicMock(spec=httpx.Response)
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "object": "list",
        "data": [{"object": "embedding", "index": 0, "embedding": mock_vec}],
    }

    mock_client = AsyncMock(spec=httpx.AsyncClient)
    mock_client.post.return_value = mock_response

    provider = KoboldCppEmbeddingProvider(client=mock_client)
    result = await provider.embed("Show customers")

    assert len(result) == 384
    assert result == mock_vec
    mock_client.post.assert_called_once()


@pytest.mark.asyncio
async def test_embedding_provider_http_error():
    mock_client = AsyncMock(spec=httpx.AsyncClient)
    mock_client.post.side_effect = httpx.ConnectError("Connection refused")

    provider = KoboldCppEmbeddingProvider(client=mock_client)
    with pytest.raises(ConnectionError, match="KoboldCpp embedding request failed"):
        await provider.embed("Show customers")


@pytest.mark.asyncio
async def test_embedding_provider_invalid_response_format():
    mock_response = MagicMock(spec=httpx.Response)
    mock_response.status_code = 200
    mock_response.json.return_value = {"error": "bad request"}  # missing 'data'

    mock_client = AsyncMock(spec=httpx.AsyncClient)
    mock_client.post.return_value = mock_response

    provider = KoboldCppEmbeddingProvider(client=mock_client)
    with pytest.raises(ValueError, match="Invalid embedding response"):
        await provider.embed("Show customers")


@pytest.mark.asyncio
async def test_embedding_provider_wrong_dimension():
    # Only 128 dimensions instead of 384
    mock_vec = [0.1] * 128
    mock_response = MagicMock(spec=httpx.Response)
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "data": [{"embedding": mock_vec}]
    }

    mock_client = AsyncMock(spec=httpx.AsyncClient)
    mock_client.post.return_value = mock_response

    provider = KoboldCppEmbeddingProvider(client=mock_client, expected_dim=384)
    with pytest.raises(ValueError, match="Unexpected embedding dimension: got 128, expected 384"):
        await provider.embed("Show customers")


@pytest.mark.asyncio
async def test_embedding_provider_non_numeric_vector():
    mock_vec = [0.1] * 383 + ["invalid_string"]
    mock_response = MagicMock(spec=httpx.Response)
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "data": [{"embedding": mock_vec}]
    }

    mock_client = AsyncMock(spec=httpx.AsyncClient)
    mock_client.post.return_value = mock_response

    provider = KoboldCppEmbeddingProvider(client=mock_client)
    with pytest.raises(ValueError, match="non-numeric"):
        await provider.embed("Show customers")


@pytest.mark.asyncio
async def test_embedding_provider_empty_input():
    provider = KoboldCppEmbeddingProvider()
    with pytest.raises(ValueError, match="cannot be empty"):
        await provider.embed("   ")
