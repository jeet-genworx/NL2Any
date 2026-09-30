"""Unit tests for provider resolution and factory."""

import pytest

from backend.src.config import settings
from backend.src.control.providers.model.factory import resolve_model_provider
from backend.src.control.providers.model.gemini import GeminiProvider
from backend.src.control.providers.model.huggingface import HuggingFaceProvider
from backend.src.control.providers.model.koboldcpp import KoboldCppProvider


def test_resolve_koboldcpp():
    provider = resolve_model_provider("koboldcpp")
    assert isinstance(provider, KoboldCppProvider)

    slm_provider = resolve_model_provider("slm")
    assert isinstance(slm_provider, KoboldCppProvider)


def test_resolve_huggingface(monkeypatch):
    monkeypatch.setattr(settings, "hf_api_key", "test-key")
    monkeypatch.setattr(settings, "hf_model", "meta-llama/Meta-Llama-3-8B-Instruct")

    provider = resolve_model_provider("huggingface")
    assert isinstance(provider, HuggingFaceProvider)
    assert provider.model == "meta-llama/Meta-Llama-3-8B-Instruct"

    hf_provider = resolve_model_provider("hf")
    assert isinstance(hf_provider, HuggingFaceProvider)


def test_resolve_huggingface_missing_config(monkeypatch):
    monkeypatch.setattr(settings, "hf_api_key", "")
    monkeypatch.setattr(settings, "hf_model", "")

    with pytest.raises(ValueError, match="HF_API_KEY must be configured"):
        resolve_model_provider("huggingface")


def test_resolve_gemini(monkeypatch):
    monkeypatch.setattr(settings, "gemini_api_key", "test-key")
    monkeypatch.setattr(settings, "gemini_model", "gemini-2.5-flash")

    provider = resolve_model_provider("gemini")
    assert isinstance(provider, GeminiProvider)
    assert provider.model == "gemini-2.5-flash"


def test_resolve_gemini_missing_config(monkeypatch):
    monkeypatch.setattr(settings, "gemini_api_key", "")
    monkeypatch.setattr(settings, "gemini_model", "")

    with pytest.raises(ValueError, match="GEMINI_API_KEY must be configured"):
        resolve_model_provider("gemini")


def test_resolve_invalid_provider():
    with pytest.raises(ValueError, match="Unsupported LLM provider"):
        resolve_model_provider("unknown-provider")


@pytest.mark.asyncio
async def test_orchestrator_unconfigured_huggingface_fails_clearly(monkeypatch):
    monkeypatch.setattr(settings, "hf_api_key", "")
    monkeypatch.setattr(settings, "hf_model", "")

    from backend.src.core.query_processing.pipeline.orchestrator import NL2AnyQueryOrchestrator
    orchestrator = NL2AnyQueryOrchestrator()

    with pytest.raises(ValueError, match="HF_API_KEY must be configured"):
        await orchestrator.execute_pipeline(
            "Show all customers",
            database="postgres",
            provider="huggingface",
        )


@pytest.mark.asyncio
async def test_orchestrator_unconfigured_gemini_fails_clearly(monkeypatch):
    monkeypatch.setattr(settings, "gemini_api_key", "")
    monkeypatch.setattr(settings, "gemini_model", "")

    from backend.src.core.query_processing.pipeline.orchestrator import NL2AnyQueryOrchestrator
    orchestrator = NL2AnyQueryOrchestrator()

    with pytest.raises(ValueError, match="GEMINI_API_KEY must be configured"):
        await orchestrator.execute_pipeline(
            "Show all customers",
            database="postgres",
            provider="gemini",
        )


@pytest.mark.asyncio
async def test_orchestrator_spelling_checker_execution():
    from backend.src.core.query_processing.pipeline.orchestrator import NL2AnyQueryOrchestrator
    from tests.test_orchestrator import (
        MockEmbeddingProvider,
        MockExecutor,
        MockGuardrail,
        MockPlanner,
        MockPolicy,
        MockPostgresGen,
        MockSelector,
        MockSemantic,
        MockValidator,
        MockVectorRetriever,
    )

    orchestrator = NL2AnyQueryOrchestrator(
        guardrail=MockGuardrail(),
        semantic=MockSemantic(),
        selector=MockSelector(),
        planner=MockPlanner(),
        postgres_gen=MockPostgresGen(),
        validator=MockValidator(),
        policy=MockPolicy(),
        executor=MockExecutor(),
        embedding_provider=MockEmbeddingProvider(),
        vector_retriever=MockVectorRetriever(),
    )

    # Query with a typo in customrs and technical jargon
    resp = await orchestrator.execute_pipeline(
        question="Show all customrs from Chicago in PostgreSQL",
        database="postgres",
        provider="koboldcpp",
        jargons=["PostgreSQL"],
    )

    assert resp.question == "Show all customrs from Chicago in PostgreSQL"
    assert resp.provider == "koboldcpp"
    assert resp.spelling_correction is not None
    assert resp.spelling_correction.corrected_question == "Show all customers from Chicago in PostgreSQL"
    assert any(c.original == "customrs" and c.corrected == "customers" for c in resp.spelling_correction.corrections)


def test_api_unconfigured_provider_returns_400(monkeypatch):
    monkeypatch.setattr(settings, "hf_api_key", "")
    monkeypatch.setattr(settings, "hf_model", "")

    from fastapi.testclient import TestClient
    from backend.src.api.rest.app import app

    client = TestClient(app)
    response = client.post(
        "/query",
        json={
            "database": "postgres",
            "question": "Show all customers",
            "provider": "huggingface",
        },
    )
    assert response.status_code == 400
    assert "HF_API_KEY must be configured" in response.json()["detail"]
