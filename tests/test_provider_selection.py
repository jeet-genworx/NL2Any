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


def _stage_providers(orchestrator, **kwargs):
    """The provider instance each stage ended up bound to."""
    guardrail, semantic, selector, planner, postgres_gen, mongo_gen, validator = (
        orchestrator._get_stages_for_provider(**kwargs)
    )
    return {
        "guardrail": guardrail.provider,
        "semantic": semantic.provider,
        "selector": selector.provider,
        "planner": planner.provider,
        "postgres_gen": postgres_gen.provider,
        "mongo_gen": mongo_gen.provider,
        "validator": validator.provider,
    }


def test_stage_providers_default_to_the_active_one():
    from backend.src.core.query_processing.pipeline.orchestrator import NL2AnyQueryOrchestrator

    orchestrator = NL2AnyQueryOrchestrator()
    active = KoboldCppProvider()

    bound = _stage_providers(orchestrator, active_provider=active)
    assert set(bound.values()) == {active}


def test_selection_and_generation_bind_to_their_own_providers(monkeypatch):
    monkeypatch.setattr(settings, "gemini_api_key", "test-key")
    monkeypatch.setattr(settings, "gemini_model", "gemini-2.5-flash")
    monkeypatch.setattr(settings, "hf_api_key", "test-key")
    monkeypatch.setattr(settings, "hf_model", "meta-llama/Meta-Llama-3-8B-Instruct")

    from backend.src.core.query_processing.pipeline.orchestrator import NL2AnyQueryOrchestrator

    orchestrator = NL2AnyQueryOrchestrator()
    active = KoboldCppProvider()
    sel = GeminiProvider()
    gen = HuggingFaceProvider()

    bound = _stage_providers(
        orchestrator,
        active_provider=active,
        selector_provider=sel,
        generator_provider=gen,
    )

    # The stages that read the question stay on the default provider.
    assert bound["guardrail"] is active
    assert bound["semantic"] is active
    assert bound["selector"] is sel
    # Planning and validation ride with generation, since they feed back into it.
    for stage in ("planner", "postgres_gen", "mongo_gen", "validator"):
        assert bound[stage] is gen


def test_one_stage_override_leaves_the_rest_on_the_default(monkeypatch):
    monkeypatch.setattr(settings, "gemini_api_key", "test-key")
    monkeypatch.setattr(settings, "gemini_model", "gemini-2.5-flash")

    from backend.src.core.query_processing.pipeline.orchestrator import NL2AnyQueryOrchestrator

    orchestrator = NL2AnyQueryOrchestrator()
    active = KoboldCppProvider()
    sel = GeminiProvider()

    bound = _stage_providers(
        orchestrator, active_provider=active, selector_provider=sel
    )
    assert bound["selector"] is sel
    assert bound["planner"] is active
    assert bound["guardrail"] is active


def test_injected_stages_win_over_a_stage_provider(monkeypatch):
    """A stage passed to the constructor is a deliberate override."""
    monkeypatch.setattr(settings, "gemini_api_key", "test-key")
    monkeypatch.setattr(settings, "gemini_model", "gemini-2.5-flash")

    from backend.src.core.query_processing.pipeline.orchestrator import NL2AnyQueryOrchestrator
    from tests.test_orchestrator import MockPlanner, MockSelector

    injected_selector = MockSelector()
    injected_planner = MockPlanner()
    orchestrator = NL2AnyQueryOrchestrator(
        selector=injected_selector, planner=injected_planner
    )

    _g, _s, selector, planner, *_rest = orchestrator._get_stages_for_provider(
        KoboldCppProvider(),
        selector_provider=GeminiProvider(),
        generator_provider=GeminiProvider(),
    )
    assert selector is injected_selector
    assert planner is injected_planner


def test_resolve_provider_passes_through_none_for_unset_stages():
    from backend.src.core.query_processing.pipeline.orchestrator import NL2AnyQueryOrchestrator

    orchestrator = NL2AnyQueryOrchestrator()
    assert orchestrator._resolve_provider(None) == (None, None)
    # A blank string is what an unset frontend picker degrades to.
    assert orchestrator._resolve_provider("") == (None, None)

    name, instance = orchestrator._resolve_provider("slm")
    assert name == "slm"
    assert isinstance(instance, KoboldCppProvider)


@pytest.mark.asyncio
async def test_response_reports_the_provider_of_each_stage():
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

    resp = await orchestrator.execute_pipeline(
        question="Show all customers from Chicago",
        database="postgres",
        provider="koboldcpp",
        selector_provider="slm",
    )

    # `provider` keeps naming the default, so an older caller reads what it did.
    assert resp.provider == "koboldcpp"
    assert resp.providers == {
        "default": "koboldcpp",
        "selector": "slm",
        "generator": "koboldcpp",
    }


def test_api_accepts_per_stage_providers_and_rejects_unknown_ones():
    from fastapi.testclient import TestClient
    from backend.src.api.rest.app import app

    client = TestClient(app)
    response = client.post(
        "/query",
        json={
            "database": "postgres",
            "question": "Show all customers",
            "provider": "koboldcpp",
            "selector_provider": "nonsense",
        },
    )
    assert response.status_code == 422
