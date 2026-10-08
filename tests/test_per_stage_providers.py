"""Tests for per-stage model provider selection and strict KV cache isolation.

Verifies:
1. Discovery via GET /providers
2. DB selection warm-up runs ONLY when Stage 1 is koboldcpp (0 admin calls for HF/Gemini)
3. Table selection with HuggingFace, Gemini, and KoboldCpp with strict assertions on /api/admin/* calls
4. Query generation and validation stages never touch cache endpoints
5. Mixed provider execution with stage_providers map in response
6. Fail-fast error handling with explicit stage and provider attribution
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
import pytest
from fastapi.testclient import TestClient

from backend.src.api.rest.app import app
from backend.src.config import settings
from backend.src.control.providers.model.base import ModelProvider
from backend.src.control.providers.model.gemini import GeminiProvider
from backend.src.control.providers.model.huggingface import HuggingFaceProvider
from backend.src.control.providers.model.koboldcpp import KoboldCppProvider
from backend.src.control.providers.model.koboldcpp_cache import KoboldCppCacheManager
from backend.src.core.query_processing.pipeline.orchestrator import NL2AnyQueryOrchestrator
from backend.src.core.query_processing.pipeline.selector import TableSelector
from backend.src.data.models.schema import (
    DatabaseSchema,
    DatabaseType,
    Field,
    SchemaObject,
    SchemaObjectKind,
)
from backend.src.schemas.pipeline import (
    GeneratedQuery,
    GuardrailDecision,
    GuardrailResult,
    PolicyResult,
    TableSelectionResult,
    ValidationErrorType,
    ValidationResult,
)
from backend.src.schemas.request import QueryRequest, StageProviders


def dummy_schema(name: str = "test_db") -> DatabaseSchema:
    return DatabaseSchema(
        database_type=DatabaseType.POSTGRESQL,
        database_name=name,
        objects=[
            SchemaObject(
                name="users",
                kind=SchemaObjectKind.TABLE,
                description="User table",
                fields=[
                    Field(name="id", type="integer"),
                    Field(name="name", type="text"),
                ],
            )
        ],
    )


class MockProvider(ModelProvider):
    def __init__(self, name: str, canned_responses: dict[str, str] | None = None):
        self._name = name
        self.canned_responses = canned_responses or {}
        self.calls: list[str] = []

    async def generate(
        self,
        prompt: str,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        system_prompt: str | None = None,
    ) -> str:
        self.calls.append(prompt)
        for key, resp in self.canned_responses.items():
            if key in prompt:
                return resp
        return '{"selected_tables": ["users"], "reason": "found"}'


def test_get_providers_endpoint():
    """Verify GET /providers endpoint reports available providers."""
    client = TestClient(app)
    response = client.get("/providers")
    assert response.status_code == 200
    providers = response.json()
    assert isinstance(providers, list)
    ids = [p["id"] for p in providers]
    assert "koboldcpp" in ids
    assert "huggingface" in ids
    assert "gemini" in ids


def test_db_selection_admin_calls_isolation():
    """Verify POST /databases/{db}/select triggers admin warm-up ONLY for koboldcpp.

    For huggingface and gemini, make ZERO requests to /api/admin/*.
    """
    client = TestClient(app)

    with patch(
        "backend.src.control.providers.model.koboldcpp_cache.KoboldCppCacheManager.trigger_warmup"
    ) as mock_warmup:
        # 1. HuggingFace: must be skipped with ZERO admin/warmup calls
        resp_hf = client.post(
            "/databases/postgres/select",
            params={"stage1_provider": "huggingface"},
        )
        assert resp_hf.status_code == 200
        data_hf = resp_hf.json()
        assert data_hf["status"] == "skipped"
        assert "only enabled for koboldcpp" in data_hf.get("reason", "")
        mock_warmup.assert_not_called()

        # 2. Gemini: must be skipped with ZERO admin/warmup calls
        resp_gemini = client.post(
            "/databases/postgres/select",
            params={"stage1_provider": "gemini"},
        )
        assert resp_gemini.status_code == 200
        data_gemini = resp_gemini.json()
        assert data_gemini["status"] == "skipped"
        assert "only enabled for koboldcpp" in data_gemini.get("reason", "")
        mock_warmup.assert_not_called()

        # 3. KoboldCpp: triggers warmup
        resp_kobold = client.post(
            "/databases/postgres/select",
            params={"stage1_provider": "koboldcpp"},
        )
        assert resp_kobold.status_code == 200
        data_kobold = resp_kobold.json()
        assert data_kobold["status"] == "warming"
        mock_warmup.assert_called_once()


@pytest.mark.asyncio
async def test_table_selector_cache_isolation_hf_gemini_kobold():
    """Assert admin calls each time: huggingface (0), gemini (0), koboldcpp (>0)."""
    schema = dummy_schema("cache_test_db")
    cache_mgr = KoboldCppCacheManager(max_slots=2)
    cache_mgr.save_state = AsyncMock(return_value=True)
    cache_mgr.load_state = AsyncMock(return_value=True)
    cache_mgr.get_perf = AsyncMock(return_value={"last_process_speed": 100.0})

    # Case 1: Stage 1 on HuggingFace
    hf_provider = MagicMock(spec=HuggingFaceProvider)
    hf_provider.generate = AsyncMock(return_value='{"selected_tables": ["users"]}')
    selector_hf = TableSelector(provider=hf_provider, cache_manager=cache_mgr)

    res_hf = await selector_hf.select("Show users", schema=schema)
    assert res_hf.selected_objects == ["users"]
    # Assert ZERO admin calls
    cache_mgr.save_state.assert_not_called()
    cache_mgr.load_state.assert_not_called()
    cache_mgr.get_perf.assert_not_called()

    # Case 2: Stage 1 on Gemini
    gemini_provider = MagicMock(spec=GeminiProvider)
    gemini_provider.generate = AsyncMock(return_value='{"selected_tables": ["users"]}')
    selector_gemini = TableSelector(provider=gemini_provider, cache_manager=cache_mgr)

    res_gemini = await selector_gemini.select("Show users", schema=schema)
    assert res_gemini.selected_objects == ["users"]
    # Assert ZERO admin calls
    cache_mgr.save_state.assert_not_called()
    cache_mgr.load_state.assert_not_called()
    cache_mgr.get_perf.assert_not_called()

    # Case 3: Stage 1 on KoboldCpp
    kobold_provider = MagicMock(spec=KoboldCppProvider)
    kobold_provider.generate = AsyncMock(return_value='{"selected_tables": ["users"]}')
    selector_kobold = TableSelector(provider=kobold_provider, cache_manager=cache_mgr)

    res_kobold = await selector_kobold.select("Show users", schema=schema)
    assert res_kobold.selected_objects == ["users"]
    # Assert admin calls WERE invoked
    assert cache_mgr.save_state.call_count >= 1
    assert cache_mgr.load_state.call_count >= 1
    assert cache_mgr.get_perf.call_count >= 1


@pytest.mark.asyncio
async def test_query_generation_and_validation_never_call_cache():
    """Verify that Stages 2 and 3 never make requests to admin cache endpoints."""
    cache_mgr = KoboldCppCacheManager(max_slots=2)
    cache_mgr.save_state = AsyncMock(return_value=True)
    cache_mgr.load_state = AsyncMock(return_value=True)
    cache_mgr.get_perf = AsyncMock(return_value={"last_process_speed": 100.0})

    # Table selection on HuggingFace, Gen on Kobold, Val on Kobold
    hf_provider = MagicMock(spec=HuggingFaceProvider)
    hf_provider.generate = AsyncMock(return_value='{"selected_tables": ["users"]}')

    kobold_provider = MagicMock(spec=KoboldCppProvider)
    kobold_provider.generate = AsyncMock(
        side_effect=[
            "SELECT * FROM users;",  # Generator
            '{"valid": true, "issues": []}',  # Validator
        ]
    )

    orchestrator = NL2AnyQueryOrchestrator(cache_manager=cache_mgr)
    orchestrator.get_or_load_schema = MagicMock(return_value=dummy_schema("test_db"))
    orchestrator.executor.execute = MagicMock(return_value=(["id", "name"], [{"id": 1, "name": "Alice"}]))

    # Mock guardrail classify to allow query
    with patch(
        "backend.src.core.query_processing.pipeline.guardrail.GuardrailClassifier.classify",
        new=AsyncMock(return_value=GuardrailResult(decision=GuardrailDecision.READ_QUERY)),
    ):
        resp = await orchestrator.execute_pipeline(
            "Show users",
            database="postgres",
            stages={
                "table_selection": "huggingface",
                "query_generation": "koboldcpp",
                "validation": "koboldcpp",
            },
        )

    # Cache manager should have ZERO admin calls across the entire pipeline
    cache_mgr.save_state.assert_not_called()
    cache_mgr.load_state.assert_not_called()
    cache_mgr.get_perf.assert_not_called()


@pytest.mark.asyncio
async def test_mixed_provider_execution_and_stage_providers_map():
    """Test full mixed-provider execution: HF for Stage 1, Gemini for Stage 2, Kobold for Stage 3."""
    mock_hf = MagicMock(spec=HuggingFaceProvider)
    mock_hf.generate = AsyncMock(return_value='{"selected_tables": ["users"]}')

    mock_gemini = MagicMock(spec=GeminiProvider)
    mock_gemini.generate = AsyncMock(return_value="SELECT * FROM users;")

    mock_kobold = MagicMock(spec=KoboldCppProvider)
    mock_kobold.generate = AsyncMock(return_value='{"valid": true, "issues": []}')

    orchestrator = NL2AnyQueryOrchestrator()
    orchestrator.get_or_load_schema = MagicMock(return_value=dummy_schema("test_db"))
    orchestrator.executor.execute = MagicMock(return_value=(["id", "name"], [{"id": 1, "name": "Alice"}]))

    with patch(
        "backend.src.core.query_processing.pipeline.orchestrator.resolve_model_provider",
        side_effect=lambda name: {
            "huggingface": mock_hf,
            "gemini": mock_gemini,
            "koboldcpp": mock_kobold,
        }[name],
    ):
        with patch(
            "backend.src.core.query_processing.pipeline.guardrail.GuardrailClassifier.classify",
            new=AsyncMock(return_value=GuardrailResult(decision=GuardrailDecision.READ_QUERY)),
        ):
            resp = await orchestrator.execute_pipeline(
                "Show users",
                database="postgres",
                stages=StageProviders(
                    table_selection="huggingface",
                    query_generation="gemini",
                    validation="koboldcpp",
                ),
            )

    assert resp.stage_providers == {
        "table_selection": "huggingface",
        "query_generation": "gemini",
        "validation": "koboldcpp",
    }
    assert resp.provider == "huggingface/gemini/koboldcpp"
    assert resp.error is None
    assert resp.results is not None
    assert resp.results.row_count == 1


@pytest.mark.asyncio
async def test_fail_fast_attribution_stage1():
    """Verify Stage 1 error clearly attributes Stage 1 and provider name."""
    mock_hf = MagicMock(spec=HuggingFaceProvider)
    mock_hf.generate = AsyncMock(side_effect=RuntimeError("HF Endpoint Connection Failed"))

    orchestrator = NL2AnyQueryOrchestrator()
    orchestrator.get_or_load_schema = MagicMock(return_value=dummy_schema("test_db"))

    with patch(
        "backend.src.core.query_processing.pipeline.orchestrator.resolve_model_provider",
        return_value=mock_hf,
    ):
        with patch(
            "backend.src.core.query_processing.pipeline.guardrail.GuardrailClassifier.classify",
            new=AsyncMock(return_value=GuardrailResult(decision=GuardrailDecision.READ_QUERY)),
        ):
            resp = await orchestrator.execute_pipeline(
                "Show users",
                database="postgres",
                stages={"table_selection": "huggingface", "query_generation": "koboldcpp", "validation": "koboldcpp"},
            )

    assert resp.error is not None
    assert "Stage 1 (Table Selection) failed using provider 'huggingface'" in resp.error
    assert "HF Endpoint Connection Failed" in resp.error


@pytest.mark.asyncio
async def test_fail_fast_attribution_stage2():
    """Verify Stage 2 error clearly attributes Stage 2 and provider name."""
    mock_gemini = MagicMock(spec=GeminiProvider)
    mock_gemini.generate = AsyncMock(side_effect=RuntimeError("Gemini 429 Resource Exhausted"))

    orchestrator = NL2AnyQueryOrchestrator()
    orchestrator.get_or_load_schema = MagicMock(return_value=dummy_schema("test_db"))

    # Table selection succeeds
    orchestrator.selector.select = AsyncMock(
        return_value=TableSelectionResult(selected_objects=["users"], sufficient=True)
    )

    with patch(
        "backend.src.core.query_processing.pipeline.orchestrator.resolve_model_provider",
        side_effect=lambda name: mock_gemini if name == "gemini" else KoboldCppProvider(),
    ):
        with patch(
            "backend.src.core.query_processing.pipeline.guardrail.GuardrailClassifier.classify",
            new=AsyncMock(return_value=GuardrailResult(decision=GuardrailDecision.READ_QUERY)),
        ):
            resp = await orchestrator.execute_pipeline(
                "Show users",
                database="postgres",
                stages={"table_selection": "koboldcpp", "query_generation": "gemini", "validation": "koboldcpp"},
            )

    assert resp.error is not None
    assert "Stage 2 (Query Generation) failed using provider 'gemini'" in resp.error
    assert "Gemini 429 Resource Exhausted" in resp.error


@pytest.mark.asyncio
async def test_fail_fast_attribution_stage3():
    """Verify Stage 3 error clearly attributes Stage 3 and provider name."""
    mock_hf = MagicMock(spec=HuggingFaceProvider)
    mock_hf.generate = AsyncMock(side_effect=RuntimeError("Validation API Timeout"))

    orchestrator = NL2AnyQueryOrchestrator()
    orchestrator.get_or_load_schema = MagicMock(return_value=dummy_schema("test_db"))
    orchestrator.selector.select = AsyncMock(
        return_value=TableSelectionResult(selected_objects=["users"], sufficient=True)
    )
    orchestrator.postgres_gen.generate_query = AsyncMock(
        return_value=GeneratedQuery(
            database_type=DatabaseType.POSTGRESQL,
            raw_query="SELECT * FROM users;",
            formatted_query="SELECT * FROM users;",
        )
    )

    with patch(
        "backend.src.core.query_processing.pipeline.orchestrator.resolve_model_provider",
        side_effect=lambda name: mock_hf if name == "huggingface" else KoboldCppProvider(),
    ):
        with patch(
            "backend.src.core.query_processing.pipeline.guardrail.GuardrailClassifier.classify",
            new=AsyncMock(return_value=GuardrailResult(decision=GuardrailDecision.READ_QUERY)),
        ):
            resp = await orchestrator.execute_pipeline(
                "Show users",
                database="postgres",
                stages={"table_selection": "koboldcpp", "query_generation": "koboldcpp", "validation": "huggingface"},
            )

    assert resp.error is not None
    assert "Stage 3 (Validation) failed using provider 'huggingface'" in resp.error
    assert "Validation API Timeout" in resp.error


@pytest.mark.asyncio
async def test_unconfigured_stage_provider_raises_clear_stage_error(monkeypatch):
    """Verify unconfigured provider raises ValueError attributing the failing stage."""
    monkeypatch.setattr(settings, "gemini_api_key", "")
    monkeypatch.setattr(settings, "gemini_model", "")

    orchestrator = NL2AnyQueryOrchestrator()

    with pytest.raises(ValueError, match="Stage 2 \\(Query Generation\\) provider configuration error"):
        await orchestrator.execute_pipeline(
            "Show users",
            database="postgres",
            stages={"table_selection": "koboldcpp", "query_generation": "gemini", "validation": "koboldcpp"},
        )


@pytest.mark.asyncio
async def test_stage_tokens_tracked_in_pipeline_response():
    """Verify stage_tokens is populated with token counts across stages and in error states."""
    orchestrator = NL2AnyQueryOrchestrator()
    orchestrator.get_or_load_schema = MagicMock(return_value=dummy_schema("test_db"))
    orchestrator.guardrail.classify = AsyncMock(return_value=GuardrailResult(decision=GuardrailDecision.READ_QUERY))
    orchestrator.guardrail.last_prompt_tokens = 50
    orchestrator.selector.select = AsyncMock(
        return_value=TableSelectionResult(selected_objects=["users"], sufficient=True)
    )
    orchestrator.selector.last_prompt_tokens = 200
    orchestrator.postgres_gen.generate_query = AsyncMock(
        return_value=GeneratedQuery(
            database_type=DatabaseType.POSTGRESQL,
            raw_query="SELECT * FROM users;",
            formatted_query="SELECT * FROM users;",
        )
    )
    orchestrator.postgres_gen.last_prompt_tokens = 350
    orchestrator.validator.validate = AsyncMock(
        return_value=ValidationResult(valid=True, error_type=ValidationErrorType.VALID)
    )
    orchestrator.validator.postgres_validator.last_prompt_tokens = 150
    orchestrator.policy.check = MagicMock(return_value=PolicyResult(allowed=True))
    orchestrator.executor.execute = MagicMock(return_value=(["id", "name"], [{"id": 1, "name": "Alice"}]))

    resp = await orchestrator.execute_pipeline("Show users", database="postgres")

    assert resp.stage_tokens is not None
    assert resp.stage_tokens["guardrail"] == 50
    assert resp.stage_tokens["table_selection"] == 200
    assert resp.stage_tokens["stage_1"] == 250
    assert resp.stage_tokens["stage_2"] == 350
    assert resp.stage_tokens["stage_3"] == 150
    assert resp.stage_tokens["total"] == 750
