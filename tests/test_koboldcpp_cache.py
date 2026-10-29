"""Unit tests for KoboldCppCacheManager and Stage 1 KV cache preloading."""

import asyncio
import hashlib
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from backend.src.control.providers.model.koboldcpp_cache import (
    KoboldCppCacheManager,
    get_cache_manager,
)
from backend.src.data.models.targets import DatabaseTarget
from backend.src.data.models.schema import DatabaseType, DatabaseSchema, SchemaObject, Field, SchemaObjectKind
from backend.src.schemas.pipeline import TableSelectionResult


def create_dummy_schema(name: str) -> DatabaseSchema:
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
                    Field(name="email", type="text"),
                ],
            )
        ],
    )


def test_lru_slot_allocation():
    mgr = KoboldCppCacheManager(max_slots=3)

    # 1. Fill 3 slots: 0, 1, 2
    s0 = mgr.get_or_allocate_slot("db1")
    s1 = mgr.get_or_allocate_slot("db2")
    s2 = mgr.get_or_allocate_slot("db3")
    assert [s0, s1, s2] == [0, 1, 2]

    # 2. Access db1 to make db2 the LRU
    mgr.get_or_allocate_slot("db1")

    # 3. Allocate db4: should evict db2 (LRU) and take slot 1
    s3 = mgr.get_or_allocate_slot("db4")
    assert s3 == 1
    assert "db2" not in mgr._db_to_slot
    assert mgr._db_to_slot["db4"] == 1


def test_explicit_slot_mapping():
    mgr = KoboldCppCacheManager(max_slots=5, slot_mapping={"finops": 4})
    slot = mgr.get_or_allocate_slot("finops")
    assert slot == 4
    # Auto-allocation should not take slot 4 if free slots has remaining
    s0 = mgr.get_or_allocate_slot("postgres")
    assert s0 == 0


@pytest.mark.asyncio
async def test_cache_manager_save_and_load_state():
    mock_client = AsyncMock()

    # Success responses
    mock_client.post.side_effect = [
        MagicMock(status_code=200, json=lambda: {"success": True, "new_state_size": 1000}),
        MagicMock(status_code=200, json=lambda: {"success": True, "new_tokens": 50}),
    ]

    mgr = KoboldCppCacheManager(http_client=mock_client)
    saved = await mgr.save_state(0)
    assert saved is True
    assert 0 in mgr._warmed_slots

    loaded = await mgr.load_state(0)
    assert loaded is True


@pytest.mark.asyncio
async def test_cache_manager_load_state_failure():
    mock_client = AsyncMock()
    mock_client.post.return_value = MagicMock(
        status_code=200,
        json=lambda: {"success": False, "new_tokens": 0},
    )

    mgr = KoboldCppCacheManager(http_client=mock_client)
    loaded = await mgr.load_state(1)
    assert loaded is False


@pytest.mark.asyncio
async def test_single_flight_warmup():
    mgr = KoboldCppCacheManager()
    dummy_target = DatabaseTarget(
        key="test_db",
        label="Test DB",
        db_type=DatabaseType.POSTGRESQL,
        file_stem="test_db",
    )
    dummy_schema = create_dummy_schema("test_db")

    mock_selector = MagicMock()
    mock_selector.build_static_prefix.return_value = "Prefix...\n\n"
    mock_selector.provider.generate = AsyncMock(return_value="{}")

    with patch.object(mgr, "save_state", AsyncMock(return_value=True)), \
         patch.object(mgr, "check_context_capacity", AsyncMock(return_value=True)):

        # Trigger twice simultaneously
        task1 = mgr.trigger_warmup(dummy_target, mock_selector, dummy_schema)
        task2 = mgr.trigger_warmup(dummy_target, mock_selector, dummy_schema)

        # Both should be the exact same task
        assert task1 is task2

        res = await task1
        assert res is True
        assert mgr.is_warmed("test_db")


@pytest.mark.asyncio
async def test_table_selector_fallback_when_cache_fails():
    from backend.src.core.query_processing.pipeline.selector import TableSelector
    from backend.src.control.providers.model.koboldcpp import KoboldCppProvider

    mock_provider = MagicMock(spec=KoboldCppProvider)
    mock_provider.generate = AsyncMock(
        return_value=json.dumps({
            "selected_tables": ["users"],
            "sufficient": True,
            "missing_objects": [],
            "reason": "OK",
        })
    )

    mgr = KoboldCppCacheManager()
    # Mock load_state to fail
    mgr.load_state = AsyncMock(return_value=False)
    mgr.save_state = AsyncMock(return_value=False)

    selector = TableSelector(provider=mock_provider, cache_manager=mgr)
    schema = create_dummy_schema("users_db")

    res = await selector.select(
        question="Find user email",
        schema=schema,
        target_key="users_db",
    )

    assert res.selected_objects == ["users"]
    assert res.sufficient is True
    # Model should still have been called despite cache failure
    assert mock_provider.generate.call_count >= 1


@pytest.mark.asyncio
async def test_stages_2_and_3_do_not_touch_admin_endpoints():
    """Verify that Stage 2 (Generator) and Stage 3 (Validator) never call admin endpoints."""
    from backend.src.core.query_processing.pipeline.generators.postgres import PostgresQueryGenerator
    from backend.src.core.query_processing.pipeline.validator import QueryValidator
    from backend.src.schemas.pipeline import RelevantSchema

    called_urls: list[str] = []

    class SpyingClient:
        async def post(self, url, **kwargs):
            called_urls.append(str(url))
            mock_resp = MagicMock(status_code=200)
            mock_resp.json = lambda: {
                "choices": [{
                    "message": {
                        "content": json.dumps({
                            "sql": "SELECT 1;",
                            "explanation": "test",
                            "is_valid": True,
                            "issues": [],
                        })
                    }
                }]
            }
            return mock_resp

        async def get(self, url, **kwargs):
            called_urls.append(str(url))
            mock_resp = MagicMock(status_code=200)
            mock_resp.json = lambda: {}
            return mock_resp

    from backend.src.control.providers.model.koboldcpp import KoboldCppProvider
    provider = KoboldCppProvider(http_client=SpyingClient())

    gen = PostgresQueryGenerator(provider=provider)
    val = QueryValidator(provider=provider)

    dummy_rel_schema = RelevantSchema(objects=[], relationships=[])

    # Run generator
    try:
        await gen.generate_query("dummy question", dummy_rel_schema)
    except Exception:
        pass

    # Run validator
    try:
        await val.validate_query("dummy question", "SELECT 1;", dummy_rel_schema)
    except Exception:
        pass

    # Assert no admin or extra URLs were called by stage 2 or 3
    for url in called_urls:
        assert "/api/admin/" not in url
        assert "/api/extra/" not in url

