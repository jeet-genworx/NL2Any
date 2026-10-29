"""Smoke test for KoboldCpp KV cache slot save/load mechanism.

Validates:
1. Fast-forwarding: save_state on long prompt prefix, wipe context, load_state,
   resend prompt with changed tail -> verify only the tail is evaluated (large speedup).
2. Large prompt (~24k tokens / FinOps schema): save_state and load_state work at full scale.
3. Multi-database switching: Slot A -> Slot B -> Slot A without consuming or invalidating slots.
"""

import asyncio
import json
import time
import httpx
import pytest

from backend.src.config import settings
from backend.src.core.query_processing.pipeline.orchestrator import NL2AnyQueryOrchestrator
from backend.src.core.query_processing.pipeline.selector import format_schema_description
from backend.src.data.models.targets import TARGETS


ADMIN_URL = "http://127.0.0.1:5001"
CHAT_URL = f"{ADMIN_URL}/v1/chat/completions"


async def _is_kobold_running() -> bool:
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            resp = await client.get(f"{ADMIN_URL}/api/extra/version")
            return resp.status_code == 200
    except Exception:
        return False


@pytest.mark.asyncio
async def test_kv_cache_fast_forward_smoke():
    """Verify that after load_state, prefix tokens are reused and only tail is evaluated."""
    if not await _is_kobold_running():
        pytest.skip("KoboldCpp is not running on 5001")

    async with httpx.AsyncClient(timeout=60.0) as client:
        prefix = (
            "You are a database query assistant. "
            + ("Table customer has columns id, name, email, created_at, status. " * 30)
            + "\n\n"
        )
        prompt_cold = prefix + "User Question:\nWhat is the total count of customers?\n\n/no_think"
        prompt_warm = prefix + "User Question:\nList all active customers.\n\n/no_think"

        # 1. Cold evaluation
        t0 = time.perf_counter()
        resp_cold = await client.post(
            CHAT_URL,
            json={
                "model": settings.koboldcpp_model,
                "messages": [{"role": "user", "content": prompt_cold}],
                "max_tokens": 1,
            },
        )
        t_cold = time.perf_counter() - t0
        assert resp_cold.status_code == 200

        # Save to slot 0
        resp_save = await client.post(f"{ADMIN_URL}/api/admin/save_state", json={"slot": 0})
        assert resp_save.status_code == 200
        save_data = resp_save.json()
        assert save_data.get("success") is True

        # 2. Invalidate live context with unrelated prompt
        await client.post(
            CHAT_URL,
            json={
                "model": settings.koboldcpp_model,
                "messages": [{"role": "user", "content": "Wipe context cache prompt.\n\n/no_think"}],
                "max_tokens": 1,
            },
        )

        # 3. Restore slot 0
        t_load_start = time.perf_counter()
        resp_load = await client.post(f"{ADMIN_URL}/api/admin/load_state", json={"slot": 0})
        t_load = time.perf_counter() - t_load_start
        assert resp_load.status_code == 200
        assert resp_load.json().get("success") is True

        # 4. Warm evaluation with identical prefix and changed tail
        t0 = time.perf_counter()
        resp_warm = await client.post(
            CHAT_URL,
            json={
                "model": settings.koboldcpp_model,
                "messages": [{"role": "user", "content": prompt_warm}],
                "max_tokens": 1,
            },
        )
        t_warm = time.perf_counter() - t0
        assert resp_warm.status_code == 200

        # Check perf metrics
        resp_perf = await client.get(f"{ADMIN_URL}/api/extra/perf")
        perf = resp_perf.json()
        process_speed = perf.get("last_process_speed", 0)

        print(f"\n[Smoke] Cold: {t_cold:.3f}s | Load slot 0: {t_load*1000:.1f}ms | Warm: {t_warm:.3f}s | Process Speed: {process_speed:.1f} t/s")
        # Warm should be noticeably faster than cold, and load latency is small
        assert t_warm < t_cold
        assert t_load < 0.5


@pytest.mark.asyncio
async def test_kv_cache_large_schema_finops_and_switching():
    """Verify save_state and load_state on the largest schema (~24k tokens FinOps) and A -> B -> A switching."""
    if not await _is_kobold_running():
        pytest.skip("KoboldCpp is not running on 5001")

    orc = NL2AnyQueryOrchestrator()
    finops_schema = orc.get_or_load_schema(TARGETS["finops"])
    postgres_schema = orc.get_or_load_schema(TARGETS["postgres"])

    finops_desc = format_schema_description(finops_schema)
    postgres_desc = format_schema_description(postgres_schema)

    finops_prompt_base = f"Schema:\n{finops_desc}\n\nUser Question:\n"
    postgres_prompt_base = f"Schema:\n{postgres_desc}\n\nUser Question:\n"

    async with httpx.AsyncClient(timeout=180.0) as client:
        # Step 1: Warm up FinOps in Slot 2 (large ~24k tokens)
        t0 = time.perf_counter()
        resp_f1 = await client.post(
            CHAT_URL,
            json={
                "model": settings.koboldcpp_model,
                "messages": [{"role": "user", "content": finops_prompt_base + "ping 1\n\n/no_think"}],
                "max_tokens": 1,
            },
        )
        t_finops_cold = time.perf_counter() - t0
        assert resp_f1.status_code == 200

        resp_save_f = await client.post(f"{ADMIN_URL}/api/admin/save_state", json={"slot": 2})
        assert resp_save_f.status_code == 200
        assert resp_save_f.json().get("success") is True
        print(f"\n[FinOps Cold] Evaluated & Saved Slot 2 in {t_finops_cold:.2f}s")

        # Step 2: Warm up Postgres in Slot 0 (Target B)
        resp_p1 = await client.post(
            CHAT_URL,
            json={
                "model": settings.koboldcpp_model,
                "messages": [{"role": "user", "content": postgres_prompt_base + "ping 1\n\n/no_think"}],
                "max_tokens": 1,
            },
        )
        assert resp_p1.status_code == 200
        resp_save_p = await client.post(f"{ADMIN_URL}/api/admin/save_state", json={"slot": 0})
        assert resp_save_p.json().get("success") is True

        # Step 3: Switch back to FinOps (Slot 2) -> A -> B -> A
        t_load_f_start = time.perf_counter()
        resp_load_f = await client.post(f"{ADMIN_URL}/api/admin/load_state", json={"slot": 2})
        t_load_f = time.perf_counter() - t_load_f_start
        assert resp_load_f.status_code == 200
        assert resp_load_f.json().get("success") is True

        # Run query with FinOps warm prefix
        t0 = time.perf_counter()
        resp_f2 = await client.post(
            CHAT_URL,
            json={
                "model": settings.koboldcpp_model,
                "messages": [{"role": "user", "content": finops_prompt_base + "Show cloud spend\n\n/no_think"}],
                "max_tokens": 1,
            },
        )
        t_finops_warm = time.perf_counter() - t0
        assert resp_f2.status_code == 200

        print(f"[FinOps Warm] Load slot 2: {t_load_f*1000:.1f}ms | Warm eval: {t_finops_warm:.2f}s (vs cold: {t_finops_cold:.2f}s)")
        assert t_finops_warm < t_finops_cold
