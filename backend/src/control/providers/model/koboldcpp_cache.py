"""KoboldCpp KV cache slot manager for Stage 1 (Table Selection).

Preloads and manages per-database schema prompt prefixes in KoboldCpp's KV cache
slots using admin endpoints (/api/admin/save_state and /api/admin/load_state).
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import time
from typing import Any, TYPE_CHECKING
import httpx

from backend.src.config import settings
from backend.src.data.repositories import paths
from backend.src.data.models.targets import DatabaseTarget, as_target

if TYPE_CHECKING:
    from backend.src.core.query_processing.pipeline.selector import TableSelector
    from backend.src.data.models.schema import DatabaseSchema

logger = logging.getLogger(__name__)


class KoboldCppCacheManager:
    """Manages slot allocation, prompt prefix caching, and KV cache state persistence

    for KoboldCpp during Stage 1 table selection.
    """

    def __init__(
        self,
        admin_url: str | None = None,
        max_slots: int | None = None,
        slot_mapping: dict[str, int] | None = None,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self.admin_url = (admin_url or settings.effective_koboldcpp_admin_url).rstrip("/")
        self.max_slots = max_slots if max_slots is not None else settings.koboldcpp_max_slots
        self.slot_mapping = slot_mapping or dict(settings.koboldcpp_slot_mapping)
        self.admin_timeout = settings.koboldcpp_admin_timeout_seconds
        self.warmup_timeout = settings.koboldcpp_warmup_timeout_seconds

        # Concurrency & single-flight management
        self._slot_lock = asyncio.Lock()
        self._warming_tasks: dict[str, asyncio.Task[bool]] = {}

        # In-memory slot & prefix state
        self._db_to_slot: dict[str, int] = {}
        self._slot_to_db: dict[int, str] = {}
        self._slot_access_time: dict[int, float] = {}
        self._free_slots: list[int] = list(range(self.max_slots))
        self._warmed_slots: set[int] = set()

        # Cache content hashes and static prefix strings
        self._schema_hashes: dict[str, str] = {}
        self._cached_prefixes: dict[str, str] = {}
        self._disabled_databases: set[str] = set()

        self._external_client = http_client

    async def _get_client(self, timeout: float | None = None) -> httpx.AsyncClient:
        if self._external_client is not None:
            return self._external_client
        return httpx.AsyncClient(timeout=timeout or self.admin_timeout)

    async def is_reachable(self) -> bool:
        """Check if KoboldCpp admin API is reachable."""
        endpoint = f"{self.admin_url}/api/extra/version"
        try:
            client = await self._get_client(timeout=3.0)
            if self._external_client is None:
                async with client:
                    resp = await client.get(endpoint)
                    return resp.status_code == 200
            resp = await client.get(endpoint)
            return resp.status_code == 200
        except Exception:
            return False

    def compute_schema_hash(self, target: DatabaseTarget) -> str:
        """Compute SHA-256 hash of the schema TOML and descriptions TOML for change detection."""
        hasher = hashlib.sha256()
        try:
            s_path = paths.schema_path(target)
            if s_path.exists():
                hasher.update(s_path.read_bytes())
        except Exception as err:
            logger.debug("Failed to read schema TOML for hash calculation: %s", err)

        try:
            d_path = paths.descriptions_path(target)
            if d_path.exists():
                hasher.update(d_path.read_bytes())
        except Exception as err:
            logger.debug("Failed to read descriptions TOML for hash calculation: %s", err)

        return hasher.hexdigest()

    def get_or_allocate_slot(self, db_key: str) -> int:
        """Allocate or retrieve slot for a database using LRU replacement when full."""
        # 1. Configured explicit slot override
        if db_key in self.slot_mapping:
            slot = self.slot_mapping[db_key]
            self._db_to_slot[db_key] = slot
            self._slot_to_db[slot] = db_key
            self._slot_access_time[slot] = time.monotonic()
            if slot in self._free_slots:
                self._free_slots.remove(slot)
            return slot

        # 2. Already assigned
        if db_key in self._db_to_slot:
            slot = self._db_to_slot[db_key]
            self._slot_access_time[slot] = time.monotonic()
            return slot

        # 3. Take first free slot
        if self._free_slots:
            slot = self._free_slots.pop(0)
            self._db_to_slot[db_key] = slot
            self._slot_to_db[slot] = db_key
            self._slot_access_time[slot] = time.monotonic()
            logger.info("Allocated free slot %d to database '%s'", slot, db_key)
            return slot

        # 4. LRU Eviction: Find least recently accessed slot
        lru_slot = min(self._slot_access_time, key=lambda s: self._slot_access_time[s])
        old_db = self._slot_to_db.get(lru_slot)
        if old_db and old_db in self._db_to_slot:
            del self._db_to_slot[old_db]
            self._warmed_slots.discard(lru_slot)
            logger.info("Evicted database '%s' from slot %d (LRU) for '%s'", old_db, lru_slot, db_key)

        self._db_to_slot[db_key] = lru_slot
        self._slot_to_db[lru_slot] = db_key
        self._slot_access_time[lru_slot] = time.monotonic()
        return lru_slot

    def is_warmed(self, db_key: str) -> bool:
        """Check if the database's slot has been warmed in memory."""
        slot = self._db_to_slot.get(db_key)
        if slot is None:
            return False
        return slot in self._warmed_slots

    def get_cached_prefix(self, db_key: str) -> str | None:
        """Return the pre-built static prefix string for a database if cached."""
        return self._cached_prefixes.get(db_key)

    def cache_prefix(self, db_key: str, prefix: str) -> None:
        """Store the static prefix in memory."""
        self._cached_prefixes[db_key] = prefix

    async def save_state(self, slot: int) -> bool:
        """Save KoboldCpp's live context into slot via POST /api/admin/save_state."""
        endpoint = f"{self.admin_url}/api/admin/save_state"
        try:
            client = await self._get_client()
            if self._external_client is None:
                async with client:
                    resp = await client.post(endpoint, json={"slot": slot})
            else:
                resp = await client.post(endpoint, json={"slot": slot})

            if resp.status_code == 200:
                data = resp.json()
                if data.get("success") is True:
                    self._warmed_slots.add(slot)
                    logger.debug("Successfully saved state to slot %d", slot)
                    return True
            logger.warning("save_state failed for slot %d: %s", slot, resp.text)
            return False
        except Exception as err:
            logger.warning("save_state request error for slot %d: %s", slot, err)
            return False

    async def load_state(self, slot: int) -> bool:
        """Load slot into KoboldCpp's live context via POST /api/admin/load_state."""
        endpoint = f"{self.admin_url}/api/admin/load_state"
        try:
            client = await self._get_client()
            if self._external_client is None:
                async with client:
                    resp = await client.post(endpoint, json={"slot": slot})
            else:
                resp = await client.post(endpoint, json={"slot": slot})

            if resp.status_code == 200:
                data = resp.json()
                if data.get("success") is True:
                    logger.debug("Successfully loaded state from slot %d (new_tokens=%s)", slot, data.get("new_tokens"))
                    return True
                logger.warning("load_state returned success=False for slot %d (slot may be empty)", slot)
                return False
            logger.warning("load_state HTTP %d for slot %d: %s", resp.status_code, slot, resp.text)
            return False
        except Exception as err:
            logger.warning("load_state request error for slot %d: %s", slot, err)
            return False

    async def get_perf(self) -> dict[str, Any]:
        """Fetch timing and token performance metrics from GET /api/extra/perf."""
        endpoint = f"{self.admin_url}/api/extra/perf"
        try:
            client = await self._get_client(timeout=3.0)
            if self._external_client is None:
                async with client:
                    resp = await client.get(endpoint)
            else:
                resp = await client.get(endpoint)
            if resp.status_code == 200:
                return resp.json()
        except Exception:
            pass
        return {}

    async def check_context_capacity(self, prompt: str) -> bool:
        """Check that prompt + max_tokens fits within KoboldCpp's configured context size."""
        try:
            client = await self._get_client(timeout=5.0)
            # 1. Fetch actual max context size
            max_ctx = 32768
            try:
                if self._external_client is None:
                    async with client:
                        resp = await client.get(f"{self.admin_url}/api/extra/true_max_context_length")
                else:
                    resp = await client.get(f"{self.admin_url}/api/extra/true_max_context_length")
                if resp.status_code == 200:
                    max_ctx = int(resp.json().get("value", max_ctx))
            except Exception:
                pass

            # 2. Measure tokens of prompt
            tok_count = len(prompt) // 4  # fallback heuristic
            try:
                client = await self._get_client(timeout=10.0)
                if self._external_client is None:
                    async with client:
                        resp = await client.post(f"{self.admin_url}/api/extra/tokencount", json={"prompt": prompt})
                else:
                    resp = await client.post(f"{self.admin_url}/api/extra/tokencount", json={"prompt": prompt})
                if resp.status_code == 200:
                    tok_count = int(resp.json().get("value", tok_count))
            except Exception:
                pass

            total_needed = tok_count + settings.table_selector_max_tokens
            if total_needed > max_ctx:
                logger.warning(
                    "Stage 1 prompt token count (%d) + gen budget (%d) = %d exceeds context size (%d). Caching skipped.",
                    tok_count,
                    settings.table_selector_max_tokens,
                    total_needed,
                    max_ctx,
                )
                return False
            return True
        except Exception as err:
            logger.debug("Failed context capacity check: %s", err)
            return True

    def trigger_warmup(
        self,
        target: DatabaseTarget,
        selector: TableSelector,
        schema: DatabaseSchema,
    ) -> asyncio.Task[bool]:
        """Trigger background warm-up task for a selected database (single-flight per DB)."""
        db_key = target.key

        # If warm-up is already running for this database, reuse the existing task
        existing = self._warming_tasks.get(db_key)
        if existing and not existing.done():
            return existing

        # Create single-flight background task if running in an event loop
        try:
            loop = asyncio.get_running_loop()
            task = loop.create_task(
                self._warmup_worker(target=target, selector=selector, schema=schema),
                name=f"warmup-{db_key}",
            )
            self._warming_tasks[db_key] = task
            return task
        except RuntimeError:
            # Fallback if invoked outside an active event loop
            return None

    async def _warmup_worker(
        self,
        target: DatabaseTarget,
        selector: TableSelector,
        schema: DatabaseSchema,
    ) -> bool:
        """Execute single warm-up probe and snapshot slot under lock."""
        db_key = target.key
        try:
            current_hash = self.compute_schema_hash(target)
            slot = self.get_or_allocate_slot(db_key)

            # Check if already warmed with same schema
            if self.is_warmed(db_key) and self._schema_hashes.get(db_key) == current_hash:
                logger.debug("Database '%s' is already warmed in slot %d with matching schema hash", db_key, slot)
                return True

            # Format prefix and full warmup prompt
            prefix = selector.build_static_prefix(schema)
            self.cache_prefix(db_key, prefix)

            warmup_question = settings.koboldcpp_warmup_dummy_question
            warmup_prompt = f"{prefix}User Question:\n{warmup_question}\n"

            # Check context capacity
            fits = await self.check_context_capacity(warmup_prompt)
            if not fits:
                self._disabled_databases.add(db_key)
                return False

            logger.info("Starting KV cache warm-up for database '%s' in slot %d...", db_key, slot)
            t0 = time.perf_counter()

            # Warm-up must run dummy generate and save_state under the same lock hold,
            # with no await on anything else between them.
            async with self._slot_lock:
                await selector.provider.generate(
                    prompt=warmup_prompt,
                    temperature=0.0,
                    max_tokens=1,
                )
                saved = await self.save_state(slot)

            elapsed = time.perf_counter() - t0
            if saved:
                self._schema_hashes[db_key] = current_hash
                self._warmed_slots.add(slot)
                logger.info(
                    "KV cache warm-up complete for database '%s' (slot %d) in %.2fs",
                    db_key,
                    slot,
                    elapsed,
                )
                return True
            else:
                logger.warning("KV cache save_state failed during warm-up for '%s' (slot %d)", db_key, slot)
                return False

        except Exception as err:
            logger.warning("KV cache warm-up error for database '%s': %s", db_key, err)
            return False


# Global default cache manager instance
_global_cache_manager: KoboldCppCacheManager | None = None


def get_cache_manager() -> KoboldCppCacheManager:
    """Get or create singleton KoboldCppCacheManager."""
    global _global_cache_manager
    if _global_cache_manager is None:
        _global_cache_manager = KoboldCppCacheManager()
    return _global_cache_manager
