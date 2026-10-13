"""Repository for persisting, loading, and cleaning up pipeline run states."""

import json
import logging
import os
import re
import tempfile
import time
from pathlib import Path
from typing import Any

from backend.src.data.repositories.paths import runs_dir

logger = logging.getLogger(__name__)

RUN_ID_REGEX = re.compile(r"^[a-f0-9]{32}$")


def validate_run_id(run_id: str) -> bool:
    """Validate that run_id is strictly a 32-character hexadecimal string."""
    return bool(RUN_ID_REGEX.match(run_id))


def get_run_file_path(run_id: str) -> Path:
    """Return the filesystem path for a run ID, enforcing security checks."""
    if not validate_run_id(run_id):
        raise ValueError(f"Invalid run_id format: {run_id!r}. Must be 32 hex characters.")
    base_dir = runs_dir().resolve()
    target_path = (base_dir / f"{run_id}.json").resolve()
    if target_path.parent != base_dir:
        raise ValueError("Directory traversal attempt detected.")
    return target_path


def sanitize_run_data_for_persistence(data: dict[str, Any]) -> dict[str, Any]:
    """Strip bulky execution output (preview_rows, csv_data) to retain only what reruns need."""
    payload = dict(data)
    # Strip results preview rows & csv data if present
    if "results" in payload and isinstance(payload["results"], dict):
        res = dict(payload["results"])
        res.pop("preview_rows", None)
        res.pop("csv_data", None)
        payload["results"] = res
    elif hasattr(payload.get("results"), "model_dump"):
        res = payload["results"].model_dump()
        res.pop("preview_rows", None)
        res.pop("csv_data", None)
        payload["results"] = res

    # Ensure updated_at timestamp is present
    if "updated_at" not in payload:
        payload["updated_at"] = time.time()
    return payload


def save_run_state(run_data: dict[str, Any]) -> Path:
    """Atomically save run state to disk using a temporary file and replace."""
    run_id = run_data.get("run_id")
    if not run_id or not isinstance(run_id, str):
        raise ValueError("Missing or invalid run_id in run_data.")

    target_path = get_run_file_path(run_id)
    sanitized = sanitize_run_data_for_persistence(run_data)
    json_bytes = json.dumps(sanitized, indent=2, ensure_ascii=False, default=str).encode("utf-8")

    # Write atomically in the same directory using a temporary file
    target_dir = target_path.parent
    target_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(dir=target_dir, delete=False, prefix=f"{run_id}_tmp_", suffix=".json") as tf:
        tf.write(json_bytes)
        temp_path = Path(tf.name)

    try:
        os.replace(temp_path, target_path)
    except Exception:
        if temp_path.exists():
            temp_path.unlink()
        raise

    return target_path


def load_run_state(run_id: str) -> dict[str, Any] | None:
    """Load run state from disk, returning None if not found."""
    try:
        path = get_run_file_path(run_id)
    except ValueError:
        return None

    if not path.is_file():
        return None

    try:
        text = path.read_text(encoding="utf-8")
        return json.loads(text)
    except Exception as err:
        logger.error("Failed to load run state for %s: %s", run_id, err)
        return None


def cleanup_old_runs(max_age_seconds: int = 86400, max_files: int = 500) -> int:
    """Remove run files older than max_age_seconds and trim oldest files beyond max_files."""
    base_dir = runs_dir()
    if not base_dir.exists():
        return 0

    now = time.time()
    deleted_count = 0
    run_files: list[tuple[Path, float]] = []

    try:
        for p in base_dir.glob("*.json"):
            if p.is_file():
                try:
                    mtime = p.stat().st_mtime
                    if now - mtime > max_age_seconds:
                        p.unlink(missing_ok=True)
                        deleted_count += 1
                    else:
                        run_files.append((p, mtime))
                except Exception:
                    continue

        if len(run_files) > max_files:
            # Sort by mtime ascending (oldest first)
            run_files.sort(key=lambda x: x[1])
            excess = len(run_files) - max_files
            for p, _ in run_files[:excess]:
                try:
                    p.unlink(missing_ok=True)
                    deleted_count += 1
                except Exception:
                    continue

    except Exception as err:
        logger.warning("Error during run cleanup: %s", err)

    return deleted_count
