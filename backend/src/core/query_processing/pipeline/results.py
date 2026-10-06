"""Result processing and bounded CSV generation."""

import csv
from datetime import date, datetime
from decimal import Decimal
import io
import json
from typing import Any

try:
    from bson import Decimal128, ObjectId
except ImportError:  # pragma: no cover
    Decimal128 = None  # type: ignore[assignment,misc]
    ObjectId = None  # type: ignore[assignment,misc]

from backend.src.schemas.pipeline import ExecutionResult


def _sanitize_for_json(val: Any) -> Any:
    """Ensure all cell values are JSON-serializable, recursively handling dicts, lists, and BSON types."""
    if val is None:
        return None
    if isinstance(val, bool):
        return val
    if isinstance(val, (int, float, str)):
        return val
    if isinstance(val, (datetime, date)):
        return val.isoformat()
    if isinstance(val, Decimal):
        return float(val)
    if Decimal128 is not None and isinstance(val, Decimal128):
        try:
            return float(val.to_decimal())
        except Exception:
            return float(str(val))
    if ObjectId is not None and isinstance(val, ObjectId):
        return str(val)
    if isinstance(val, bytes):
        return val.hex()
    if isinstance(val, dict):
        return {str(k): _sanitize_for_json(v) for k, v in val.items()}
    if isinstance(val, (list, tuple, set)):
        return [_sanitize_for_json(item) for item in val]
    try:
        if hasattr(val, "to_decimal"):
            return float(val.to_decimal())
        if hasattr(val, "as_datetime"):
            return val.as_datetime().isoformat()
    except Exception:
        pass
    return str(val)


def _generate_csv(columns: list[str], rows: list[dict[str, Any]]) -> str:
    """Generate in-memory CSV string from bounded rows without duplicating memory."""
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=columns, extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        formatted_row: dict[str, Any] = {}
        for k in columns:
            v = row.get(k)
            if isinstance(v, (dict, list)):
                formatted_row[k] = json.dumps(v, ensure_ascii=False)
            elif v is None:
                formatted_row[k] = ""
            else:
                formatted_row[k] = str(v)
        writer.writerow(formatted_row)
    return output.getvalue()


class ResultProcessor:
    """Processes query result rows into preview rows and on-demand CSV data."""

    def process(
        self,
        columns: list[str],
        raw_rows: list[dict[str, Any]],
    ) -> ExecutionResult:
        """Process rows into memory-efficient ExecutionResult."""
        row_count = len(raw_rows)
        sanitized_rows = [
            {str(k): _sanitize_for_json(v) for k, v in row.items()}
            for row in raw_rows
        ]

        if row_count >= 100:
            preview = sanitized_rows[:10]
            truncated = True
            csv_str = _generate_csv(columns, sanitized_rows)
        else:
            preview = sanitized_rows
            truncated = False
            csv_str = None

        return ExecutionResult(
            row_count=row_count,
            columns=columns,
            preview_rows=preview,
            truncated=truncated,
            csv_data=csv_str,
        )

