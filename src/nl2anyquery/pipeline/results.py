"""Result processing and bounded CSV generation."""

import csv
from datetime import date, datetime
from decimal import Decimal
import io
from typing import Any
from bson import ObjectId

from nl2anyquery.models.pipeline import ExecutionResult


def _sanitize_for_json(val: Any) -> Any:
    """Ensure all cell values are JSON-serializable."""
    if val is None:
        return None
    if isinstance(val, (datetime, date)):
        return val.isoformat()
    if isinstance(val, Decimal):
        return float(val)
    if isinstance(val, ObjectId):
        return str(val)
    if isinstance(val, (dict, list)):
        return val
    return str(val) if not isinstance(val, (int, float, bool)) else val


def _generate_csv(columns: list[str], rows: list[dict[str, Any]]) -> str:
    """Generate in-memory CSV string from bounded rows without duplicating memory."""
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=columns, extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        writer.writerow({k: _sanitize_for_json(row.get(k)) for k in columns})
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
            {k: _sanitize_for_json(v) for k, v in row.items()}
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
