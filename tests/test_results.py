"""Tests for ResultProcessor."""

from backend.src.core.query_processing.pipeline.results import ResultProcessor


def test_result_processor_small_result():
    processor = ResultProcessor()
    columns = ["id", "name"]
    raw_rows = [{"id": i, "name": f"User {i}"} for i in range(10)]

    res = processor.process(columns, raw_rows)
    assert res.row_count == 10
    assert len(res.preview_rows) == 10
    assert res.truncated is False
    assert res.csv_data is None


def test_result_processor_large_result():
    processor = ResultProcessor()
    columns = ["id", "name"]
    raw_rows = [{"id": i, "name": f"User {i}"} for i in range(150)]

    res = processor.process(columns, raw_rows)
    assert res.row_count == 150
    assert len(res.preview_rows) == 10  # 10-row preview
    assert res.truncated is True
    assert res.csv_data is not None
    assert "User 149" in res.csv_data
    assert "id,name" in res.csv_data


def test_result_processor_bson_decimal128_and_nested_types():
    from bson import Decimal128, ObjectId
    from datetime import datetime
    import json

    processor = ResultProcessor()
    columns = ["_id", "model", "pricing", "tags"]
    raw_rows = [
        {
            "_id": ObjectId("65123456789abcdef0123456"),
            "model": "gpt-4o",
            "pricing": {
                "input_cost": Decimal128("2.50"),
                "output_cost": Decimal128("10.00"),
            },
            "tags": ["prod", "llm"],
            "created_at": datetime(2026, 1, 1, 12, 0, 0),
        }
    ]

    res = processor.process(columns, raw_rows)
    assert res.row_count == 1
    row = res.preview_rows[0]
    assert row["_id"] == "65123456789abcdef0123456"
    assert row["model"] == "gpt-4o"
    assert row["pricing"]["input_cost"] == 2.5
    assert row["pricing"]["output_cost"] == 10.0
    assert row["tags"] == ["prod", "llm"]
    assert row["created_at"] == "2026-01-01T12:00:00"

    # Must be JSON-dumpable with standard json without any custom encoders
    dumped = json.dumps(res.model_dump())
    assert "2.5" in dumped
    assert "65123456789abcdef0123456" in dumped

