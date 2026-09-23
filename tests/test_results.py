"""Tests for ResultProcessor."""

from nl2anyquery.pipeline.results import ResultProcessor


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
