"""Tests for text and model response utilities."""

import pytest
from nl2anyquery.core.text_utils import extract_json_block, strip_think_tags


def test_strip_think_tags_closed():
    raw = "<think>\nThinking about this...\nDone thinking.\n</think>\nFinal answer here."
    cleaned = strip_think_tags(raw)
    assert cleaned == "Final answer here."


def test_strip_think_tags_unclosed():
    raw = "<think>\nThinking about this but token limit truncated..."
    cleaned = strip_think_tags(raw)
    assert cleaned == ""


def test_strip_think_tags_no_tags():
    raw = "No reasoning tags here."
    assert strip_think_tags(raw) == "No reasoning tags here."


def test_extract_json_block_markdown_fence():
    text = (
        "<think>Let's create json</think>\n"
        "Here is the JSON:\n"
        "```json\n"
        '{"name": "customers", "count": 100}\n'
        "```"
    )
    result = extract_json_block(text)
    assert result == {"name": "customers", "count": 100}


def test_extract_json_block_raw_object():
    text = (
        "<think>Reasoning...</think>\n"
        'Prefix text {"table": "orders", "valid": true} suffix text'
    )
    result = extract_json_block(text)
    assert result == {"table": "orders", "valid": True}


def test_extract_json_block_raw_array():
    text = (
        "<think>Reasoning...</think>\n"
        '[{"id": 1}, {"id": 2}]'
    )
    result = extract_json_block(text)
    assert result == [{"id": 1}, {"id": 2}]


def test_extract_json_block_invalid():
    with pytest.raises(ValueError, match="Failed to extract valid JSON"):
        extract_json_block("<think>Thinking</think> No valid json at all")
