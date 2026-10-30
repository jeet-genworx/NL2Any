"""Recovery from the malformed Stage 1 payloads a small model actually emits.

Each case here is a real KoboldCpp response shape observed against the FinOps
schema, where the model picked the right tables and the answer was discarded
over the shape of the surrounding JSON.
"""

import pytest

from backend.src.schemas.pipeline import TableSelectionResult
from backend.src.utils.text_utils import extract_json_block

# Observed verbatim: the value is delimited by \" rather than ", which is
# illegal JSON. Both the plain parse and repair_json fail on it.
OVER_ESCAPED = r'''{
  "reason": \"(1) Concepts: Customer name and email from invoices. (2) Table Coverage: The required customer details are stored in finopsiq.customers and linked via finopsiq.invoices.customer_id. (3) Joins: a direct foreign key path exists. (4) Sufficiency: all concepts are covered.\",
  "selected_tables": [
    "finopsiq.invoices",
    "finopsiq.customers"
  ],
  "sufficient": true,
  "missing_objects": []
}'''


def test_over_escaped_delimiters_are_recovered():
    data = extract_json_block(OVER_ESCAPED)
    assert data["selected_tables"] == ["finopsiq.invoices", "finopsiq.customers"]
    assert data["reason"].startswith("(1) Concepts")


def test_over_escaped_payload_validates_into_a_selection():
    result = TableSelectionResult.model_validate(extract_json_block(OVER_ESCAPED))
    assert result.selected_objects == ["finopsiq.invoices", "finopsiq.customers"]
    assert result.sufficient is True


def test_over_escaped_payload_survives_a_prose_preamble():
    """The model often prefixes its answer, so the brace scan has to run first."""
    data = extract_json_block("Output: " + OVER_ESCAPED)
    assert data["selected_tables"] == ["finopsiq.invoices", "finopsiq.customers"]


def test_legitimate_inner_escaped_quotes_are_left_alone():
    """Pass 3 must not fire on a payload that already parses."""
    payload = r'''{"reason": "the column is named \"status\"", "selected_tables": ["t"]}'''
    data = extract_json_block(payload)
    assert data["reason"] == 'the column is named "status"'


def test_unparseable_response_still_raises():
    with pytest.raises(ValueError, match="Failed to extract valid JSON"):
        extract_json_block("I'm not going to generate a query for that.")


def test_reason_returned_as_a_list_is_joined():
    """Observed: the model answers `reason` with its reasoning steps as a list."""
    result = TableSelectionResult.model_validate(
        {
            "selected_tables": ["finopsiq.customers"],
            "sufficient": True,
            "reason": ["Concepts: the question asks for customers.", "Coverage: one table."],
        }
    )
    assert result.selected_objects == ["finopsiq.customers"]
    assert result.reason == (
        "Concepts: the question asks for customers. Coverage: one table."
    )


def test_reason_returned_as_a_non_string_scalar_is_coerced():
    result = TableSelectionResult.model_validate(
        {"selected_tables": ["t"], "sufficient": True, "reason": 42}
    )
    assert result.reason == "42"


def test_reason_as_a_plain_string_is_untouched():
    result = TableSelectionResult.model_validate(
        {"selected_tables": ["t"], "sufficient": True, "reason": "direct FK path"}
    )
    assert result.reason == "direct FK path"
