"""Unit tests for SymSpell SpellingChecker stage."""

import pytest

from backend.src.core.query_processing.pipeline.spelling import SpellingChecker


@pytest.fixture(scope="module")
def checker() -> SpellingChecker:
    return SpellingChecker()


def test_correctly_spelled_query(checker: SpellingChecker):
    question = "Show me all customers from Chicago"
    res = checker.check(question)
    assert res.corrected_question == question
    assert len(res.corrections) == 0


def test_misspelled_query(checker: SpellingChecker):
    question = "Show me all customrs from Chicago"
    res = checker.check(question)
    assert res.corrected_question == "Show me all customers from Chicago"
    assert len(res.corrections) == 1
    assert res.corrections[0].original == "customrs"
    assert res.corrections[0].corrected == "customers"


def test_query_with_user_provided_jargon(checker: SpellingChecker):
    question = "Show PostgreSQL customrs"
    res = checker.check(question, jargons=["PostgreSQL"])
    assert "PostgreSQL" in res.corrected_question
    assert "customers" in res.corrected_question
    assert res.corrected_question == "Show PostgreSQL customers"
    # customrs should be in corrections, but PostgreSQL should not be altered
    assert any(c.original == "customrs" and c.corrected == "customers" for c in res.corrections)
    assert not any(c.original == "PostgreSQL" for c in res.corrections)


def test_misspelled_user_provided_jargon(checker: SpellingChecker):
    question = "Show me all orders in postgreql"
    res = checker.check(question, jargons=["PostgreSQL"])
    assert res.corrected_question == "Show me all orders in PostgreSQL"
    assert any(c.original == "postgreql" and c.corrected == "PostgreSQL" for c in res.corrections)


def test_multiple_spelling_errors(checker: SpellingChecker):
    question = "Show all customrs and ordrs"
    res = checker.check(question)
    assert res.corrected_question == "Show all customers and orders"
    assert len(res.corrections) == 2
    assert any(c.original == "customrs" and c.corrected == "customers" for c in res.corrections)
    assert any(c.original == "ordrs" and c.corrected == "orders" for c in res.corrections)


def test_technical_identifier_with_underscores(checker: SpellingChecker):
    question = "Filter by customer_id and total_amount"
    res = checker.check(question, jargons=["customer_id", "total_amount"])
    assert "customer_id" in res.corrected_question
    assert "total_amount" in res.corrected_question
    assert res.corrected_question == "Filter by customer_id and total_amount"


def test_empty_jargon_list(checker: SpellingChecker):
    # Empty jargon list should work normally without failing or using hardcoded domain lists
    question = "Show me all customrs"
    res1 = checker.check(question, jargons=[])
    assert res1.corrected_question == "Show me all customers"

    res2 = checker.check(question, jargons=None)
    assert res2.corrected_question == "Show me all customers"


def test_empty_question(checker: SpellingChecker):
    res = checker.check("")
    assert res.corrected_question == ""
    assert res.corrections == []
