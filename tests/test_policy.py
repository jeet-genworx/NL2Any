"""Tests for SafetyPolicyValidator."""

from query_processing.models.pipeline import GeneratedQuery, MongoQuery
from query_processing.models.schema import DatabaseType
from query_processing.pipeline.policy import SafetyPolicyValidator


def test_sql_policy_allow_select():
    policy = SafetyPolicyValidator()
    q = GeneratedQuery(
        database_type=DatabaseType.POSTGRESQL,
        raw_query="SELECT id, name FROM customers WHERE city = 'Boston';",
        formatted_query="SELECT id, name FROM customers WHERE city = 'Boston';",
    )
    res = policy.check(q)
    assert res.allowed is True


def test_sql_policy_reject_insert():
    policy = SafetyPolicyValidator()
    q = GeneratedQuery(
        database_type=DatabaseType.POSTGRESQL,
        raw_query="INSERT INTO customers (id, name) VALUES (1, 'Evil');",
        formatted_query="INSERT INTO customers (id, name) VALUES (1, 'Evil');",
    )
    res = policy.check(q)
    assert res.allowed is False
    assert "Prohibited SQL expression" in res.reason or "not evaluate to a read-only SELECT" in res.reason


def test_sql_policy_reject_drop():
    policy = SafetyPolicyValidator()
    q = GeneratedQuery(
        database_type=DatabaseType.POSTGRESQL,
        raw_query="DROP TABLE customers;",
        formatted_query="DROP TABLE customers;",
    )
    res = policy.check(q)
    assert res.allowed is False


def test_sql_policy_reject_multiple_statements():
    policy = SafetyPolicyValidator()
    q = GeneratedQuery(
        database_type=DatabaseType.POSTGRESQL,
        raw_query="SELECT 1; DROP TABLE customers;",
        formatted_query="SELECT 1; DROP TABLE customers;",
    )
    res = policy.check(q)
    assert res.allowed is False
    assert "multiple SQL statements detected" in res.reason


def test_mongo_policy_allow_find_and_aggregate():
    policy = SafetyPolicyValidator()

    find_q = MongoQuery(operation="find", collection="orders", filter={"status": "completed"})
    q1 = GeneratedQuery(database_type=DatabaseType.MONGODB, raw_query=find_q, formatted_query="{}")
    assert policy.check(q1).allowed is True

    agg_q = MongoQuery(operation="aggregate", collection="orders", pipeline=[{"$match": {"status": "completed"}}])
    q2 = GeneratedQuery(database_type=DatabaseType.MONGODB, raw_query=agg_q, formatted_query="{}")
    assert policy.check(q2).allowed is True


def test_mongo_policy_reject_disallowed_stage():
    policy = SafetyPolicyValidator()
    agg_q = MongoQuery(operation="aggregate", collection="orders", pipeline=[{"$out": "new_orders"}])
    q = GeneratedQuery(database_type=DatabaseType.MONGODB, raw_query=agg_q, formatted_query="{}")
    res = policy.check(q)
    assert res.allowed is False
    assert "Prohibited MongoDB aggregation stage: '$out'" in res.reason
