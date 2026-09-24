"""Tests for in-memory BM25 schema object retriever."""

from query_processing.models.schema import (
    DatabaseSchema,
    DatabaseType,
    Field,
    SchemaObject,
    SchemaObjectKind,
)
from query_processing.retrieval.bm25 import BM25Retriever


def test_bm25_retriever_ranking():
    schema = DatabaseSchema(
        database_type=DatabaseType.POSTGRESQL,
        database_name="company",
        objects=[
            SchemaObject(
                name="customers",
                kind=SchemaObjectKind.TABLE,
                description="Contains client profiles, contact emails, and billing cities",
                fields=[
                    Field(name="id", type="integer"),
                    Field(name="email", type="varchar"),
                    Field(name="city", type="varchar"),
                ],
            ),
            SchemaObject(
                name="orders",
                kind=SchemaObjectKind.TABLE,
                description="Records transactions, dates, and order status",
                fields=[
                    Field(name="id", type="integer"),
                    Field(name="customer_id", type="integer"),
                    Field(name="order_date", type="timestamp"),
                    Field(name="status", type="varchar"),
                ],
            ),
            SchemaObject(
                name="support_tickets",
                kind=SchemaObjectKind.TABLE,
                description="Customer complaints, bug reports, and resolution logs",
                fields=[
                    Field(name="id", type="integer"),
                    Field(name="priority", type="varchar"),
                    Field(name="subject", type="varchar"),
                ],
            ),
        ],
    )

    retriever = BM25Retriever(schema)

    # 1. Query for tickets
    ticket_results = retriever.retrieve("Show all urgent customer complaints and bug reports", top_k=3)
    assert len(ticket_results) == 3
    assert ticket_results[0].object_name == "support_tickets"
    assert ticket_results[0].score > ticket_results[1].score

    # 2. Query for orders
    order_results = retriever.retrieve("Find transactions by status and order date", top_k=2)
    assert len(order_results) == 2
    assert order_results[0].object_name == "orders"

    # 3. Query for billing city and customer email
    cust_results = retriever.retrieve("customer billing city and email", top_k=2)
    assert cust_results[0].object_name == "customers"
