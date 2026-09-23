"""Unified command-line interface entrypoints for NL2AnyQuery."""

import argparse
import asyncio
from pathlib import Path
import sys

from nl2anyquery.core.config import settings
from nl2anyquery.databases.mongo.adapter import MongoDBAdapter
from nl2anyquery.databases.postgres.adapter import PostgreSQLAdapter
from nl2anyquery.models.schema import DatabaseType
from nl2anyquery.nlp.linguistic import LinguisticAnalyzer
from nl2anyquery.providers.model.koboldcpp import KoboldCppProvider
from nl2anyquery.retrieval.bm25 import BM25Retriever
from nl2anyquery.schema.manager import SchemaManager, get_default_schema_path


def seed_postgres_cli() -> None:
    """CLI handler for seed-postgres."""
    from nl2anyquery.databases.seed.postgres import seed_postgres
    seed_postgres()


def seed_mongo_cli() -> None:
    """CLI handler for seed-mongo."""
    from nl2anyquery.databases.seed.mongo import seed_mongo
    seed_mongo()


def test_model_cli() -> None:
    """CLI handler for test-model: verifies KoboldCpp connectivity."""
    from nl2anyquery.core.text_utils import strip_think_tags

    async def _run() -> None:
        print(f"Connecting to KoboldCpp at: {settings.koboldcpp_base_url}")
        print(f"Target model: {settings.koboldcpp_model}")
        provider = KoboldCppProvider()
        try:
            response = await provider.test_connection()
            print("KoboldCpp connection successful!")
            cleaned = strip_think_tags(response)
            if cleaned:
                print(f"Model response: {cleaned}")
            else:
                print(f"Model raw output: {response.strip()!r}")
        except Exception as err:
            print(f"Error connecting to KoboldCpp: {err}", file=sys.stderr)
            sys.exit(1)

    asyncio.run(_run())


def init_schema_cli() -> None:
    """CLI handler for init-schema."""
    parser = argparse.ArgumentParser(description="Initialize database schema TOML.")
    parser.add_argument(
        "--database",
        "-d",
        choices=["postgres", "postgresql", "mongo", "mongodb"],
        required=True,
        help="Target database type",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output TOML file path (defaults to schemas/<db>.toml)",
    )
    parser.add_argument(
        "--describe",
        action="store_true",
        help="Enrich schema with descriptions via KoboldCpp SLM",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Force regeneration of existing descriptions",
    )
    args = parser.parse_args()

    async def _run() -> None:
        db_arg = args.database.lower()
        if db_arg in ("postgres", "postgresql"):
            db_type = DatabaseType.POSTGRESQL
            adapter = PostgreSQLAdapter(dsn=settings.postgres_dsn)
        else:
            db_type = DatabaseType.MONGODB
            adapter = MongoDBAdapter(uri=settings.mongodb_uri, database=settings.mongodb_database)

        out_path = Path(args.output) if args.output else get_default_schema_path(db_type)
        print(f"Extracting authoritative metadata for {db_type.value}...")
        manager = SchemaManager()

        with adapter:
            schema = await manager.initialize_schema(
                adapter=adapter,
                output_path=out_path,
                describe_with_slm=args.describe,
                refresh=args.refresh,
            )

        print(f"\nGenerated schema saved to: {out_path}")
        print(f"Total objects: {len(schema.objects)}")
        for obj in schema.objects:
            desc_preview = f" - '{obj.description}'" if obj.description else " (no description)"
            print(f"  [{obj.kind.value}] {obj.name} ({len(obj.fields)} fields){desc_preview}")
        print(f"Total relationships: {len(schema.relationships)}")

    asyncio.run(_run())


def test_bm25_cli() -> None:
    """CLI handler for test-bm25: test BM25 schema retrieval against a TOML schema."""
    parser = argparse.ArgumentParser(description="Test BM25 schema retrieval.")
    parser.add_argument(
        "--query",
        "-q",
        type=str,
        required=True,
        help="Natural language or keyword query",
    )
    parser.add_argument(
        "--database",
        "-d",
        choices=["postgres", "postgresql", "mongo", "mongodb"],
        default="postgres",
        help="Schema database type (defaults to postgres)",
    )
    parser.add_argument(
        "--schema-file",
        "-s",
        type=str,
        default=None,
        help="Custom path to schema TOML file",
    )
    parser.add_argument(
        "--top-k",
        "-k",
        type=int,
        default=None,
        help="Number of results to return",
    )
    args = parser.parse_args()

    db_type = (
        DatabaseType.POSTGRESQL
        if args.database.lower() in ("postgres", "postgresql")
        else DatabaseType.MONGODB
    )
    schema_path = Path(args.schema_file) if args.schema_file else get_default_schema_path(db_type)

    if not schema_path.exists():
        print(
            f"Schema file not found at: {schema_path}. "
            f"Run 'uv run init-schema --database {db_type.value}' first.",
            file=sys.stderr,
        )
        sys.exit(1)

    retriever = BM25Retriever.from_toml_file(schema_path)
    results = retriever.retrieve(args.query, top_k=args.top_k)

    print(f"\nBM25 Query: {args.query!r}")
    print(f"Schema Source: {schema_path}")
    print(f"Top matches ({len(results)}):\n" + "-" * 40)
    for r in results:
        field_count = len(r.schema_object.fields)
        print(f"• [{r.score:.4f}] {r.object_name} ({r.schema_object.kind.value}, {field_count} fields)")
        if r.schema_object.description:
            print(f"    Description: {r.schema_object.description}")


def test_nlp_cli() -> None:
    """CLI handler for test-nlp: test spaCy deterministic linguistic analysis."""
    parser = argparse.ArgumentParser(description="Test spaCy linguistic analyzer.")
    parser.add_argument(
        "--text",
        "-t",
        type=str,
        required=True,
        help="Input text / question to analyze",
    )
    args = parser.parse_args()

    analyzer = LinguisticAnalyzer()
    analysis = analyzer.analyze(args.text)

    print(f"\nInput: {args.text!r}")
    print("Nouns / Noun Chunks:", analysis.nouns)
    print("Verbs:", analysis.verbs)
    print("Named Entities:")
    if analysis.entities:
        for ent in analysis.entities:
            print(f"  • {ent.text} ({ent.label})")
    else:
        print("  (None detected)")


def run_api_cli() -> None:
    """Run FastAPI server."""
    import uvicorn
    print("Starting NL2AnyQuery FastAPI server on http://127.0.0.1:8000 ...")
    uvicorn.run("nl2anyquery.api.main:app", host="127.0.0.1", port=8000, reload=True)


def run_ui_cli() -> None:
    """Run Streamlit frontend."""
    import subprocess
    print("Starting NL2AnyQuery Streamlit frontend...")
    subprocess.run(["streamlit", "run", "frontend/app.py"])
