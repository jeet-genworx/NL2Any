"""Unified command-line interface entrypoints for NL2AnyQuery."""

import argparse
import asyncio
from pathlib import Path
import sys

from backend.src.config import settings
from backend.src.data.models.schema import DatabaseType
from backend.src.data.models.targets import TARGETS, resolve_target
from backend.src.core.query_processing.nlp.linguistic import LinguisticAnalyzer
from backend.src.control.providers.model.koboldcpp import KoboldCppProvider
from backend.src.core.query_processing.retrieval.semantic import SemanticRetriever
from backend.src.core.ingestion.pipeline import extract_and_save_schema, run_ingestion_pipeline
from backend.src.core.ingestion.schema.describe import (
    apply_descriptions,
    describe_tables,
    table_descriptions,
)
from backend.src.core.ingestion.schema.embed import embed_descriptions
from backend.src.data.repositories import (
    description_repository,
    embedding_repository,
    paths,
    schema_repository,
)

_DATABASE_CHOICES = ("postgres", "postgresql", "mongo", "mongodb", "finops")


def _add_database_argument(parser: argparse.ArgumentParser, *, required: bool = False) -> None:
    """Add the --database/-d option every ingestion command shares."""
    parser.add_argument(
        "--database",
        "-d",
        choices=_DATABASE_CHOICES,
        required=required,
        default=None if required else "postgres",
        help="Target database type",
    )


def _database_target(value: str):
    """Map a --database argument onto its DatabaseTarget."""
    return resolve_target(value)


def _database_type(value: str) -> DatabaseType:
    """Map a --database argument onto its engine type."""
    return resolve_target(value).db_type


def seed_postgres_cli() -> None:
    """CLI handler for seed-postgres."""
    from backend.src.data.seed.postgres import seed_postgres
    seed_postgres()


def seed_mongo_cli() -> None:
    """CLI handler for seed-mongo."""
    from backend.src.data.seed.mongo import seed_mongo
    seed_mongo()


def test_model_cli() -> None:
    """CLI handler for test-model: verifies KoboldCpp connectivity."""
    from backend.src.utils.text_utils import strip_think_tags

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
    """CLI handler for init-schema: extracts authoritative metadata via a database
    adapter and writes the canonical schema TOML, graph TOML, and (Postgres
    only) MST TOML consumed by query processing and by describe-schema."""
    parser = argparse.ArgumentParser(description="Initialize database schema TOML.")
    _add_database_argument(parser, required=True)
    args = parser.parse_args()

    db_type = _database_target(args.database)

    print(f"Extracting authoritative metadata for {db_type.key}...")
    schema, schema_path, graph_path, mst_path = extract_and_save_schema(db_type)

    print(f"\nGenerated schema saved to: {schema_path}")
    print(f"Generated graph saved to: {graph_path}")
    if mst_path:
        print(f"Generated minimum spanning tree saved to: {mst_path}")
    else:
        print("Skipping MST (MongoDB has no foreign-key relationships to reduce).")

    print(f"Total objects: {len(schema.objects)}")
    for obj in schema.objects:
        print(f"  [{obj.kind.value}] {obj.name} ({len(obj.fields)} fields)")
    print(f"Total relationships: {len(schema.relationships)}")


def describe_schema_cli() -> None:
    """CLI handler for describe-schema: generates an SLM description for each
    table and for each of its columns, processed in batches of connected tables
    so related tables are described with relationship context. Table
    Table and column descriptions are saved as a documentation TOML (the
    embedding stage's input) and are also written back into the canonical
    schema TOML. Requires a running KoboldCpp
    instance. Run init-schema first."""
    parser = argparse.ArgumentParser(
        description="Generate per-table and per-column descriptions via the SLM, "
        "from the schema graph/MST."
    )
    _add_database_argument(parser)
    parser.add_argument(
        "--use-mst",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use the MST (tree order, reduced edges) instead of the plain graph "
        "(extraction order, full edge set). Ignored for MongoDB, which has no MST.",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output TOML file path (defaults to backend/src/data/schemas/<db>_descriptions.toml)",
    )
    args = parser.parse_args()

    db_type = _database_target(args.database)
    use_mst = args.use_mst and db_type.has_mst
    source_path = paths.mst_path(db_type) if use_mst else paths.graph_path(db_type)
    out_path = Path(args.output) if args.output else paths.descriptions_path(db_type)

    async def _run() -> None:
        print(f"Reading {'MST' if use_mst else 'graph'} from: {source_path}")
        print(f"Connecting to KoboldCpp at: {settings.koboldcpp_base_url}")
        try:
            descriptions = await describe_tables(source_path)
        except FileNotFoundError as err:
            print(str(err), file=sys.stderr)
            sys.exit(1)
        except ConnectionError as err:
            print(f"Error connecting to KoboldCpp: {err}", file=sys.stderr)
            sys.exit(1)

        # Loaded up front so the documentation file can record which database
        # these descriptions came from.
        schema_path = paths.schema_path(db_type)
        schema = schema_repository.load_schema(schema_path) if schema_path.exists() else None

        # Fold both into the canonical schema TOML, the file query processing
        # reads, before writing the documentation from it.
        if schema is not None:
            column_count = apply_descriptions(schema, descriptions)
            schema_repository.save_schema(schema, schema_path)
            print(f"Wrote {column_count} column descriptions into: {schema_path}")
        else:
            print(
                f"Schema TOML not found at {schema_path}; the documentation file "
                "will list only the columns the model returned. "
                "Run 'uv run init-schema' first.",
                file=sys.stderr,
            )

        description_repository.save_descriptions(
            descriptions,
            out_path,
            database_type=db_type,
            database_name=schema.database_name if schema else "",
            schema=schema,
        )

        column_total = sum(len(table.columns) for table in descriptions.values())
        print(
            f"\nGenerated {len(descriptions)} table descriptions and {column_total} "
            f"column descriptions, saved to: {out_path}"
        )

        for table_name, table in descriptions.items():
            print(f"  {table_name} ({len(table.columns)} columns): {table.description}")

    asyncio.run(_run())


def embed_schema_cli() -> None:
    """CLI handler for embed-schema: embeds each table's description (from
    describe-schema's output) via the local MiniLM embedding model served by
    KoboldCpp's --embeddingsmodel endpoint, and saves table_name -> vector
    as a JSON file. Requires a running KoboldCpp instance with an embeddings
    model loaded."""
    parser = argparse.ArgumentParser(
        description="Embed per-table descriptions via the local embedding model."
    )
    _add_database_argument(parser)
    parser.add_argument(
        "--descriptions-file",
        type=str,
        default=None,
        help="Custom path to the descriptions TOML file (defaults to backend/src/data/schemas/<db>_descriptions.toml)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output JSON file path (defaults to settings.postgres_embeddings_path for postgres, "
        "settings.mongo_embeddings_path for mongo)",
    )
    args = parser.parse_args()

    db_type = _database_target(args.database)
    descriptions_path = (
        Path(args.descriptions_file) if args.descriptions_file else paths.descriptions_path(db_type)
    )
    out_path = Path(args.output) if args.output else paths.embeddings_path(db_type)

    async def _run() -> None:
        print(f"Reading descriptions from: {descriptions_path}")
        try:
            descriptions = description_repository.load_descriptions(descriptions_path)
        except FileNotFoundError as err:
            print(str(err), file=sys.stderr)
            sys.exit(1)

        print(f"Embedding {len(descriptions)} table descriptions via: {settings.koboldcpp_embedding_model}")
        print("Column descriptions in the file are not embedded.")
        print(f"Connecting to KoboldCpp at: {settings.koboldcpp_base_url}")
        try:
            # Narrowed to table descriptions only -- column text stays out of
            # the retrieval vectors.
            embeddings = await embed_descriptions(table_descriptions(descriptions))
        except ConnectionError as err:
            print(f"Error connecting to KoboldCpp: {err}", file=sys.stderr)
            sys.exit(1)

        embedding_repository.save_embeddings(embeddings, out_path)

        dims = len(next(iter(embeddings.values()))) if embeddings else 0
        print(f"\nGenerated {len(embeddings)} table embeddings ({dims} dimensions each), saved to: {out_path}")

    asyncio.run(_run())


def ingest_schema_cli() -> None:
    """CLI handler for ingest-schema: runs the full ingestion flow in one
    shot (metadata -> graph/MST -> SLM descriptions -> embeddings) -- the
    same pipeline the API's /ingest/{database} endpoint runs when a user
    selects a database in the frontend. Requires a running KoboldCpp
    instance with an embeddings model loaded."""
    parser = argparse.ArgumentParser(description="Run the full ingestion pipeline for a database.")
    _add_database_argument(parser, required=True)
    parser.add_argument(
        "--use-mst",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use the MST for contextual table grouping instead of the plain graph. "
        "Ignored for MongoDB, which has no MST.",
    )
    parser.add_argument(
        "--force",
        "-f",
        action="store_true",
        help="Re-ingest even if this database already has embeddings on disk "
        "(the default is to reuse them and skip the run).",
    )
    args = parser.parse_args()

    db_type = _database_target(args.database)

    async def _run() -> None:
        print(f"Running full ingestion pipeline for {db_type.key}...")
        try:
            summary = await run_ingestion_pipeline(
                db_type, use_mst=args.use_mst, force=args.force
            )
        except ConnectionError as err:
            print(f"Error connecting to KoboldCpp: {err}", file=sys.stderr)
            sys.exit(1)

        if summary.get("skipped"):
            print(
                f"\nSkipped: {summary['database_name']} already has "
                f"{summary['embedding_count']} embeddings at {summary['embeddings_path']}.\n"
                "Re-run with --force to regenerate."
            )
            return

        print(f"\nDatabase: {summary['database_name']} ({summary['table_count']} tables, "
              f"{summary['relationship_count']} relationships)")
        print(f"Used MST for description grouping: {summary['used_mst']}")
        print(f"Descriptions: {summary['description_count']} -> {summary['descriptions_path']}")
        print(
            f"Column descriptions: {summary['column_description_count']} -> {summary['schema_path']}"
        )
        print(
            f"Embeddings: {summary['embedding_count']} "
            f"({summary['embedding_dimensions']} dimensions each) -> {summary['embeddings_path']}"
        )

    asyncio.run(_run())


def test_retrieval_cli() -> None:
    """CLI handler for test-retrieval: test semantic schema retrieval against
    a TOML schema and its embeddings JSON. Requires a running KoboldCpp
    instance with an embeddings model loaded, and that 'ingest-schema' (or
    'describe-schema' + 'embed-schema') has already been run for this database."""
    parser = argparse.ArgumentParser(description="Test semantic schema retrieval.")
    parser.add_argument(
        "--query",
        "-q",
        type=str,
        required=True,
        help="Natural language query",
    )
    _add_database_argument(parser)
    parser.add_argument(
        "--schema-file",
        "-s",
        type=str,
        default=None,
        help="Custom path to schema TOML file",
    )
    parser.add_argument(
        "--embeddings-file",
        type=str,
        default=None,
        help="Custom path to embeddings JSON file",
    )
    parser.add_argument(
        "--top-k",
        "-k",
        type=int,
        default=None,
        help="Number of results to return",
    )
    args = parser.parse_args()

    db_type = _database_target(args.database)
    schema_path = Path(args.schema_file) if args.schema_file else paths.schema_path(db_type)
    embeddings_path = (
        Path(args.embeddings_file) if args.embeddings_file else paths.embeddings_path(db_type)
    )

    if not schema_path.exists():
        print(
            f"Schema file not found at: {schema_path}. "
            f"Run 'uv run init-schema --database {db_type.key}' first.",
            file=sys.stderr,
        )
        sys.exit(1)

    async def _run() -> None:
        try:
            embeddings = embedding_repository.load_embeddings(embeddings_path)
        except FileNotFoundError as err:
            print(str(err), file=sys.stderr)
            sys.exit(1)

        schema = schema_repository.load_schema(schema_path)
        retriever = SemanticRetriever(schema, embeddings)
        try:
            results = await retriever.retrieve(args.query, top_k=args.top_k)
        except ConnectionError as err:
            print(f"Error connecting to KoboldCpp: {err}", file=sys.stderr)
            sys.exit(1)

        print(f"\nQuery: {args.query!r}")
        print(f"Schema Source: {schema_path}")
        print(f"Top matches ({len(results)}):\n" + "-" * 40)
        for r in results:
            field_count = len(r.schema_object.fields)
            print(f"• [{r.score:.4f}] {r.object_name} ({r.schema_object.kind.value}, {field_count} fields)")
            if r.schema_object.description:
                print(f"    Description: {r.schema_object.description}")

    asyncio.run(_run())


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
    uvicorn.run("backend.src.main:app", host="127.0.0.1", port=8000, reload=True)


def run_ui_cli() -> None:
    """Run Streamlit frontend."""
    import subprocess
    print("Starting NL2AnyQuery Streamlit frontend...")
    subprocess.run(["streamlit", "run", "frontend/app.py"])


def generate_test_embeddings_cli() -> None:
    """Generate table description embeddings using KoboldCpp and all-MiniLM-L6-v2."""
    import json
    from backend.src.control.providers.embedding.koboldcpp import KoboldCppEmbeddingProvider

    parser = argparse.ArgumentParser(
        description="Generate test table description embeddings using KoboldCpp."
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="tests/fixtures/postgres_embeddings.json",
        help="Output file path (default: tests/fixtures/postgres_embeddings.json)",
    )
    parser.add_argument(
        "--production",
        "-p",
        action="store_true",
        help="Also write to backend/src/data/embeddings/postgres_embeddings.json",
    )
    args = parser.parse_args()

    table_descriptions = {
        "customers": "Customer records containing customer identity, name, email, phone, and city information.",
        "orders": "Orders placed by customers containing order dates, status, total amounts, shipping city, and customer id.",
        "order_items": "Individual items within customer orders, containing product id, order id, quantity, and unit price.",
        "products": "Products available for purchase including product names, categories, prices, and stock quantity.",
        "employees": "Employee workforce records containing employee name, email, department, role, city, and hire date.",
        "support_tickets": "Customer support ticket records containing issue subject, status, priority, customer id, and employee id.",
    }

    async def _generate() -> None:
        print(f"Connecting to KoboldCpp embedding endpoint at: {settings.koboldcpp_base_url}")
        print(f"Embedding model: {settings.embedding_model}")
        provider = KoboldCppEmbeddingProvider()

        embeddings: dict[str, list[float]] = {}
        for tbl_name, desc in table_descriptions.items():
            print(f"Generating embedding for '{tbl_name}'...")
            vec = await provider.embed(desc)
            if len(vec) != 384:
                raise ValueError(f"Table '{tbl_name}' vector dimension {len(vec)} != 384")
            embeddings[tbl_name] = vec

        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(embeddings, indent=2), encoding="utf-8")
        print(f"Saved {len(embeddings)} table embeddings to: {out_path}")

        if args.production:
            prod_path = Path("backend/src/data/embeddings/postgres_embeddings.json")
            prod_path.parent.mkdir(parents=True, exist_ok=True)
            prod_path.write_text(json.dumps(embeddings, indent=2), encoding="utf-8")
            print(f"Also saved production embeddings to: {prod_path}")

    asyncio.run(_generate())
