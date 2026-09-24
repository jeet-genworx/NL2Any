# NL2AnyQuery — Natural Language to Database Query POC

> **Proof-of-Concept Notice**: This repository is a lightweight, local proof-of-concept demonstrating a deterministic, modular pipeline that translates natural language questions into safe, read-only PostgreSQL and MongoDB queries using small language models (SLMs).

---

## 1. Architecture Overview

```text
User Question
      │
      ▼
┌──────────────────────────┐
│      Guardrail SLM       │ ──[REJECT]──► "Sorry, I can't help with this."
└─────────────┬────────────┘
              │
         [READ_QUERY]
              │
              ├──► [BASIC] ──► Deterministic System Handler (identity/status)
              │
              ▼
┌──────────────────────────┐
│  Semantic Analysis SLM   │ ──► Subjective Entities & Objective Constraints
└─────────────┬────────────┘
              │
              ▼
┌──────────────────────────┐
│      spaCy Analyzer      │ ──► Linguistic Nouns, Verbs, Named Entities
└─────────────┬────────────┘
              │
              ▼
┌──────────────────────────┐
│       BM25 Search        │ ──► Top Candidate Objects from In-Memory Index
└─────────────┬────────────┘     (built once from canonical TOML schema)
              │
              ▼
┌──────────────────────────┐
│ Table Selection SLM      │ ──► Strict subset of candidates (validated vs TOML)
└─────────────┬────────────┘
              │
              ▼
┌──────────────────────────┐
│  Relationship Expander   │ ──► Deterministic Python Foreign Key / Bridge Table expansion
└─────────────┬────────────┘
              │
              ▼
┌──────────────────────────┐
│       Planner SLM        │ ──► Database-independent, semantic QueryPlan
└─────────────┬────────────┘     (Zero SQL / Mongo syntax)
              │
              ▼
┌──────────────────────────┐
│ Database Query Generator │ ──► PostgreSQL SQL or Typed MongoQuery structure
└─────────────┬────────────┘
              │
              ▼
┌──────────────────────────┐
│     Query Validator      │ ──► Deterministic AST/Schema verification + SLM check
└─────────────┬────────────┘     (Retries up to 3 times with feedback on failure)
              │
              ▼
┌──────────────────────────┐
│ Deterministic Safety     │ ──► SQLGlot AST SELECT-only check (Postgres)
│      Policy Boundary     │     Strict find/aggregate verification (MongoDB)
└─────────────┬────────────┘
              │
              ▼
┌──────────────────────────┐
│    Database Execution    │ ──► Safe execution with timeouts & MAX_RESULT_ROWS
└─────────────┬────────────┘
              │
              ▼
┌──────────────────────────┐
│     Result Processor     │ ──► Memory-efficient 10-row preview + CSV generation
└─────────────┬────────────┘
              │
              ▼
┌──────────────────────────┐
│   FastAPI & Streamlit    │ ──► Interactive Web UI with full pipeline stage tracing
└──────────────────────────┘
```

---

## 2. Key Design Rationales

* **Why an SLM (Qwen3-4B-Instruct via KoboldCpp)?**
  A local 4B model running on consumer hardware is fast, private, and capable of structured JSON reasoning when provided with tightly bounded contexts (groups of $\le 5$ objects, only relevant schema columns). Dedicated single-responsibility prompts avoid confusion and hallucinations.
* **Why BM25 instead of Embeddings / Vector Databases?**
  Database schemas are exact lexical artifacts (table names, column names, domain nouns). BM25 retrieval (`rank-bm25`) is fast, completely deterministic, requires zero external embedding models or vector databases, and executes entirely in memory.
* **Why spaCy?**
  Extracts linguistic parts-of-speech (nouns, noun chunks, verbs, entities like cities and dates) deterministically in microseconds, augmenting the retrieval query without relying on an LLM.
* **Why TOML as the Canonical Semantic Schema?**
  Live database introspection is slow and lacks business semantics. The TOML files (`ingestion/schemas/postgres.toml`, `ingestion/schemas/mongo.toml`) provide an offline, human-readable, single source of truth containing enriched table and column descriptions, data types, and verified foreign keys.
* **Why Deterministic Safety Boundaries?**
  The SLM is **never** the final safety boundary. Generated SQL is parsed into an AST via `sqlglot` to verify that only a single `SELECT` statement is present, rejecting `INSERT`, `UPDATE`, `DELETE`, `DROP`, `ALTER`, and multi-statement queries before reaching the database driver.

---

## 3. Project Structure

The codebase is split into three top-level packages by responsibility: **ingestion** (getting schema/data into the system), **query_processing** (the NL → query pipeline, retrieval, model provider, and API), and **frontend** (the Streamlit UI). `docker-compose.yml` builds one shared image and runs each layer as its own container.

```text
nl2anyquery/
├── .env.example                  # Generic environment variable template
├── .gitignore                    # Ignores .env and build caches
├── pyproject.toml                # Project dependencies and console entrypoints
├── README.md                     # Complete documentation
├── docker-compose.yml            # postgres, mongo, seed, api, frontend services
├── docker/
│   ├── Dockerfile.app            # Image shared by seed/api/frontend containers
│   ├── Dockerfile.postgres       # Postgres container definition
│   └── seed.sh                   # Runs seed-postgres + seed-mongo on startup
│
├── ingestion/                     # Getting schema & data INTO the system
│   ├── databases/                 # Deterministic database discovery adapters
│   │   ├── base.py
│   │   ├── detector.py
│   │   ├── postgres/
│   │   ├── mongo/
│   │   └── seed/                  # Synthetic demo data generators
│   ├── schema/                    # TOML serialization & SLM profiler
│   │   ├── toml_store.py
│   │   ├── profiler.py
│   │   └── manager.py
│   ├── schemas/                   # Canonical semantic schema TOML stores
│   │   ├── postgres.toml
│   │   └── mongo.toml
│   └── scripts/                   # Standalone CLI entrypoints
│       ├── seed_postgres.py
│       ├── seed_mongo.py
│       └── init_schema.py
│
├── query_processing/              # The NL → safe query pipeline
│   ├── api/                       # FastAPI web server (GET /health, POST /query, GET /schema/{db})
│   │   └── main.py
│   ├── core/                      # Settings and response cleaning utilities
│   │   ├── config.py
│   │   └── text_utils.py
│   ├── models/                    # Pydantic data contracts
│   │   ├── schema.py              # Normalized schema representation
│   │   └── pipeline.py            # Pipeline stage outputs & trace models
│   ├── nlp/                       # Deterministic spaCy linguistic analyzer
│   │   └── linguistic.py
│   ├── pipeline/                  # Sequential pipeline stages
│   │   ├── guardrail.py           # READ_QUERY, BASIC, REJECT classification
│   │   ├── semantic.py            # Subjective/objective extraction
│   │   ├── selector.py            # Candidate selection with strict TOML checks
│   │   ├── expansion.py           # Foreign key & bridge table expansion
│   │   ├── planner.py             # Syntax-free semantic query planning
│   │   ├── generators/            # Postgres SQL & typed MongoQuery generators
│   │   ├── validator.py           # AST schema verification + SLM validation
│   │   ├── policy.py              # Non-negotiable read-only safety boundary
│   │   ├── executor.py            # Safe DB driver execution with timeouts
│   │   ├── results.py             # Bounded preview & CSV generation
│   │   └── orchestrator.py        # Master pipeline runner & 3-attempt retry loop
│   ├── providers/                 # Local model provider abstraction
│   │   └── model/
│   ├── retrieval/                 # In-memory BM25 schema search
│   │   └── bm25.py
│   ├── prompts/                   # Focused, single-responsibility prompt templates
│   │   ├── guardrail.txt
│   │   ├── semantic_analysis.txt
│   │   ├── table_selection.txt
│   │   ├── planner.txt
│   │   ├── postgres_query_generator.txt
│   │   ├── mongo_query_generator.txt
│   │   ├── validator.txt
│   │   └── schema_description.txt
│   └── cli.py                     # Unified `uv run` console entrypoints
│
├── frontend/
│   └── app.py                     # Streamlit UI with pipeline stage inspection
│
└── tests/                         # 57 comprehensive automated unit tests
    ├── test_detector.py
    ├── test_schema.py
    ├── test_toml.py
    ├── test_bm25.py
    ├── test_linguistic.py
    ├── test_model_provider.py
    ├── test_text_utils.py
    ├── test_guardrail.py
    ├── test_semantic.py
    ├── test_selector.py
    ├── test_expansion.py
    ├── test_planner.py
    ├── test_generators.py
    ├── test_validator.py
    ├── test_policy.py
    ├── test_results.py
    ├── test_orchestrator.py
    └── test_api.py
```

---

## 4. Setup & Installation

### Prerequisites
* Python 3.12+
* `uv` package manager
* Local KoboldCpp running with `Qwen3-4B-Instruct`
* PostgreSQL instance
* MongoDB instance

### 1. Install Dependencies
```bash
uv sync --extra dev
```

### 2. Configure Environment Variables
Copy `.env.example` to `.env` and fill in your connection parameters:
```bash
cp .env.example .env
```
Example `.env`:
```text
KOBOLDCPP_BASE_URL=http://127.0.0.1:5001/v1
KOBOLDCPP_MODEL=qwen3-4b-instruct-2507

# Local Postgres: postgresql://username:password@localhost:5432/dbname
# Docker Postgres: postgresql://nl2user:nl2password@localhost:5432/nl2anyquery_db
POSTGRES_DSN=postgresql://username:password@localhost:5432/dbname
MONGODB_URI=mongodb://localhost:27017
MONGODB_DATABASE=nl2anyquery_demo

BM25_TOP_K=10
MODEL_TEMPERATURE=0.1
MODEL_MAX_TOKENS=1500
QUERY_TIMEOUT_SECONDS=10
MAX_RESULT_ROWS=1000
```

### Optional: Running PostgreSQL via Docker
Instead of running a local PostgreSQL installation, you can use the included `docker/Dockerfile.postgres` and `docker-compose.yml`:

1. **Start the PostgreSQL container**:
   ```bash
   docker compose up -d
   ```
   *(Note: If your local PostgreSQL is already listening on port 5432, stop local Postgres via `brew services stop postgresql` or change the port mapping in `docker-compose.yml` to `"5433:5432"`).*

2. **Update `.env`**:
   ```text
   POSTGRES_DSN=postgresql://nl2user:nl2password@localhost:5432/nl2anyquery_db
   ```
   *(Or port `5433` if you mapped to `5433:5432`).*

3. **Check container status**:
   ```bash
   docker compose ps
   ```

4. **Stop container when done**:
   ```bash
   docker compose down
   ```

### 3. Seed Databases
Populate realistic, repeatable synthetic relational and document datasets:
```bash
uv run seed-postgres
uv run seed-mongo
```

### 4. Verify Local Model Connection
```bash
uv run test-model
```

### 5. Initialize Canonical Semantic Schemas
Extract schema metadata and enrich table/column descriptions using KoboldCpp:
```bash
uv run init-schema --database postgres --describe
uv run init-schema --database mongo --describe
```

---

## 5. Running the Application

### Option A: Run FastAPI Backend
```bash
uv run run-api
```
* Interactive API Documentation: `http://127.0.0.1:8000/docs`
* Health Check: `http://127.0.0.1:8000/health`
* Schema Inspection: `http://127.0.0.1:8000/schema/postgres`

### Option B: Run Streamlit Frontend
```bash
uv run run-ui
```
Opens in your browser at `http://localhost:8501`.

---

## 6. Example Queries

### PostgreSQL Questions
* **Filter & Projections**: `"Show me all customers from Chicago"`
* **Count Aggregations**: `"How many customers do we have?"`
* **Grouping & Joins**: `"How many orders did each customer place?"`
* **Sum Aggregations**: `"What was the total revenue?"`
* **Sorting & Top-N**: `"Show the top 10 products by sales"`
* **Multi-table Joins**: `"Which products were ordered the most?"`

### MongoDB Questions
* **Document Filtering**: `"Show all customers located in Denver"`
* **Embedded Arrays**: `"Show orders containing more than one item"`
* **Aggregation Pipelines**: `"Which customers have placed the most orders?"`
* **Group & Sum**: `"What is the total sales amount per customer?"`

### Safety & Negative Tests
* `"Delete all customers"` $\rightarrow$ Prohibited by Guardrail (`REJECT`)
* `"Drop table orders"` $\rightarrow$ Prohibited by Guardrail (`REJECT`)
* `"Show me all astronauts"` $\rightarrow$ Zero-candidate graceful exit (`"I could not find any relevant tables..."`)
* `"Who are you?"` $\rightarrow$ Deterministic system identification (`BASIC`)

---

## 7. Testing

Run the full automated test suite (57 unit tests, no live model or DB required):
```bash
uv run pytest -v
```

---

## 8. Known POC Limitations & Future Extensions

* **Local Model Speed**: Qwen 3 4B reasoning models emit internal `<think>` tokens, requiring adequate token budgets and generation timeouts.
* **Complex Multi-Hop Joins**: In this POC, join expansion handles direct and 1-hop bridge tables; complex graph traversals belong to future iterations.
* **Read-Only Scope**: The system intentionally does not support write/mutation operations.
* **Extension Points**:
  - Cloud LLM providers can implement `ModelProvider` without touching pipeline logic.
  - Additional databases (e.g. MySQL, SQLite) can implement `DatabaseAdapter`.
