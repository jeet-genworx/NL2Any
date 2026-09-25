# NL2AnyQuery — Natural Language to Database Query POC

> **Proof-of-Concept Notice**: This repository is a lightweight, local proof-of-concept demonstrating a deterministic, modular pipeline that translates natural language questions into safe, read-only PostgreSQL queries using small language models (SLMs).
> 
> *Note on Database Scope*: The current refactored pipeline is **PostgreSQL-focused**. Legacy MongoDB components are preserved for backwards compatibility, with MongoDB vector adaptation slated for a later step.

---

## 1. Architecture Overview

```text
Database Ingestion Layer (Upstream)
      │
      ▼
postgres_embeddings.json (384-d vectors generated with all-MiniLM-L6-v2)
      │
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
┌─────────────────────────────────────────────────────────────┐
│ Concurrent Processing:                                      │
│  ├── Query Embedding SLM     (all-MiniLM-L6-v2)             │
│  ├── spaCy Linguistic Parser (nouns, verbs, named entities) │
│  └── Semantic Analysis SLM   (subjective/objective concepts)│
└─────────────────────────────┬───────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│ Embedding Table Retrieval   (Cosine Similarity > 0.80, ≤ 15)│
└─────────────────────────────┬───────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│ Table Selector SLM          (Bounded sufficiency retry ≤ 3) │
└─────────────────────────────┬───────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│ Relationship Expander       (Deterministic foreign keys)    │
└─────────────────────────────┬───────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│ Planner SLM                 (Database-independent QueryPlan)│
└─────────────────────────────┬───────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│ PostgreSQL Generator        (Translates QueryPlan into SQL) │
└─────────────────────────────┬───────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│ Validator SLM + Fast Checks (Deterministic + SLM validation)│
└──────────────┬──────────────────────────────┬───────────────┘
               │                              │
         [SYNTAX_ERROR /               [PLANNER_ERROR]
        GENERATION_ERROR]                     │
               │                        Retry Planner
         Retry Generator               (Targeted, ≤ 3)
         (Targeted, ≤ 3)                      │
               │                              │
               └──────────────┬───────────────┘
                              │
                           [VALID] (or [UNSAFE] ──► Immediate Halt)
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│ Deterministic Safety Policy (SQLGlot single-statement SELECT│
│                             AST read-only enforcement)      │
└─────────────────────────────┬───────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│ PostgreSQL Execution        (Query timeouts & row limits)   │
└─────────────────────────────┬───────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│ Result Processor            (10-row preview, CSV for ≥ 100) │
└─────────────────────────────┬───────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│ FastAPI & Streamlit UI      (11 dedicated tracing expanders)│
└─────────────────────────────────────────────────────────────┘
```

---

## 2. Key Design Rationales

* **Why `all-MiniLM-L6-v2` for Embeddings?**
  `all-MiniLM-L6-v2` is a lightweight, efficient sentence-transformer model producing compact 384-dimensional dense vectors. It provides excellent semantic matching between colloquial user questions and structured table descriptions without demanding significant VRAM or GPU compute. It runs locally alongside the KoboldCpp SLM.
* **Why Embedding Retrieval for Tables, but Schema Metadata for Planning & Generation?**
  Dense vector embeddings are optimal for *table relevance discovery*—rapidly identifying candidate tables out of dozens of candidates based on semantic intent and table descriptions, without cluttering the prompt context.
  However, vectors cannot represent the precise structural constraints of a database (exact column data types, foreign key constraints, nullable flags, compound keys). Once relevant tables are selected, the canonical schema metadata (from TOML) provides the exact schema definition required by the Expander, Planner, and Query Generator to craft accurate queries without hallucinations.
* **Why an SLM (Qwen3-4B-Instruct via KoboldCpp)?**
  A local 4B model running on consumer hardware is fast, private, and capable of structured JSON reasoning when provided with tightly bounded contexts (groups of $\le 5$ objects, only relevant schema columns). Dedicated single-responsibility prompts avoid confusion and hallucinations.
* **Why spaCy Linguistic Analysis in Parallel?**
  Extracts linguistic parts-of-speech (nouns, verbs, named entities like cities, dates, and names) deterministically in milliseconds, complementing the SLM semantic analysis and assisting table selection.
* **Why Targeted Retries?**
  Rather than blindly restarting the entire pipeline upon validation failure, the Validator explicitly classifies errors:
  - `SYNTAX_ERROR` / `GENERATION_ERROR`: Retries only the Query Generator with error feedback, preserving the existing `QueryPlan`.
  - `PLANNER_ERROR`: Retries from the Query Planner with feedback, then regenerates and revalidates.
  - `UNSAFE`: Halts immediately with 0 retries.
  - `MAX_RETRIES = 3`: Strictly bounded to prevent infinite loops.
* **Why Deterministic Safety Boundaries?**
  The SLM is **never** the final safety boundary. Generated SQL is parsed into an AST via `sqlglot` to verify that only a single `SELECT` statement is present, rejecting `INSERT`, `UPDATE`, `DELETE`, `DROP`, `ALTER`, and multi-statement queries before reaching the database driver.

---

## 3. Project Structure

```text
nl2anyquery/
├── .env.example                  # Generic environment variable template
├── .gitignore                    # Ignores .env and build caches
├── pyproject.toml                # Project dependencies and console entrypoints
├── README.md                     # Complete documentation
├── docker-compose.yml            # postgres, mongo, seed, api, frontend services
├── postgres_embeddings.json      # Precomputed 384-d table embeddings (committed, used by retrieval/vector.py)
├── docker/
│   ├── Dockerfile.app            # Image shared by seed/api/frontend containers
│   ├── Dockerfile.postgres       # Postgres container definition
│   └── seed.sh                   # Runs seed-postgres + seed-mongo on startup
│
├── ingestion/                     # Getting schema & data INTO the system
│   ├── pipeline.py                 # Full orchestrator: metadata → graph/MST → SLM descriptions → embeddings
│   ├── databases/                 # Deterministic database discovery adapters
│   │   ├── base.py
│   │   ├── detector.py
│   │   ├── postgres/               # information_schema-based extraction (universal, no hardcoding)
│   │   ├── mongo/                  # Document-sampling-based extraction
│   │   └── seed/                  # Synthetic demo data generators
│   ├── schema/
│   │   ├── manager.py             # get_default_schema_path(), shared with query_processing
│   │   ├── toml_store.py          # Canonical schema TOML serialization
│   │   ├── graph.py                # Nodes/edges graph view (tables + columns + relationships)
│   │   ├── mst.py                  # Minimum spanning tree/forest reduction of the graph
│   │   ├── describe.py             # Batched SLM table descriptions (MST or plain graph order)
│   │   └── embed.py                # Embeds descriptions via the local MiniLM model
│   └── schemas/                   # Canonical schema/graph/MST TOML + descriptions JSON (generated, not committed)
│
├── query_processing/              # The NL → safe query pipeline
│   ├── api/                       # FastAPI web server (GET /health, POST /query, GET /schema/{db}, POST /ingest/{db})
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
│   │   ├── selector.py            # Candidate selection with strict TOML checks + targeted retries
│   │   ├── expansion.py           # Foreign key & bridge table expansion
│   │   ├── planner.py             # Syntax-free semantic query planning
│   │   ├── generators/            # Postgres SQL & typed MongoQuery generators
│   │   ├── validator.py           # AST schema verification + SLM validation
│   │   ├── policy.py              # Non-negotiable read-only safety boundary
│   │   ├── executor.py            # Safe DB driver execution with timeouts
│   │   ├── results.py             # Bounded preview & CSV generation
│   │   └── orchestrator.py        # Master pipeline runner with bounded, targeted retries
│   ├── providers/
│   │   ├── model/                 # KoboldCpp chat + embedding provider used by the ingestion pipeline
│   │   └── embedding/             # KoboldCpp embedding provider used live by retrieval/vector.py
│   ├── retrieval/                 # Embedding-based schema search
│   │   ├── vector.py               # Live retriever: reads the committed postgres_embeddings.json
│   │   ├── store.py                # Embedding store backing vector.py
│   │   └── semantic.py             # Alternate retriever reading query_processing/embeddings/ (ingestion pipeline output; not currently wired into the orchestrator)
│   ├── prompts/                   # Focused, single-responsibility prompt templates
│   │   ├── guardrail.txt
│   │   ├── semantic_analysis.txt
│   │   ├── table_selection.txt
│   │   ├── planner.txt
│   │   ├── postgres_query_generator.txt
│   │   ├── mongo_query_generator.txt
│   │   ├── validator.txt
│   │   └── table_description.txt  # Batched per-table SLM description prompt (ingestion pipeline)
│   ├── embeddings/                 # table_name -> vector JSON from the ingestion pipeline (generated, not committed)
│   └── cli.py                     # Unified `uv run` console entrypoints
│
├── frontend/
│   └── app.py                     # Streamlit UI with pipeline stage inspection + database ingestion controls
│
└── tests/                         # Comprehensive automated unit tests
```

---

## 4. Getting Started

### Prerequisites
* Python 3.11+
* `uv` package manager
* Local KoboldCpp running with `Qwen3-4B-Instruct` **and** an embedding model (`all-MiniLM-L6-v2`) loaded via `--embeddingsmodel`, e.g.:
  ```bash
  ./koboldcpp --model Qwen3-4B-Q4_K_M.gguf --embeddingsmodel all-MiniLM-L6-v2-Q8_0.gguf --contextsize 8192 --port 5001
  ```
* PostgreSQL instance
* MongoDB instance (optional -- the live retrieval/query pipeline is currently PostgreSQL-focused)

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

RETRIEVAL_TOP_K=10
MODEL_TEMPERATURE=0.1
MODEL_MAX_TOKENS=1500
QUERY_TIMEOUT_SECONDS=10
MODEL_TIMEOUT_SECONDS=300
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
Extract authoritative table/column metadata straight from the database adapters and write the canonical TOML that query processing reads from:
```bash
uv run init-schema --database postgres
uv run init-schema --database mongo
```

### 6. Ingest: Generate Table Descriptions & Embeddings
**Required before querying** -- the query pipeline's retrieval stage reads precomputed table embeddings, not raw keywords. Run the full ingestion pipeline (metadata → graph/MST → SLM descriptions → embeddings) for each database:
```bash
uv run ingest-schema --database postgres
uv run ingest-schema --database mongo
```
This is also available as an "Initialize Database" button in the Streamlit frontend sidebar, so you can re-run it any time the underlying database schema changes.

---

## 5. Running the Application

### Option A: Run FastAPI Backend
```bash
uv run run-api
```
* Interactive API Documentation: `http://127.0.0.1:8000/docs`
* Health Check: `http://127.0.0.1:8000/health`
* Schema Inspection: `http://127.0.0.1:8000/schema/postgres`
* Ingestion (metadata → graph/MST → descriptions → embeddings): `POST /ingest/postgres` or `POST /ingest/mongo`

### Option B: Run Streamlit Frontend
```bash
uv run run-ui
```
Opens in your browser at `http://localhost:8501`. Use the sidebar's **Initialize Database** button before asking questions against a database for the first time, or after its schema changes.

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

Run the full automated test suite (no live model or DB required -- SLM/embedding calls are mocked):
```bash
uv run pytest -v
```
