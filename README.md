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
├── ingestion/                  # Upstream schema ingestion (TOML canonical schemas)
├── postgres_embeddings.json    # 384-d table embeddings for PostgreSQL tables
├── query_processing/           # Core query-processing pipeline
│   ├── api/                    # FastAPI backend endpoints (/health, /query, /schema)
│   ├── core/                   # Configuration and text utilities
│   ├── models/                 # Pydantic models (PipelineResponse, QueryPlan, etc.)
│   ├── nlp/                    # spaCy linguistic analyzer
│   ├── pipeline/               # Stage components:
│   │   ├── guardrail.py        # Proactive intent classifier (READ_QUERY, BASIC, REJECT)
│   │   ├── semantic.py         # Semantic concept extraction SLM
│   │   ├── selector.py         # Table Selector SLM with sufficiency check
│   │   ├── expansion.py        # Foreign key & relationship expander
│   │   ├── planner.py          # Database-independent Query Planner SLM
│   │   ├── generators/         # Postgres and Mongo query generators
│   │   ├── validator.py        # Deterministic AST + SLM validator
│   │   ├── policy.py           # SQLGlot deterministic safety policy boundary
│   │   ├── executor.py         # Database query executor
│   │   ├── results.py          # Preview & CSV result processor
│   │   └── orchestrator.py     # Master thin coordinator with bounded retries
│   ├── providers/              # KoboldCpp SLM and Embedding providers
│   ├── prompts/                # Dedicated prompt templates
│   └── retrieval/              # Vector retriever and embedding store
├── frontend/                   # Streamlit web application
└── tests/                      # 106 automated unit and integration tests
```

---

## 4. Getting Started

### Prerequisites
* Python 3.11+
* `uv` package manager
* Local KoboldCpp instance running Qwen3-4B-Instruct (`http://127.0.0.1:5001/v1`)
* Local PostgreSQL instance with target database

### Running the API & Frontend
```bash
# Run FastAPI server
uv run run-api

# Run Streamlit frontend (in a separate terminal)
uv run run-ui
```

### Running Tests
```bash
uv run pytest -v
```
