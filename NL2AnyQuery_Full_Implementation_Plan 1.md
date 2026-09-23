# NL2AnyQuery — Full Implementation Plan

## 1. Overview

The NL2AnyQuery system is designed to convert natural-language questions into read-only SQL, NoSQL, or graph database queries.

The architecture is split into two major pipelines:

1. **Schema Intelligence Pipeline** — discovers, profiles, enriches, and indexes database metadata.
2. **Question-to-Query Pipeline** — validates the user question, analyzes its semantics, identifies relevant tables, plans the query, generates the query, validates it, executes it, and formats the result.

The V1 design intentionally avoids embeddings and vector databases. Relevant-table selection uses **BM25 retrieval followed by an SLM-based table selector**.

---

# 2. High-Level Architecture

```text
                         ┌──────────────────────┐
                         │    Connection String │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │ Database Type Detect │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │ Schema TOML Manager  │
                         └──────────┬───────────┘
                                    │
                  ┌─────────────────┴─────────────────┐
                  │                                   │
             TOML exists                         TOML missing
                  │                                   │
                  │                                   ▼
                  │                         Fetch DB Metadata
                  │                                   │
                  │                                   ▼
                  │                              Create TOML
                  │                                   │
                  │                                   ▼
                  │                         Schema Description SLM
                  │                                   │
                  └─────────────────┬─────────────────┘
                                    │
                                    ▼
                            Semantic TOML
                                    │
                                    ▼
                              BM25 Index
                                    │
                  ═══════════════════════════════════
                              QUERY FLOW
                  ═══════════════════════════════════
                                    │
                                    ▼
                              User Question
                                    │
                                    ▼
                            ┌────────────────┐
                            │  Guardrail SLM │
                            └───────┬────────┘
                                    │
                         ┌──────────┴──────────┐
                         │                     │
                      REJECT                 ALLOW
                         │                     │
                         ▼                     ▼
                  "Sorry, I can't       Semantic SLM
                   help with this."          │
                                              │
                                     ┌────────┴────────┐
                                     │                 │
                                  Subjective       Objective
                                     │                 │
                                     └────────┬────────┘
                                              │
                                              ▼
                                         spaCy
                                              │
                                       Noun + Verb
                                              │
                                              ▼
                                      BM25 Retrieval
                                              │
                                              ▼
                                    Candidate Tables
                                              │
                                              ▼
                                      Table Selector
                                            SLM
                                              │
                                              ▼
                                     Relevant Tables
                                              │
                                              ▼
                                   Relationship Expansion
                                              │
                                              ▼
                                    Relevant Columns
                                              │
                                              ▼
                                        Planner SLM
                                              │
                                              ▼
                                   Query Plan / IR
                                              │
                                              ▼
                                     Query Generator
                                              SLM
                                              │
                                              ▼
                                      Validation SLM
                                              │
                                ┌─────────────┴─────────────┐
                                │                           │
                              VALID                      INVALID
                                │                           │
                                ▼                           ▼
                           Execute DB              Retry Generation
                                │                    with feedback
                                │                           │
                                │                     max 3 attempts
                                │                           │
                                │                           ▼
                                │                      Manual fallback
                                │
                                ▼
                           Result Processor
                                │
                         ┌──────┴───────┐
                         │              │
                      <100 rows      >=100 rows
                         │              │
                         ▼              ▼
                    Pagination      10-row preview
                                       +
                                      CSV
                         │              │
                         └──────┬───────┘
                                ▼
                         Chart Analyzer
                                │
                                ▼
                         Recharts Config
```

---

# 3. Component Responsibilities

| Component | Responsibility | Technology |
|---|---|---|
| DB Detector | Identify DB from connection string | Python |
| Schema Manager | Create/update/read TOML | Python |
| Metadata Extractor | Get tables/columns/types/relationships | DB drivers |
| Schema Profiler | Describe tables/columns | SLM |
| Semantic Analyzer | Extract subjective/objective | SLM |
| Linguistic Analyzer | Nouns/verbs/entities | spaCy |
| Table Retriever | Candidate table retrieval | BM25 |
| Table Selector | Decide relevant tables | SLM |
| Relationship Resolver | Expand required related tables | Python |
| Planner | Create database-independent plan | SLM |
| Query Generator | Generate SQL/NoSQL/Cypher | SLM |
| Query Validator | Check semantic correctness | SLM |
| Policy Validator | Final read-only safety boundary | Python |
| Executor | Execute approved query | Python/DB |
| Result Processor | Pagination/CSV | Python |
| Chart Analyzer | Select visualization | Small SLM/Python |
| Chart Renderer | Produce Recharts config | Python |

---

# 4. Schema Initialization

The application receives:

```text
connection_string
schema_file_path
```

Example:

```text
postgresql://...
/schemas/customer_db.toml
```

## 4.1 Detect Database

Database type detection should be deterministic and should not require an SLM.

Examples:

```text
postgresql:// → PostgreSQL
mysql://      → MySQL
mongodb://    → MongoDB
neo4j://      → Neo4j
```

Return:

```json
{
  "database_type": "postgresql"
}
```

---

# 5. TOML as the Single Source of Truth

The TOML becomes the application's canonical **semantic schema**.

The actual database remains the execution source, but all query-generation components should use the TOML representation instead of repeatedly inspecting the database.

The TOML should contain:

- Database metadata
- Tables
- Columns
- Data types
- Table descriptions
- Column descriptions
- Relationships
- Optional sample/value metadata

Example:

```toml
[database]
type = "postgresql"

[tables.runs]
description = "Represents an execution of an AI agent."

[tables.runs.columns.id]
type = "uuid"
description = "Unique identifier for the run."

[tables.runs.columns.created_at]
type = "timestamp"
description = "Time when the run was created."

[tables.llm_calls]
description = "Records individual LLM calls associated with agent runs."

[tables.llm_calls.columns.id]
type = "uuid"
description = "Unique identifier for the LLM call."

[tables.llm_calls.columns.run_id]
type = "uuid"
description = "Run associated with the LLM call."

[tables.llm_calls.columns.model]
type = "text"
description = "LLM model used for the call."

[tables.llm_calls.columns.created_at]
type = "timestamp"
description = "Time when the LLM call occurred."
```

Relationships:

```toml
[[relationships]]
from_table = "llm_calls"
from_column = "run_id"
to_table = "runs"
to_column = "id"
type = "many_to_one"
```

---

# 6. Schema Description Pipeline

When the TOML does not exist:

1. Connect to the database.
2. Detect the database type.
3. Fetch metadata.
4. Create the initial TOML.
5. Split tables into groups of five.
6. Send each group to the Schema Description SLM.
7. Add descriptions to the TOML.
8. Build the BM25 index.

## Five Tables Per SLM Call

Do not send hundreds of tables in one SLM request.

Example:

```text
Tables 1–5   → SLM
Tables 6–10  → SLM
Tables 11–15 → SLM
...
```

Example input:

```text
Database:
PostgreSQL

Table:
llm_calls

Columns:
id UUID
run_id UUID
model TEXT
created_at TIMESTAMP

Sample values:
...
```

Expected output:

```json
{
  "table_description": "Records individual LLM invocations associated with agent runs.",
  "columns": {
    "model": "LLM model used for the invocation.",
    "run_id": "Run associated with the invocation."
  }
}
```

Merge this information into the TOML.

## Incremental Schema Updates

Do not regenerate descriptions on every application start.

Track metadata such as:

```toml
schema_version = "1"
last_updated = "..."
description_generated_at = "..."
```

Only profile:

- New tables
- New columns
- Explicitly refreshed schemas

---

# 7. BM25 Index

After the semantic TOML is ready, create a lightweight BM25 index.

No embeddings or vector database are required.

For every table, create a searchable document containing:

```text
TABLE:
llm_calls

DESCRIPTION:
Records individual LLM invocations associated with agent runs.

COLUMNS:
id
run_id
model
created_at

COLUMN DESCRIPTIONS:
model = LLM model used for the invocation
run_id = Run associated with the invocation
created_at = Time when the invocation occurred
```

This allows the system to search:

- Table names
- Table descriptions
- Column names
- Column descriptions

BM25 should be built once and refreshed whenever the TOML changes.

---

# 8. Guardrail SLM

Every user question first passes through the Guardrail SLM.

Its responsibility should be intentionally narrow:

> Determine whether the user is asking an allowed read-only database question.

Examples:

### Allowed

```text
Show me all employees from Chennai.
```

```json
{
  "allowed": true
}
```

### Rejected

```text
Delete all employees.
```

```json
{
  "allowed": false
}
```

### Rejected

```text
Change John's email address.
```

```json
{
  "allowed": false
}
```

The application responds:

```text
Sorry, I can't help with this.
```

## Basic System Questions

Basic conversational/system questions can be handled separately:

```text
Who are you?
What can you do?
What time is it?
What database are you connected to?
```

Architecture:

```text
User Question
     │
     ▼
Guardrail SLM
     │
 ┌───┼───────────┐
 │   │           │
READ BASIC    REJECT
 │   │           │
 ▼   ▼           ▼
Query System   "Sorry..."
Flow  Handler
```

---

# 9. Semantic Analyzer

For an allowed database question, call a Semantic Analyzer SLM.

Example:

> Give me the number of LLM calls per Run in 24 hours.

Output:

```json
{
  "subjective": [
    "LLM calls",
    "Run"
  ],
  "objective": [
    "number",
    "per Run",
    "in 24 hours"
  ]
}
```

The semantic analyzer should **not** convert these into SQL operations.

It should not produce:

```text
COUNT
GROUP BY
WHERE
```

Those are responsibilities of the Planner SLM.

---

# 10. spaCy Linguistic Analyzer

Run spaCy independently to extract linguistic information.

For:

> Give me the number of LLM calls per Run in 24 hours.

The analyzer may produce:

```json
{
  "noun": [
    "number",
    "LLM calls",
    "Run",
    "hours"
  ],
  "verb": [
    "give"
  ]
}
```

Exact output depends on the spaCy model and tokenization.

The separation is:

```text
Semantic SLM
    ↓
Subjective / Objective

spaCy
    ↓
Noun / Verb / Entities
```

These can then be combined into the query-analysis context:

```json
{
  "noun": ["LLM calls", "Run", "hours"],
  "verb": ["give"],
  "subjective": ["LLM calls", "Run"],
  "objective": ["number", "per Run", "in 24 hours"]
}
```

---

# 11. Relevant Table Selection

Do not send the complete schema to the Planner SLM.

Use a lightweight retrieval pipeline.

```text
Question
   │
   ├── Subjective
   ├── Objective
   ├── Nouns
   └── Entities
   │
   ▼
BM25 Search
   │
   ▼
Top 10–20 Candidate Tables
   │
   ▼
Table Selection SLM
   │
   ▼
Relevant Tables
   │
   ▼
Relationship Expansion
   │
   ▼
Relevant Columns
   │
   ▼
Final Relevant Schema
```

## Example

Question:

> Give me the number of LLM calls per Run in 24 hours.

BM25 might produce:

```text
llm_calls       0.92
runs            0.87
model_usage     0.68
agents          0.41
users           0.12
```

Send the top candidates to the Table Selection SLM.

Expected result:

```json
{
  "tables": [
    "llm_calls",
    "runs"
  ]
}
```

## Relationship Expansion

If the TOML contains:

```text
llm_calls.run_id → runs.id
```

the relationship resolver automatically includes `runs`.

This should be deterministic Python logic rather than another SLM call.

---

# 12. Planner SLM

The Planner SLM receives:

- User question
- Subjective
- Objective
- Nouns/entities
- Relevant tables
- Relevant columns
- Relationships
- Database type

Example:

```text
Question:
Give me the number of LLM calls per Run in 24 hours.

Database:
PostgreSQL

Relevant schema:
runs(id, created_at)
llm_calls(id, run_id, model, created_at)

Relationship:
llm_calls.run_id → runs.id
```

The planner should return a structured query plan.

Example:

```json
{
  "operation": "READ",
  "source": [
    "llm_calls",
    "runs"
  ],
  "aggregation": {
    "function": "COUNT",
    "column": "llm_calls.id"
  },
  "group_by": [
    "runs.id"
  ],
  "filters": [
    {
      "column": "llm_calls.created_at",
      "operator": ">=",
      "value": "CURRENT_TIME - 24 HOURS"
    }
  ]
}
```

Use a strict Pydantic model to validate the planner output.

---

# 13. Query Generator SLM

The Query Generator receives:

- Database type
- Relevant schema
- Relationships
- Query plan

For PostgreSQL, it might generate:

```sql
SELECT
    r.id AS run_id,
    COUNT(l.id) AS llm_call_count
FROM runs r
JOIN llm_calls l
    ON l.run_id = r.id
WHERE l.created_at >= NOW() - INTERVAL '24 hours'
GROUP BY r.id;
```

For MongoDB, generate an aggregation pipeline.

For Neo4j, generate Cypher.

The planner remains database-independent where possible; the Query Generator translates the plan into the target database language.

---

# 14. Validation SLM

The generated query then goes to the Validation SLM.

Provide:

```text
Original question
Relevant schema
Query plan
Generated query
Database type
```

The validator checks:

1. Is the query valid for the target database?
2. Are all tables valid?
3. Are all columns valid?
4. Are relationships valid?
5. Does the query satisfy the user's question?
6. Are filters correct?
7. Is aggregation correct?
8. Is grouping correct?
9. Is the query read-only?

Example:

```json
{
  "valid": false,
  "issues": [
    "The query uses llm_calls.timestamp but the available column is created_at."
  ],
  "suggestion": "Use llm_calls.created_at."
}
```

---

# 15. Query Retry Loop

Maximum three generation attempts.

```text
Query Generator
      │
      ▼
Validation SLM
      │
 ┌────┴──────────┐
 │               │
VALID          INVALID
 │               │
 ▼               ▼
Execute      Generator
                │
                ▼
             Validator
                │
                ▼
             Attempt 3
                │
                ▼
         Still Invalid
                │
                ▼
        Manual Verification
```

The validation feedback should be passed to the next generation attempt.

Example:

```text
Previous query:
...

Validation issues:
1. employee.full_name does not exist.
2. Available column is employee.name.

Generate a corrected query.
```

---

# 16. Final Deterministic Policy Check

Even though semantic validation is performed by an SLM, add one deterministic policy boundary before execution.

For SQL:

```text
Allowed:
SELECT

Rejected:
INSERT
UPDATE
DELETE
DROP
ALTER
TRUNCATE
CREATE
...
```

For MongoDB:

```text
Allowed:
find
aggregate

Rejected:
delete
update
replace
drop
```

For Neo4j:

```text
Allowed:
MATCH
RETURN
WITH
WHERE
ORDER BY
LIMIT
...

Rejected:
CREATE
DELETE
SET
REMOVE
MERGE
DROP
```

The SLM provides semantic validation, while this policy layer provides a final hard execution boundary.

---

# 17. Database Execution

Only execute after:

```text
Guardrail ✓
Semantic Analysis ✓
Table Selection ✓
Planning ✓
Query Generation ✓
Validation ✓
Read-only Policy ✓
```

Then execute against the database.

Recommended execution controls:

- Connection pooling
- Query timeout
- Maximum result size
- Read-only transaction/session where supported
- Database-specific resource limits

---

# 18. Result Processing

After execution:

```text
Query Result
     │
 ┌───┴──────────┐
 │              │
<100 rows      >=100 rows
 │              │
 ▼              ▼
Paginated      10-row preview
table             +
              CSV download
 │              │
 └──────┬───────┘
        ▼
   Chart Analyzer
        │
        ▼
   Recharts Config
```

## Less than 100 rows

Display the result with pagination.

## 100 or more rows

Display:

- First 10 rows
- CSV export containing the complete result

---

# 19. Chart Generation

The chart layer receives:

```json
{
  "columns": [
    {
      "name": "run_id",
      "type": "string"
    },
    {
      "name": "llm_call_count",
      "type": "integer"
    }
  ],
  "sample_data": [
    {
      "run_id": "run-001",
      "llm_call_count": 12
    }
  ]
}
```

A small SLM or deterministic rules can select the most suitable chart type.

Example:

```json
{
  "chart_type": "bar",
  "x_axis": "run_id",
  "y_axis": "llm_call_count"
}
```

The application should then generate the actual Recharts configuration deterministically.

---

# 20. Recommended End-to-End Flow

```text
                         ┌─────────────────┐
                         │ Connection String│
                         └────────┬────────┘
                                  ▼
                         Database Detection
                                  │
                                  ▼
                           TOML Schema Manager
                                  │
                       ┌──────────┴──────────┐
                       │                     │
                  Existing TOML          New TOML
                       │                     │
                       │               DB Metadata
                       │                     │
                       │                     ▼
                       │              Initial TOML
                       │                     │
                       │                     ▼
                       │              Schema SLM
                       │                5 tables
                       │                     │
                       └──────────┬──────────┘
                                  ▼
                            Semantic TOML
                                  │
                                  ▼
                              BM25 Index
                                  │
        ═══════════════════════════════════════════
                         USER QUERY
        ═══════════════════════════════════════════
                                  │
                                  ▼
                           Guardrail SLM
                                  │
                    ┌─────────────┼─────────────┐
                    │             │             │
                  READ          BASIC         REJECT
                    │             │             │
                    │             ▼             ▼
                    │       System Handler   Sorry...
                    │
                    ▼
               Semantic SLM
                    │
             Subjective/Objective
                    │
                    ▼
                  spaCy
                    │
               Noun / Verb
                    │
                    ▼
              BM25 Retrieval
                    │
                    ▼
           Top Candidate Tables
                    │
                    ▼
           Table Selection SLM
                    │
                    ▼
            Relevant Tables
                    │
                    ▼
          Relationship Expansion
                    │
                    ▼
           Relevant Columns
                    │
                    ▼
               Planner SLM
                    │
                    ▼
             Structured Plan
                    │
                    ▼
          Query Generator SLM
                    │
                    ▼
             Generated Query
                    │
                    ▼
             Validation SLM
                    │
             ┌──────┴──────┐
             │             │
           VALID         INVALID
             │             │
             ▼             ▼
       Policy Check     Retry ≤ 3
             │             │
             ▼             ▼
          Execute      Manual Review
             │
             ▼
        Result Processor
             │
       ┌─────┴─────┐
       ▼           ▼
    Table         CSV
       │
       ▼
   Chart Analyzer
       │
       ▼
 Recharts Config
```

---

# 21. SLM Responsibilities

The system should avoid giving one model too many responsibilities.

```text
Guardrail SLM
    → Is this an allowed read question?

Semantic SLM
    → What is the user talking about?
    → Subjective / Objective

Schema Description SLM
    → What do these tables/columns represent?

Table Selection SLM
    → Which candidate tables are relevant?

Planner SLM
    → What database operations are required?

Query Generator SLM
    → Convert plan into target query language

Validation SLM
    → Does the generated query actually satisfy the request?

Chart SLM
    → What visualization is appropriate?
```

The following should remain deterministic:

```text
Database detection
TOML management
Metadata extraction
BM25 retrieval
Relationship expansion
Pydantic validation
Read-only policy enforcement
Database execution
Pagination
CSV generation
Recharts configuration generation
```

---

# 22. Recommended V1 Technology Stack

```text
Language:
Python 3.12+

API:
FastAPI

NLP:
spaCy

Retrieval:
BM25

Schema:
TOML

Structured outputs:
Pydantic

SLM:
Qwen3-4B-Instruct-2507 as the initial general-purpose SLM

Query generation:
Benchmark Qwen3-4B against a 7B–8B coder/instruction model

Databases:
Database-specific Python drivers

Frontend:
Existing application / React

Charts:
Recharts
```

The SLM choice should ultimately be validated against your own NL2SQL/NL2NoSQL/NL2Graph test set rather than selected solely from general benchmarks.

---

# 23. Suggested Development Phases

## Phase 1 — Database & Schema

- Connection-string parsing
- Database-type detection
- Database metadata adapters
- TOML schema generation
- TOML read/write utilities
- Five-table schema profiling
- Schema description SLM
- Schema refresh/versioning

## Phase 2 — Retrieval

- TOML-to-search-document conversion
- BM25 index
- Keyword normalization
- Candidate table retrieval
- Relationship metadata
- Relationship expansion
- Table Selection SLM

## Phase 3 — Question Understanding

- Guardrail SLM
- Basic/system-question handling
- spaCy integration
- Noun extraction
- Verb extraction
- Semantic SLM
- Subjective/objective extraction
- Unified question-analysis Pydantic model

## Phase 4 — Planning

- Planner prompt
- Query-plan Pydantic schema
- SQL planning
- NoSQL planning
- Graph planning
- Table/column validation against TOML

## Phase 5 — Query Generation

- SQL generator
- NoSQL generator
- Graph query generator
- Structured output
- Query normalization

## Phase 6 — Validation

- Validation SLM
- Semantic validation
- Schema validation
- Query-database compatibility validation
- Read-only policy
- Three-attempt retry loop
- Manual fallback

## Phase 7 — Execution & Results

- Database execution layer
- Timeouts
- Read-only sessions
- Result pagination
- CSV generation
- Large-result handling

## Phase 8 — Visualization

- Result-type detection
- Chart selection
- Recharts configuration
- Frontend rendering

## Phase 9 — Evaluation

Build a test dataset covering:

```text
Simple retrieval
Aggregation
Filtering
Multiple filters
Grouping
Sorting
Joins
Nested queries
Date/time questions
Ambiguous questions
Synonyms
Different English phrasing
Wrong questions
Write questions
Unknown tables
Unknown columns
Cross-table questions
No-result queries
Large-result queries
```

Measure at minimum:

```text
Guardrail accuracy
Subjective/objective extraction accuracy
Table Recall@K
Table selection accuracy
Plan accuracy
Query execution accuracy
Validation accuracy
Retry success rate
Latency
Token consumption
```

The most important retrieval metric is:

> **Did the candidate table set contain every table required to produce the correct answer?**

This lets you determine whether BM25 is sufficient before introducing embeddings.

---

# 24. Final Design Principle

The architecture should follow this rule:

```text
SLM = understanding and reasoning
Python = orchestration and deterministic logic
TOML = semantic schema source of truth
BM25 = fast schema retrieval
Database = actual source of data
```

In particular:

```text
Do NOT:
Question → giant SLM → SQL

Instead:
Question
   ↓
Guardrail
   ↓
Semantic understanding
   ↓
Lightweight schema retrieval
   ↓
Relevant schema
   ↓
Planning
   ↓
Query generation
   ↓
Validation
   ↓
Deterministic policy
   ↓
Execution
```

This keeps the system modular, debuggable, and much easier to evaluate as you expand from SQL to NoSQL and graph databases.
