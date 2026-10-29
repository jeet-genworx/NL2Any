"""Streamlit frontend for NL2AnyQuery."""

import json
import os
import time
import httpx
import pandas as pd
import streamlit as st

API_BASE_URL = os.environ.get("API_BASE_URL", "http://127.0.0.1:8000")

st.set_page_config(
    page_title="NL2AnyQuery",
    page_icon="🔍",
    layout="wide",
)

st.title("🔍 NL2AnyQuery")
st.caption("Proof-of-Concept: Natural Language to Safe Database Query Pipeline")


DEFAULT_DATABASES = [
    {"key": "postgres", "label": "PostgreSQL (Relational)", "database_type": "postgresql"},
    {"key": "mongo", "label": "MongoDB (Document)", "database_type": "mongodb"},
    {"key": "finops", "label": "FinOps (PostgreSQL)", "database_type": "postgresql"},
]

DEFAULT_PROVIDERS = [
    {"id": "koboldcpp", "name": "KoboldCpp (Local SLM)", "configured": True},
    {"id": "huggingface", "name": "Hugging Face", "configured": False},
    {"id": "gemini", "name": "Google Gemini", "configured": False},
]


@st.cache_data(ttl=60)
def fetch_providers(api_url: str) -> list[dict]:
    """Fetch available model providers and configuration status from API."""
    try:
        resp = httpx.get(f"{api_url}/providers", timeout=5.0)
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return list(DEFAULT_PROVIDERS)


@st.cache_data(ttl=60)
def fetch_databases(api_url: str) -> list[dict]:
    """Fetch the selectable databases from the API.

    Returns the built-in targets followed by every database saved from a
    connection string, so the picker stays in step with the backend registry
    rather than hardcoding a list. Falls back to the built-in names when the
    backend is not up yet.

    Each row carries `configured` (a connection string is set) and `ingested`
    (embeddings are already on disk), which the picker turns into a status
    suffix so a database says why it is unusable, or that selecting it is free,
    before anyone clicks it.
    """
    try:
        resp = httpx.get(f"{api_url}/databases", timeout=5.0)
        resp.raise_for_status()
        return resp.json()["databases"]
    except Exception:
        return list(DEFAULT_DATABASES)


def describe_database(row: dict) -> str:
    """The picker's label for one database: its name plus its readiness."""
    if not row.get("configured", True):
        return f"{row['label']} - not configured"
    if row.get("ingested"):
        return f"{row['label']} - ready"
    return f"{row['label']} - needs ingestion"


def add_connection(api_url: str, connection_string: str, label: str) -> tuple[bool, object]:
    """Save a connection string with the API. Returns (ok, response_or_error)."""
    try:
        resp = httpx.post(
            f"{api_url}/connections",
            json={
                "connection_string": connection_string.strip(),
                "label": label.strip() or None,
            },
            # Generous: the API proves the connection works before saving it,
            # and an unreachable host takes as long as the driver's own timeout.
            timeout=30.0,
        )
        resp.raise_for_status()
        return True, resp.json()
    except httpx.ConnectError:
        return False, (
            f"Could not connect to FastAPI backend at {api_url}. "
            "Please make sure it is running via `uv run run-api`."
        )
    except httpx.HTTPStatusError as err:
        try:
            detail = err.response.json().get("detail", err.response.text)
        except Exception:
            detail = err.response.text
        return False, detail
    except Exception as err:
        return False, str(err)


def run_ingestion(database: str, use_mst: bool, api_url: str, force: bool = False) -> tuple[bool, object]:
    """Run the ingestion pipeline for a database. Returns (ok, summary_or_error)."""
    try:
        resp = httpx.post(
            f"{api_url}/ingest/{database}",
            params={"use_mst": use_mst, "force": force},
            timeout=None,
        )
        resp.raise_for_status()
        return True, resp.json()
    except httpx.ConnectError:
        return False, (
            f"Could not connect to FastAPI backend at {api_url}. "
            "Please make sure it is running via `uv run run-api`."
        )
    except httpx.HTTPStatusError as err:
        try:
            detail = err.response.json().get("detail", err.response.text)
        except Exception:
            detail = err.response.text
        return False, f"Ingestion failed: {detail}"
    except Exception as err:
        return False, f"Ingestion failed: {err}"


STAGE_CONFIG: dict[str, tuple[str, str]] = {
    "spelling_check": ("🔤", "Spelling Check"),
    "question_guardrail": ("🛡️", "Safety & Intent Guardrail"),
    "feedback_guardrail": ("🛡️", "Feedback Guardrail"),
    "table_selection": ("📋", "Table & Schema Selection"),
    "query_generation": ("⚙️", "Query Generation"),
    "query_validation": ("🔍", "Query Validation"),
    "policy_check": ("🔒", "Safety Policy Verification"),
    "query_execution": ("⚡", "Database Query Execution"),
}


def execute_rerun(
    api_url: str,
    run_id: str,
    starting_stage: str,
    feedback_payload: dict[str, str],
) -> None:
    """Execute partial pipeline rerun from a specified stage with stage progress updates."""
    start_time = time.perf_counter()
    status_label = f"Rerunning pipeline from {starting_stage.replace('_', ' ').title()}..."

    with st.status(f"🔄 {status_label}", expanded=True) as status_box:
        pipeline_data = None
        stream_error = None

        try:
            with httpx.stream(
                "POST",
                f"{api_url}/rerun/stream",
                json={
                    "run_id": run_id,
                    "starting_stage": starting_stage,
                    "feedback": feedback_payload,
                },
                timeout=None,
            ) as resp:
                resp.raise_for_status()
                for line in resp.iter_lines():
                    if not line or not line.startswith("data: "):
                        continue
                    raw_data = line[len("data: "):].strip()
                    try:
                        evt = json.loads(raw_data)
                    except Exception:
                        continue

                    evt_type = evt.get("event")
                    stage = evt.get("stage", "")
                    icon, name = STAGE_CONFIG.get(stage, ("⏳", stage.replace("_", " ").title()))

                    if evt_type == "stage_start":
                        status_box.update(
                            label=f"⏳ Running: {name} (rerun)...",
                            state="running",
                            expanded=True,
                        )
                    elif evt_type == "stage_done":
                        summary = evt.get("summary") or {}
                        extra = ""
                        if stage == "table_selection" and summary.get("selected_objects"):
                            objs = summary["selected_objects"]
                            extra = f" (Selected: `{', '.join(objs)}`)"
                        elif stage == "query_validation":
                            extra = " (Valid query)" if summary.get("is_valid") else " (Retrying or invalid)"
                        elif stage == "policy_check":
                            extra = " (Approved)" if summary.get("passed") else " (Rejected by policy)"

                        st.write(f"{icon} **{name}** done ✅{extra}")
                    elif evt_type == "complete":
                        pipeline_data = evt.get("data")
                        status_box.update(
                            label="✅ Rerun completed!",
                            state="complete",
                            expanded=False,
                        )
                    elif evt_type == "error":
                        status_code = evt.get("status_code", 500)
                        detail = evt.get("detail", "Error during rerun")
                        stream_error = (status_code, detail)
                        status_box.update(
                            label=f"❌ Error: {detail}",
                            state="error",
                            expanded=True,
                        )
                        break

            elapsed_time = time.perf_counter() - start_time
            if stream_error:
                code, detail = stream_error
                if code == 404:
                    st.session_state["query_data"] = None
                    st.session_state["query_execution_time"] = None
                    st.session_state["query_error"] = (
                        "This run has expired or does not exist. Please ask your question again."
                    )
                    st.session_state["feedback_warning"] = None
                elif code == 400:
                    st.session_state["feedback_warning"] = f"🛡️ {detail}"
                elif code >= 500:
                    st.session_state["query_error"] = f"❌ Backend Server Error (HTTP {code}): {detail}"
                else:
                    st.session_state["query_error"] = f"❌ Pipeline Error: {detail}"
            elif pipeline_data:
                st.session_state["query_data"] = pipeline_data
                st.session_state["query_execution_time"] = elapsed_time
                st.session_state["query_error"] = None
                st.session_state["feedback_warning"] = None

        except httpx.ConnectError:
            st.session_state["query_error"] = (
                f"Could not connect to FastAPI backend at {api_url}. "
                "Please make sure it is running via `uv run run-api`."
            )
        except httpx.HTTPStatusError as err:
            try:
                detail = err.response.json().get("detail", err.response.text)
            except Exception:
                detail = err.response.text
            if err.response.status_code == 404:
                st.session_state["query_data"] = None
                st.session_state["query_execution_time"] = None
                st.session_state["query_error"] = (
                    "This run has expired or does not exist. Please ask your question again."
                )
                st.session_state["feedback_warning"] = None
            elif err.response.status_code == 400:
                st.session_state["feedback_warning"] = f"🛡️ {detail}"
            elif err.response.status_code >= 500:
                st.session_state["query_error"] = f"❌ Backend Server Error (HTTP {err.response.status_code}): {detail}"
            else:
                st.session_state["query_error"] = f"❌ Pipeline Error: {detail}"
        except Exception as err:
            st.session_state["query_error"] = f"Error during rerun: {err}"

        st.rerun()




# Sidebar
with st.sidebar:
    st.header("Settings")
    # Read before anything that calls the API, since the connection form and the
    # database list both have to go to whichever backend is named here.
    api_url = st.text_input("FastAPI Backend URL", value=API_BASE_URL)

    # Adding a database is handled before the picker is drawn: a successful
    # connect selects the new database, and Streamlit does not allow a widget's
    # value to be set after that widget has been created in the same run.
    #
    # The result of the last attempt is read before the expander is built, so
    # the expander can open itself to show it -- a connect is followed by a
    # rerun, and a message rendered inside a collapsed expander is a message
    # nobody sees.
    notice = st.session_state.get("connection_notice")
    with st.expander("➕ Add a database", expanded=bool(notice)):
        st.caption(
            "Paste a PostgreSQL DSN or MongoDB URI. The engine is detected from "
            "the string, and the connection is tested before it is saved."
        )
        new_connection = st.text_input(
            "Connection string",
            type="password",
            placeholder="postgresql://user:password@host:5432/dbname",
            help=(
                "Also accepts mongodb:// and mongodb+srv:// URIs, and libpq's "
                "'host=... dbname=...' form. Stored on the backend; it is not "
                "sent back to this page again."
            ),
        )
        new_label = st.text_input(
            "Label (optional)",
            placeholder="Named after the database when left blank",
        )
        if st.button("Connect", use_container_width=True, type="primary"):
            if not new_connection.strip():
                st.session_state["connection_notice"] = ("error", "Enter a connection string first.")
            else:
                with st.spinner("Testing the connection..."):
                    ok, payload = add_connection(api_url, new_connection, new_label)
                if not ok:
                    st.session_state["connection_notice"] = ("error", str(payload))
                else:
                    # A database already saved comes back as the existing entry
                    # rather than a duplicate, with its ingested artifacts
                    # intact -- so selecting it below reuses the embeddings
                    # instead of running the pipeline again.
                    if not payload.get("already_saved"):
                        st.session_state["connection_notice"] = (
                            "success",
                            f"Saved '{payload['label']}' ({payload['database_type']}).",
                        )
                    elif payload.get("ingested"):
                        st.session_state["connection_notice"] = (
                            "info",
                            f"'{payload['label']}' is already set up — "
                            "loading its existing embeddings, not re-ingesting.",
                        )
                    else:
                        st.session_state["connection_notice"] = (
                            "info",
                            f"'{payload['label']}' is already saved, but has not "
                            "been ingested yet. Ingesting it now.",
                        )
                    # The picker is cached for a minute; drop it so the new
                    # database is in the options this rerun builds.
                    fetch_databases.clear()
                    st.session_state["db_choice"] = payload["key"]
                    st.rerun()

        # Read at the top of this block, before the expander decided whether to
        # open. Left in place rather than cleared, so the outcome of the last
        # attempt stays readable until the next one replaces it.
        if notice:
            level, message = notice
            getattr(st, level)(message)

    # Fetched from the API so the picker stays in step with the target registry
    # rather than hardcoding a list here; falls back to the built-in targets if
    # the backend is not up yet. Saved connections are listed alongside the
    # configured databases, each labelled with whether it is ready to query.
    db_rows = fetch_databases(api_url)
    db_options = [row["key"] for row in db_rows]
    db_labels = {row["key"]: describe_database(row) for row in db_rows}
    db_types = {row["key"]: row.get("database_type", "postgresql") for row in db_rows}

    # A database removed from the backend between reruns would leave a stale
    # selection pointing at a key that no longer exists.
    if st.session_state.get("db_choice") not in db_options:
        st.session_state.pop("db_choice", None)

    db_choice = st.selectbox(
        "Database",
        options=db_options,
        format_func=lambda key: db_labels.get(key, key),
        key="db_choice",
    )
    db_type = db_types.get(db_choice, "postgresql")

    providers = fetch_providers(api_url)
    provider_ids = [p["id"] for p in providers]
    provider_labels = {
        p["id"]: p["name"] if p.get("configured", True) else f"{p['name']} (unconfigured)"
        for p in providers
    }

    st.markdown("#### Model Providers (Per-Stage)")
    stage1_choice = st.selectbox(
        "Stage 1: Table Selection & Guardrail",
        options=provider_ids,
        format_func=lambda key: provider_labels.get(key, key),
        index=0,
        help="Select model provider for Guardrail classification and Table Selection.",
        key="stage1_choice",
    )
    stage2_choice = st.selectbox(
        "Stage 2: Query Generation",
        options=provider_ids,
        format_func=lambda key: provider_labels.get(key, key),
        index=0,
        help="Select model provider for generating queries (PostgreSQL / MongoDB).",
        key="stage2_choice",
    )
    stage3_choice = st.selectbox(
        "Stage 3: Validation",
        options=provider_ids,
        format_func=lambda key: provider_labels.get(key, key),
        index=0,
        help="Select model provider for query syntax and safety validation.",
        key="stage3_choice",
    )

    # Trigger KV cache warm-up ONLY when Stage 1 table selection provider is koboldcpp:
    # Triggers when database changes (while on koboldcpp) OR when stage 1 is switched to koboldcpp.
    s1_is_kobold = stage1_choice == "koboldcpp"
    db_changed = bool(db_choice and st.session_state.get("_last_selected_db") != db_choice)
    s1_changed_to_kobold = (
        s1_is_kobold
        and st.session_state.get("_last_stage1_provider") != "koboldcpp"
    )
    if s1_is_kobold and (db_changed or s1_changed_to_kobold):
        st.session_state["_last_selected_db"] = db_choice
        st.session_state["_last_stage1_provider"] = stage1_choice
        try:
            httpx.post(f"{api_url}/databases/{db_choice}/select?stage1_provider={stage1_choice}", timeout=1.0)
        except Exception:
            pass
    else:
        if db_changed:
            st.session_state["_last_selected_db"] = db_choice
        st.session_state["_last_stage1_provider"] = stage1_choice

    st.divider()
    st.markdown("### Database Ingestion")
    use_mst = st.checkbox(
        "Use MST for contextual table grouping",
        value=True,
        help=(
            "On: tables are described in minimum-spanning-tree order, with only "
            "the reduced (cycle-free) relationships as context. Off: tables are "
            "described straight from the full schema graph instead. MongoDB "
            "always uses the graph, since it has no foreign keys."
        ),
    )
    ingest_btn = st.button(
        "🔄 Re-run Ingestion",
        use_container_width=True,
        help="Ingestion runs automatically when you select a database. Use this to "
        "re-run it after the underlying schema changes, or after toggling the MST option.",
    )

    # Ingestion runs automatically whenever the selected database changes (including
    # first load), so the embeddings query processing retrieves against always match
    # the chosen database. Keyed on db_choice in session_state so it fires on actual
    # selection changes only -- Streamlit reruns this script on every widget
    # interaction, and re-ingesting on each of those would be needlessly expensive.
    if st.session_state.get("ingested_db") != db_choice or ingest_btn:
        # A database that has been ingested before returns in milliseconds with
        # its existing artifacts, so this spinner covers both cases.
        with st.spinner(f"Preparing {db_labels.get(db_choice, db_choice)}..."):
            # The button means "do it again"; an automatic run on selection
            # reuses existing embeddings instead of re-ingesting.
            ok, payload = run_ingestion(db_choice, use_mst, api_url, force=ingest_btn)
        st.session_state["ingested_db"] = db_choice if ok else None
        st.session_state["ingestion_summary"] = payload if ok else None
        st.session_state["ingestion_error"] = None if ok else payload
        st.session_state["query_data"] = None
        st.session_state["query_execution_time"] = None
        st.session_state["query_error"] = None

    # Rendered from session_state so the status survives later reruns.
    if st.session_state.get("ingestion_error"):
        st.error(st.session_state["ingestion_error"])
    elif st.session_state.get("ingestion_summary"):
        summary = st.session_state["ingestion_summary"]
        # `skipped` says the pipeline did not run: the embeddings were already
        # on disk and were loaded as they were. Worth saying plainly, so a fast
        # response does not look like ingestion silently did nothing.
        if summary.get("skipped"):
            st.success(
                f"Loaded existing embeddings: {summary['embedding_count']} tables "
                f"({summary['embedding_dimensions']}-dim). Not re-ingested."
            )
        else:
            st.success(
                f"Ingested {summary['table_count']} tables → "
                f"{summary['description_count']} descriptions → "
                f"{summary['embedding_count']} embeddings "
                f"({summary['embedding_dimensions']}-dim, MST used: {summary['used_mst']})."
            )
        st.caption(f"Query retrieval uses: `{summary['embeddings_path']}`")
        with st.expander("Ingestion details"):
            st.json(summary)

    st.divider()
    st.markdown("### Example Questions")
    # Written against the two seeded demo databases, so they are only offered
    # for those. Any other database -- FinOps, or one added from a connection
    # string -- has its own tables, and canned questions about customers and
    # orders would be misleading rather than helpful.
    sample_questions: list[str] = []
    if db_choice == "postgres":
        sample_questions = [
            "Show me all customers from Chicago",
            "How many customers do we have?",
            "Which products were ordered the most?",
            "What was the total revenue?",
            "Which city has the most customers?",
            "Show the top 10 products by sales",
            "How many orders did each customer place?",
            "Delete all customers",  # Negative test
            "Show me all astronauts",  # Unknown domain
        ]
    elif db_choice == "mongo":
        sample_questions = [
            "Which customers have placed the most orders?",
            "What are the top-selling products?",
            "Show orders containing more than one item",
            "What is the total sales amount per customer?",
            "Show all customers located in Denver",
            "Drop the orders collection",  # Negative test
            "Find all spaceships",  # Unknown domain
        ]

    if sample_questions:
        for q in sample_questions:
            if st.button(q, key=f"btn_{q}"):
                st.session_state["question_input"] = q
                st.session_state["query_data"] = None
                st.session_state["query_execution_time"] = None
                st.session_state["query_error"] = None
                st.session_state["feedback_warning"] = None
    else:
        st.caption(
            "No canned questions for this database — ask anything about its own "
            "tables. Open 'Relevant Schema' after a query to see what was found."
        )

# Main question form
current_question = st.session_state.get("question_input", "")
question = st.text_input(
    "Enter your natural language question:",
    value=current_question,
    placeholder="e.g. Show me all customers from Chicago",
)
jargons_input = st.text_input(
    "Domain Jargon / Technical Terms (optional, comma-separated):",
    placeholder="e.g. PostgreSQL, MongoDB, KoboldCpp, NL2AnyQuery, customer_id, GenWorx",
    help="Words entered here will be treated as valid vocabulary by the spelling checker.",
)

if "is_executing_query" not in st.session_state:
    st.session_state["is_executing_query"] = False
if "query_data" not in st.session_state:
    st.session_state["query_data"] = None
if "query_execution_time" not in st.session_state:
    st.session_state["query_execution_time"] = None
if "query_error" not in st.session_state:
    st.session_state["query_error"] = None
if "feedback_warning" not in st.session_state:
    st.session_state["feedback_warning"] = None


def on_run_query_click() -> None:
    st.session_state["is_executing_query"] = True
    st.session_state["feedback_warning"] = None


is_executing = st.session_state["is_executing_query"]

col1, col2 = st.columns([1, 5])
with col1:
    st.button(
        "⏳ Running..." if is_executing else "🚀 Run Query",
        type="primary",
        use_container_width=True,
        disabled=is_executing,
        on_click=on_run_query_click,
        key="run_query_btn",
    )

with col2:
    if is_executing:
        st.caption("⏳ Executing query pipeline... please wait.")
    elif st.session_state.get("query_execution_time") is not None:
        exec_time = st.session_state["query_execution_time"]
        st.caption(f"⏱️ **Total Execution Time:** `{exec_time:.2f}s`")

if is_executing:
    if not question.strip():
        st.warning("⚠️ Please enter a question before running.")
        st.session_state["is_executing_query"] = False
    else:
        start_time = time.perf_counter()
        jargon_list = [j.strip() for j in jargons_input.split(",") if j.strip()]
        payload = {
            "database": db_choice,
            "question": question.strip(),
            "stages": {
                "table_selection": stage1_choice,
                "query_generation": stage2_choice,
                "validation": stage3_choice,
            },
            "jargons": jargon_list,
        }

        with st.status("🚀 Processing query pipeline...", expanded=True) as status_box:
            pipeline_data = None
            stream_error = None

            try:
                with httpx.stream("POST", f"{api_url}/query/stream", json=payload, timeout=None) as resp:
                    resp.raise_for_status()
                    for line in resp.iter_lines():
                        if not line or not line.startswith("data: "):
                            continue
                        raw_data = line[len("data: "):].strip()
                        try:
                            evt = json.loads(raw_data)
                        except Exception:
                            continue

                        evt_type = evt.get("event")
                        stage = evt.get("stage", "")
                        icon, name = STAGE_CONFIG.get(stage, ("⏳", stage.replace("_", " ").title()))

                        if evt_type == "stage_start":
                            status_box.update(
                                label=f"⏳ Running: {name}...",
                                state="running",
                                expanded=True,
                            )
                        elif evt_type == "stage_done":
                            summary = evt.get("summary") or {}
                            extra = ""
                            if stage == "table_selection" and summary.get("selected_objects"):
                                objs = summary["selected_objects"]
                                extra = f" (Selected: `{', '.join(objs)}`)"
                            elif stage == "spelling_check":
                                extra = " (Corrections applied)" if summary.get("has_changes") else " (No typos found)"
                            elif stage == "question_guardrail" and summary.get("decision"):
                                extra = f" (Decision: `{summary['decision']}`)"
                            elif stage == "query_validation":
                                extra = " (Valid query)" if summary.get("is_valid") else " (Retrying or invalid)"
                            elif stage == "policy_check":
                                extra = " (Approved)" if summary.get("passed") else " (Rejected by policy)"

                            st.write(f"{icon} **{name}** done ✅{extra}")
                        elif evt_type == "complete":
                            pipeline_data = evt.get("data")
                            status_box.update(
                                label="✅ Pipeline execution completed!",
                                state="complete",
                                expanded=False,
                            )
                        elif evt_type == "error":
                            status_code = evt.get("status_code", 500)
                            detail = evt.get("detail", "Error during execution")
                            stream_error = (status_code, detail)
                            status_box.update(
                                label=f"❌ Error: {detail}",
                                state="error",
                                expanded=True,
                            )
                            break

                elapsed_time = time.perf_counter() - start_time
                if stream_error:
                    code, detail = stream_error
                    st.session_state["query_data"] = None
                    st.session_state["query_execution_time"] = elapsed_time
                    if code == 400:
                        st.session_state["query_error"] = f"❌ Pipeline Error: {detail}"
                    elif code >= 500:
                        st.session_state["query_error"] = f"❌ Backend Server Error (HTTP {code}): {detail}"
                    else:
                        st.session_state["query_error"] = f"❌ Error: {detail}"
                elif pipeline_data:
                    st.session_state["query_data"] = pipeline_data
                    st.session_state["query_execution_time"] = elapsed_time
                    st.session_state["query_error"] = None

            except httpx.ConnectError:
                elapsed_time = time.perf_counter() - start_time
                st.session_state["query_data"] = None
                st.session_state["query_execution_time"] = elapsed_time
                st.session_state["query_error"] = (
                    f"Could not connect to FastAPI backend at {api_url}. "
                    "Please make sure it is running via `uv run run-api`."
                )
            except httpx.HTTPStatusError as err:
                elapsed_time = time.perf_counter() - start_time
                try:
                    detail = err.response.json().get("detail", err.response.text)
                except Exception:
                    detail = err.response.text
                if err.response.status_code >= 500:
                    err_msg = f"❌ Backend Server Error (HTTP {err.response.status_code}): {detail}"
                else:
                    err_msg = f"❌ Pipeline Error: {detail}"
                st.session_state["query_data"] = None
                st.session_state["query_execution_time"] = elapsed_time
                st.session_state["query_error"] = err_msg
            except Exception as err:
                elapsed_time = time.perf_counter() - start_time
                st.session_state["query_data"] = None
                st.session_state["query_execution_time"] = elapsed_time
                st.session_state["query_error"] = f"Error querying API: {err}"
            finally:
                st.session_state["is_executing_query"] = False
                st.rerun()

if st.session_state.get("query_error"):
    st.error(st.session_state["query_error"])
if st.session_state.get("feedback_warning"):
    st.warning(st.session_state["feedback_warning"])

data = st.session_state.get("query_data")
execution_time = st.session_state.get("query_execution_time")

if data:
    stage_provs = data.get("stage_providers")
    if stage_provs:
        st.caption(
            f"🧩 **Stage Providers:** Stage 1 (Table Selection): `{stage_provs.get('table_selection')}` | "
            f"Stage 2 (Generation): `{stage_provs.get('query_generation')}` | "
            f"Stage 3 (Validation): `{stage_provs.get('validation')}`"
        )

    stage_tokens = data.get("stage_tokens")
    if stage_tokens:
        s1_tok = stage_tokens.get("stage_1", 0)
        s2_tok = stage_tokens.get("stage_2", 0)
        s3_tok = stage_tokens.get("stage_3", 0)
        tot_tok = stage_tokens.get("total", 0)
        g_tok = stage_tokens.get("guardrail", 0)
        sel_tok = stage_tokens.get("table_selection", 0)

        st.markdown("##### 🪙 Estimated Prompt Token Consumption")
        mcol1, mcol2, mcol3, mcol4 = st.columns(4)
        with mcol1:
            st.metric(
                label="Stage 1: Selection",
                value=f"{s1_tok:,} tokens",
                help=f"Guardrail: {g_tok:,} | Table Selector: {sel_tok:,}",
            )
        with mcol2:
            st.metric(
                label="Stage 2: Generation",
                value=f"{s2_tok:,} tokens",
            )
        with mcol3:
            st.metric(
                label="Stage 3: Validation",
                value=f"{s3_tok:,} tokens",
            )
        with mcol4:
            st.metric(
                label="Total Tokens Sent",
                value=f"{tot_tok:,} tokens",
                help="Sum of prompt tokens sent across all executed pipeline stages.",
            )

    # Check for basic answers or errors
    if data.get("basic_answer"):
        st.info(f"ℹ️ **System Response:** {data['basic_answer']}")
        if execution_time is not None:
            st.caption(f"⏱️ Total execution time: `{execution_time:.2f}s`")

    elif data.get("error"):
        if data.get("guardrail", {}).get("decision") == "REJECT":
            st.warning(f"🛡️ **Guardrail Rejected Question:** {data['error']}")
            if data.get("guardrail", {}).get("reason"):
                st.caption(f"Reason: {data['guardrail']['reason']}")
        else:
            st.error(f"❌ **Pipeline Notice:** {data['error']}")
        if execution_time is not None:
            st.caption(f"⏱️ Total execution time: `{execution_time:.2f}s`")

    # Results section if present
    results = data.get("results")
    if results:
        row_count = results.get("row_count", 0)
        time_badge = f" — ⏱️ `{execution_time:.2f}s`" if execution_time is not None else ""
        st.subheader(f"📊 Results ({row_count} rows){time_badge}")

        preview_rows = results.get("preview_rows", [])
        if preview_rows:
            df = pd.DataFrame(preview_rows)
            st.dataframe(df, use_container_width=True)

            # CSV Download if truncated (>= 100 rows) or available
            if results.get("csv_data"):
                st.download_button(
                    label="📥 Download Full Results as CSV",
                    data=results["csv_data"],
                    file_name="query_results.csv",
                    mime="text/csv",
                )
                st.caption("Showing 10-row preview. Complete results available via CSV download.")

            # Simple deterministic visualization
            # Check if dataframe has 1 numeric column and 1 categorical/text column
            numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
            text_cols = df.select_dtypes(include=["object", "string"]).columns.tolist()

            if len(numeric_cols) >= 1 and len(text_cols) >= 1 and len(df) <= 30:
                with st.expander("📈 Chart Preview", expanded=True):
                    x_col = text_cols[0]
                    y_col = numeric_cols[0]
                    try:
                        chart_df = df.set_index(x_col)[y_col]
                        st.bar_chart(chart_df)
                    except Exception:
                        pass
        else:
            st.info("Query executed successfully, returning 0 rows.")

    # Pipeline Inspection Expanders
    st.divider()
    st.markdown("### 🔍 Pipeline Tracing & Inspection")

    with st.expander("1. Spelling Checker (SymSpell)", expanded=False):
        spelling_data = data.get("spelling_correction")
        if spelling_data:
            st.markdown(f"**Original Question:** `{spelling_data.get('original_question')}`")
            st.markdown(f"**Corrected Question:** `{spelling_data.get('corrected_question')}`")
            corrections = spelling_data.get("corrections", [])
            if corrections:
                st.markdown("**Corrections Made:**")
                st.dataframe(pd.DataFrame(corrections), hide_index=True, use_container_width=True)
            else:
                st.info("No spelling corrections needed.")
            if spelling_data.get("jargons_applied"):
                st.markdown(f"**Custom Jargon Applied:** `{spelling_data.get('jargons_applied')}`")
        else:
            st.info("Spelling check was not executed for this query.")

    with st.expander("2. Guardrail", expanded=False):
        if stage_tokens and stage_tokens.get("guardrail") is not None:
            st.caption(f"🪙 **Prompt Tokens Sent:** `{stage_tokens.get('guardrail', 0):,}` tokens")
        st.json(data.get("guardrail", {}))

    with st.expander("3. Table Selection & Grounded Schema", expanded=False):
        if stage_tokens and stage_tokens.get("table_selection") is not None:
            st.caption(f"🪙 **Prompt Tokens Sent:** `{stage_tokens.get('table_selection', 0):,}` tokens")

        st.markdown(f"**Selected Tables / Objects:** `{data.get('selected_objects', [])}`")
        st.markdown(f"**Selection Retries:** {data.get('selection_retries', 0)}")
        if data.get("relevant_schema"):
            st.markdown("**Grounded Schema for Selected Tables:**")
            tab_tree, tab_raw = st.tabs(["🌳 Interactive Tree View", "📝 Raw JSON View"])
            with tab_tree:
                st.json(data.get("relevant_schema", {}))
            with tab_raw:
                st.code(json.dumps(data.get("relevant_schema", {}), indent=2, default=str), language="json")

        run_id = data.get("run_id")
        if run_id:
            st.divider()
            st.markdown("##### 💬 Feedback & Rerun from Stage 1")
            st.caption("Provide domain guidance or table suggestions to rerun from Table Selection.")
            st.text_area(
                "Feedback for Table Selection:",
                placeholder="e.g. Include contracts table as well",
                key="feedback_stage_1",
            )
            if st.button("🔄 Rerun from Table Selection", key="rerun_btn_stage_1", type="primary", disabled=is_executing):
                fb_payload = {}
                s1_val = st.session_state.get("feedback_stage_1", "").strip()
                s2_val = st.session_state.get("feedback_stage_2", "").strip()
                s3_val = st.session_state.get("feedback_stage_3", "").strip()
                if s1_val:
                    fb_payload["table_selection"] = s1_val
                if s2_val:
                    fb_payload["query_generation"] = s2_val
                if s3_val:
                    fb_payload["validation"] = s3_val
                execute_rerun(api_url, run_id, "table_selection", fb_payload)

    with st.expander("4. Generated Query", expanded=True):
        if stage_tokens and stage_tokens.get("query_generation") is not None:
            st.caption(f"🪙 **Prompt Tokens Sent:** `{stage_tokens.get('query_generation', 0):,}` tokens")
        gen_q = data.get("generated_query")
        if gen_q:
            if db_type == "postgresql":
                st.code(gen_q, language="sql")
            else:
                st.code(
                    json.dumps(gen_q, indent=2) if isinstance(gen_q, dict) else str(gen_q),
                    language="json",
                )
        else:
            st.write("No query generated.")

        run_id = data.get("run_id")
        if run_id:
            st.divider()
            st.markdown("##### 💬 Feedback & Rerun from Stage 2")
            st.caption("Clarify query intent, metrics, or filters to rerun from Query Generation (reuses selected tables).")
            st.text_area(
                "Feedback for Query Generation:",
                placeholder="e.g. Group by vendor_id instead of vendor_name",
                key="feedback_stage_2",
            )
            if st.button("🔄 Rerun from Query Generation", key="rerun_btn_stage_2", type="primary", disabled=is_executing):
                fb_payload = {}
                s2_val = st.session_state.get("feedback_stage_2", "").strip()
                s3_val = st.session_state.get("feedback_stage_3", "").strip()
                if s2_val:
                    fb_payload["query_generation"] = s2_val
                if s3_val:
                    fb_payload["validation"] = s3_val
                execute_rerun(api_url, run_id, "query_generation", fb_payload)

    with st.expander("5. Validation", expanded=False):
        if stage_tokens and stage_tokens.get("validation") is not None:
            st.caption(f"🪙 **Prompt Tokens Sent:** `{stage_tokens.get('validation', 0):,}` tokens")
        val_data = data.get("validation", {}) or {}
        st.markdown(f"**Validation Retries:** {data.get('validation_retries', 0)}")
        st.markdown(f"**Valid:** `{val_data.get('valid')}`")
        if val_data.get("verdict"):
            st.markdown(f"**Verdict:** `{val_data.get('verdict')}`")
        st.markdown(f"**Error Classification:** `{val_data.get('error_type', 'VALID')}`")
        if val_data.get("issues"):
            st.markdown(f"**Issues:** {val_data.get('issues')}")
        if val_data.get("suggestion"):
            st.markdown(f"**Suggestion:** {val_data.get('suggestion')}")
        st.json(val_data)

        run_id = data.get("run_id")
        if run_id:
            st.divider()
            st.markdown("##### 💬 Feedback & Rerun from Stage 3")
            st.caption("Provide validation guidance or acceptable caveats to re-validate the generated query.")
            st.text_area(
                "Feedback for Validation:",
                placeholder="e.g. The query is valid; NULL handling on invoice_id is intended",
                key="feedback_stage_3",
            )
            if st.button("🔄 Rerun from Validation", key="rerun_btn_stage_3", type="primary", disabled=is_executing):
                fb_payload = {}
                s3_val = st.session_state.get("feedback_stage_3", "").strip()
                if s3_val:
                    fb_payload["validation"] = s3_val
                execute_rerun(api_url, run_id, "validation", fb_payload)

    with st.expander("6. Safety Policy", expanded=False):
        pol_data = data.get("policy", {}) or {}
        st.markdown(f"**Allowed:** `{pol_data.get('allowed')}`")
        st.markdown(f"**Reason:** {pol_data.get('reason', '')}")
        st.json(pol_data)

    with st.expander("7. Execution Results", expanded=False):
        st.json(data.get("results", {}))
