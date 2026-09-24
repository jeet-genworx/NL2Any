"""Streamlit frontend for NL2AnyQuery."""

import json
import os
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

# Sidebar
with st.sidebar:
    st.header("Settings")
    db_choice = st.selectbox(
        "Database",
        options=["postgres", "mongo"],
        format_func=lambda x: "PostgreSQL (Relational)" if x == "postgres" else "MongoDB (Document)",
    )
    api_url = st.text_input("FastAPI Backend URL", value=API_BASE_URL)

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
        "⚙️ Initialize Database",
        use_container_width=True,
        help="Extract metadata → build schema graph/MST → generate SLM table descriptions → embed them.",
    )

    if ingest_btn:
        with st.spinner(f"Running ingestion pipeline for {db_choice}... this can take a minute."):
            try:
                ingest_resp = httpx.post(
                    f"{api_url}/ingest/{db_choice}",
                    params={"use_mst": use_mst},
                    timeout=300.0,
                )
                ingest_resp.raise_for_status()
                ingest_data = ingest_resp.json()
                st.success(
                    f"Ingested {ingest_data['table_count']} tables → "
                    f"{ingest_data['description_count']} descriptions → "
                    f"{ingest_data['embedding_count']} embeddings "
                    f"({ingest_data['embedding_dimensions']}-dim, "
                    f"MST used: {ingest_data['used_mst']})."
                )
                with st.expander("Ingestion details"):
                    st.json(ingest_data)
            except httpx.ConnectError:
                st.error(
                    f"Could not connect to FastAPI backend at {api_url}. "
                    "Please make sure it is running via `uv run run-api`."
                )
            except httpx.HTTPStatusError as err:
                st.error(f"Ingestion failed: {err.response.json().get('detail', err.response.text)}")
            except Exception as err:
                st.error(f"Ingestion failed: {err}")

    st.divider()
    st.markdown("### Example Questions")
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
    else:
        sample_questions = [
            "Which customers have placed the most orders?",
            "What are the top-selling products?",
            "Show orders containing more than one item",
            "What is the total sales amount per customer?",
            "Show all customers located in Denver",
            "Drop the orders collection",  # Negative test
            "Find all spaceships",  # Unknown domain
        ]

    for q in sample_questions:
        if st.button(q, key=f"btn_{q}"):
            st.session_state["question_input"] = q

# Main question form
current_question = st.session_state.get("question_input", "")
question = st.text_input(
    "Enter your natural language question:",
    value=current_question,
    placeholder="e.g. Show me all customers from Chicago",
)

col1, col2 = st.columns([1, 5])
with col1:
    run_btn = st.button("🚀 Run Query", type="primary", use_container_width=True)

if run_btn and question.strip():
    with st.spinner("Executing pipeline through local SLM..."):
        try:
            resp = httpx.post(
                f"{api_url}/query",
                json={"database": db_choice, "question": question.strip()},
                timeout=120.0,
            )
            resp.raise_for_status()
            data = resp.json()
        except httpx.ConnectError:
            st.error(
                f"Could not connect to FastAPI backend at {api_url}. "
                "Please make sure it is running via `uv run run-api`."
            )
            st.stop()
        except Exception as err:
            st.error(f"Error querying API: {err}")
            st.stop()

    # Check for basic answers or errors
    if data.get("basic_answer"):
        st.info(f"ℹ️ **System Response:** {data['basic_answer']}")

    elif data.get("error"):
        if data.get("guardrail", {}).get("decision") == "REJECT":
            st.warning(f"🛡️ **Guardrail Rejected Question:** {data['error']}")
            if data.get("guardrail", {}).get("reason"):
                st.caption(f"Reason: {data['guardrail']['reason']}")
        else:
            st.error(f"❌ **Pipeline Notice:** {data['error']}")

    # Results section if present
    results = data.get("results")
    if results:
        row_count = results.get("row_count", 0)
        st.subheader(f"📊 Results ({row_count} rows)")

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

    with st.expander("Guardrail"):
        st.json(data.get("guardrail", {}))

    with st.expander("Semantic Analysis"):
        st.json(data.get("semantic_analysis", {}))

    with st.expander("Linguistic Analysis"):
        st.json(data.get("linguistic_analysis", {}))

    with st.expander("Embedding Retrieval"):
        candidate_tables = data.get("candidate_tables", [])
        if candidate_tables:
            st.markdown("**Embedding Candidates (Cosine Similarity > 0.80, Top 15 max):**")
            cand_df = pd.DataFrame(candidate_tables)[["rank", "table_name", "similarity"]]
            cand_df.columns = ["Rank", "Table", "Similarity Score"]
            st.dataframe(cand_df, hide_index=True, use_container_width=True)
        else:
            st.info("No candidate tables retrieved or applicable.")

    with st.expander("Table Selection"):
        st.markdown(f"**Selected Tables / Objects:** `{data.get('selected_objects', [])}`")
        st.markdown(f"**Selection Retries:** {data.get('selection_retries', 0)}")

    with st.expander("Relevant Schema"):
        st.json(data.get("relevant_schema", {}))

    with st.expander("Query Plan"):
        st.json(data.get("query_plan", {}))

    with st.expander("Generated Query"):
        gen_q = data.get("generated_query")
        if gen_q:
            if db_choice == "postgres":
                st.code(gen_q, language="sql")
            else:
                st.code(
                    json.dumps(gen_q, indent=2) if isinstance(gen_q, dict) else str(gen_q),
                    language="json",
                )
        else:
            st.write("No query generated.")

    with st.expander("Validation"):
        val_data = data.get("validation", {}) or {}
        st.markdown(f"**Validation Retries:** {data.get('validation_retries', 0)}")
        st.markdown(f"**Valid:** `{val_data.get('valid')}`")
        st.markdown(f"**Error Classification:** `{val_data.get('error_type', 'VALID')}`")
        if val_data.get("issues"):
            st.markdown(f"**Issues:** {val_data.get('issues')}")
        if val_data.get("suggestion"):
            st.markdown(f"**Suggestion:** {val_data.get('suggestion')}")
        st.json(val_data)

    with st.expander("Policy"):
        pol_data = data.get("policy", {}) or {}
        st.markdown(f"**Allowed:** `{pol_data.get('allowed')}`")
        st.markdown(f"**Reason:** {pol_data.get('reason', '')}")
        st.json(pol_data)

    with st.expander("Results"):
        st.json(data.get("results", {}))
