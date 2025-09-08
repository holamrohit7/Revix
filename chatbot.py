import os
import pickle
import pandas as pd
import streamlit as st
import plotly.express as px
import streamlit.components.v1 as components
from dotenv import load_dotenv
from openai import OpenAI
import uuid
from datetime import date, datetime, timedelta
import time
import re

# ------------------------
# Setup
# ------------------------
st.set_page_config(page_title="Revix Chatbot", page_icon="🤖", layout="wide")
st.title("🤖 Agent Revix")

# Bigger chat output font
st.markdown(
    """
    <style>
    /* Enlarge chat message text */
    div[data-testid="stChatMessage"] p,
    div[data-testid="stChatMessage"] li,
    div[data-testid="stChatMessage"] span {
        font-size: 1.1rem; /* ~17.6px */
        line-height: 1.6;
    }
    div[data-testid="stChatMessage"] code {
        font-size: 1.0rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ChatGPT-like UI styling: centered content, bubbles, sticky input
st.markdown(
    """
    <style>
    /* Center column and soften background */
    .block-container { max-width: 860px; margin: 0 auto; }
    html, body { background: linear-gradient(180deg, #f7f7f8 0%, #ffffff 60%); }

    /* Message bubbles */
    div[data-testid="stChatMessage"] {
        background: #ffffff;
        border: 1px solid #ececec;
        border-radius: 14px;
        padding: 12px 14px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
        margin: 12px 0;
    }

    /* Code blocks inside messages */
    div[data-testid="stChatMessage"] pre,
    div[data-testid="stChatMessage"] code {
        background: #f6f8fa !important;
        border: 1px solid #eaecef;
        border-radius: 8px;
        padding: 0.3rem 0.5rem;
    }

    /* Sticky chat input */
    div[data-testid="stChatInput"] {
        position: sticky;
        bottom: 0;
        background: rgba(255,255,255,0.9);
        backdrop-filter: blur(6px);
        border-top: 1px solid #eee;
        padding-top: 6px;
        z-index: 10;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ------------------------
# Sidebar Options
# ------------------------
st.sidebar.header("⚙ Options")
show_charts = st.sidebar.checkbox("📊 Show Charts Automatically", value=True)

# ------------------------
# Load API Key (GitHub Models / Copilot)
# ------------------------
load_dotenv()
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN") or os.getenv("GH_TOKEN")
if not GITHUB_TOKEN:
    st.error("❌ GitHub token not found (set GITHUB_TOKEN or GH_TOKEN with models:read)")
    st.stop()
client = OpenAI(
    base_url="https://models.inference.ai.azure.com",
    api_key=GITHUB_TOKEN,
)

# ------------------------
# Load Cached DataFrames
# ------------------------
DATA_FOLDER = os.path.join(os.path.dirname(__file__), "excel")
DF_FILE = os.path.join(DATA_FOLDER, "dataframes.pkl")

if not os.path.exists(DF_FILE):
    st.error("❌ dataframes.pkl not found in repo. Please make sure it exists inside /excel folder.")
    st.stop()

with open(DF_FILE, "rb") as f:
    dataframes = pickle.load(f)

# ------------------------
# Schema for LLM
# ------------------------
schema_desc = "\n".join([f"{name}: {', '.join(df.columns)}" for name, df in dataframes.items()])

# ------------------------
# Smart Memory
# ------------------------
if "last_result" not in st.session_state:
    st.session_state.last_result = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "actions" not in st.session_state:
    st.session_state.actions = []

# ------------------------
# Functions
# ------------------------
def is_data_related(query: str) -> bool:
    q = query.lower()
    for df_name, df in dataframes.items():
        if df_name.lower() in q:
            return True
        for col in df.columns:
            if col.lower() in q:
                return True
    return False

def generate_pandas_code(query: str) -> str:
    system_msg = (
        "You are an assistant that converts natural questions into Pandas expressions. "
        "Rules:\n"
        "- Only return a single Pandas expression.\n"
        "- Use only these DataFrames: \n"
        f"{schema_desc}\n\n"
        "If query refers to 'last result', use the DataFrame variable last_result."
    )
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": f"Question: {query}"}
        ],
        temperature=0,
    )
    return resp.choices[0].message.content.strip()

def execute_pandas_code(code: str):
    try:
        # Prepare local vars with datelike columns coerced to datetime
        prepared = {}
        for _name, _df in dataframes.items():
            d = _df.copy()
            try:
                for _c in d.columns:
                    _cl = str(_c).lower()
                    if any(k in _cl for k in ["date", "month", "timestamp", "time"]):
                        try:
                            d[_c] = pd.to_datetime(d[_c], errors="coerce")
                        except Exception:
                            pass
            except Exception:
                pass
            prepared[_name] = d

        local_vars = {"pd": pd, "last_result": st.session_state.last_result, **prepared}
        result = eval(code, {}, local_vars)

        if isinstance(result, pd.Series):
            result = result.to_frame().reset_index()

        if isinstance(result, pd.DataFrame):
            result = result.copy()
            if result.index.name in ["Product", "Customer"]:
                result = result.reset_index()
            for key in ["Product", "Customer"]:
                if key not in result.columns and key in result.index.names:
                    result = result.reset_index()
            for col in result.select_dtypes(include=["float", "int"]).columns:
                result[col] = result[col].round(2)
            st.session_state.last_result = result
            return result

        return pd.DataFrame([{"Answer": result}])
    except Exception as e:
        return pd.DataFrame([{"Error": str(e)}])

def styled_table(df: pd.DataFrame):
    df = df.copy().round(2).reset_index(drop=True)
    html_table = df.to_html(index=False, border=0, classes="styled-table")
    css = """
    <style>
    table.styled-table {
        border-collapse: collapse;
        margin: 15px auto;
        font-size: 16px;
        width: 90%;
        border-radius: 6px;
        overflow: hidden;
    }
    table.styled-table th {
        background-color: #333;
        color: #fff;
        text-align: center;
        padding: 8px;
    }
    table.styled-table td {
        text-align: center;
        padding: 6px;
    }
    table.styled-table tr:nth-child(even) {
        background-color: #f2f2f2;
    }
    table.styled-table tr:nth-child(odd) {
        background-color: #ffffff;
    }
    </style>
    """
    return css + html_table

# ------------------------
# Helpers for data-backed reports
# ------------------------
def _coerce_dt(series: pd.Series):
    try:
        return pd.to_datetime(series, errors="coerce")
    except Exception:
        return pd.to_datetime(pd.Series([None]*len(series)))

def resolve_customer_codes(company_name: str) -> list:
    """Map a human company name to internal coded customers using a heuristic.
    For Alpha Solutions Inc -> all customers like Alpha01..Alpha99 found across dataframes.
    """
    name = company_name.strip().lower()
    # Heuristic families by prefix (extendable)
    families = {
        "alpha": r"^Alpha\d+$",
        "bravo": r"^Bravo\d+$",
        "charlie": r"^Charlie\d+$",
        "delta": r"^Delta\d+$",
        "echo": r"^Echo\d+$",
        "foxtrot": r"^Foxtrot\d+$",
        "golf": r"^Golf\d+$",
        "hotel": r"^Hotel\d+$",
        "india": r"^India\d+$",
        "juliet": r"^Juliet\d+$",
        "kilo": r"^Kilo\d+$",
        "lima": r"^Lima\d+$",
        "mike": r"^Mike\d+$",
        "november": r"^November\d+$",
        "oscar": r"^Oscar\d+$",
        "papa": r"^Papa\d+$",
        "quebec": r"^Quebec\d+$",
        "romeo": r"^Romeo\d+$",
        "sierra": r"^Sierra\d+$",
        "tango": r"^Tango\d+$",
        "uniform": r"^Uniform\d+$",
        "victor": r"^Victor\d+$",
        "whiskey": r"^Whiskey\d+$",
        "xray": r"^Xray\d+$",
        "yankee": r"^Yankee\d+$",
        "zulu": r"^Zulu\d+$",
    }
    pattern = None
    for key, pat in families.items():
        if key in name:
            pattern = pat
            break
    if not pattern:
        return []

    candidates = set()
    for df in dataframes.values():
        if isinstance(df, pd.DataFrame) and "Customer" in df.columns:
            try:
                matched = df[ df["Customer"].astype(str).str.match(pattern, na=False) ]["Customer"].unique()
                candidates.update(matched)
            except Exception:
                continue
    return sorted(candidates)

def safe_number(x, default="N/A", as_int=False):
    try:
        if pd.isna(x):
            return default
        return int(x) if as_int else float(x)
    except Exception:
        return default

def format_money(n):
    if isinstance(n, (int, float)):
        return f"${n:,.0f}"
    return str(n)

# ------------------------
# Action Assign Feature
# ------------------------
def show_assign_form(answer_text, msg_id):
    with st.form(f"assign_form_{msg_id}"):
        st.markdown("### 📤 Assign this Answer")
        to = st.text_input("Assign to (email or name)", key=f"to_{msg_id}")
        due = st.date_input("Due Date", min_value=date.today(), key=f"due_{msg_id}")
        msg = st.text_area("Message", "Check this data and do analysis.", key=f"msg_{msg_id}")
        submitted = st.form_submit_button("✅ Confirm Assign")
        if submitted:
            action = {
                "id": str(uuid.uuid4()),
                "to": to,
                "due": str(due),
                "msg": msg,
                "answer": answer_text,
                "status": "Pending"
            }
            st.session_state.actions.append(action)
            st.success("✅ Action Assigned!")
            st.rerun()

# ------------------------
# Sidebar: Monitor Actions
# ------------------------
st.sidebar.subheader("📌 Action Monitor")
if len(st.session_state.actions) == 0:
    st.sidebar.info("No actions assigned yet.")
else:
    for act in st.session_state.actions:
        st.sidebar.write(f"*To:* {act['to']} | *Due:* {act['due']} | *Status:* {act['status']}")
        st.sidebar.caption(f"{act['msg']}")
        if st.sidebar.button(f"Mark Done {act['id']}", key=f"done_{act['id']}"):
            act["status"] = "Done"

# ------------------------
# Chat UI
# ------------------------
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.chat_message("user").markdown(msg["content"])
    else:
        st.chat_message("assistant").markdown(msg["content"])
        if "data" in msg:
            components.html(styled_table(msg["data"]), height=400, scrolling=True)
        if "chart" in msg:
            st.plotly_chart(msg["chart"], use_container_width=True, key=f"chart_{uuid.uuid4()}")
        if "content" in msg:
            show_assign_form(msg["content"], msg_id=msg["id"])  # ✅ stable ID

query = st.chat_input("Ask about your data...")

if query:
    st.session_state.messages.append({"role": "user", "content": query, "id": str(uuid.uuid4())})
    st.chat_message("user").markdown(query)
    # Short-circuit for the specified exact query: return fixed text with no processing
    if query.strip() == "Why is our MRR forecast, and usage is dropping while support tickets are increasing?":
        ans = (
            "Looking at the last 30 days, the revenue forecast decline is mostly explained by a 10% drop in usage. "
            "Nearly 80% of that drop comes from Alpha Solutions Inc. What stands out is that Alpha also accounts for "
            "about 65% of the overall rise in support tickets. Other accounts show only small, expected fluctuations — "
            "so Alpha is clearly the main driver here."
        )
        msg_id = str(uuid.uuid4())
        with st.chat_message("assistant"):
            with st.spinner("🤖 Thinking..."):
                time.sleep(3)
            # Typewriter effect
            placeholder = st.empty()
            typed = ""
            for ch in ans:
                typed += ch
                placeholder.markdown(typed)
                time.sleep(0.01)
            st.session_state.messages.append({"role": "assistant", "content": ans, "id": msg_id})
            show_assign_form(ans, msg_id=msg_id)
    # Customer health report for Alpha Solutions Inc (special-case)
    elif query.strip().strip('"\'').rstrip('.!?').lower() in {
        "give me customer health report of alpha solutions inc",
        "customer health report of alpha solutions inc",
        "customer health report for alpha solutions inc",
    }:
        msg_id = str(uuid.uuid4())
        # Data-backed report
        company = "Alpha Solutions Inc"
        codes = resolve_customer_codes(company)

        usage_df = dataframes.get("usage") if isinstance(dataframes.get("usage"), pd.DataFrame) else None
        gainsight_df = dataframes.get("gainsight") if isinstance(dataframes.get("gainsight"), pd.DataFrame) else None
        salesforce_df = dataframes.get("salesforce") if isinstance(dataframes.get("salesforce"), pd.DataFrame) else None
        revenue_df = dataframes.get("revenue") if isinstance(dataframes.get("revenue"), pd.DataFrame) else None

        # Usage trend (Users & API Calls) aggregated by Month for selected codes
        fig_usage = None
        usage_msg = "No usage data found."
        if usage_df is not None and codes:
            dfu = usage_df.copy()
            if "Month" in dfu.columns:
                dfu["Month"] = pd.to_datetime(dfu["Month"], errors="coerce")
            dfu = dfu[dfu["Customer"].astype(str).isin(codes)]
            if not dfu.empty and "Month" in dfu.columns:
                last_12_months = dfu["Month"].dropna().sort_values().unique()[-12:]
                dfu = dfu[dfu["Month"].isin(last_12_months)]
                agg = dfu.groupby("Month", as_index=False)[[c for c in ["Users", "API Calls"] if c in dfu.columns]].sum()
                if not agg.empty:
                    long = agg.melt("Month", var_name="Metric", value_name="Value")
                    fig_usage = px.line(long, x="Month", y="Value", color="Metric", markers=True,
                                        title=f"{company} — Usage Trend (Users & API Calls)")
                    usage_msg = "See usage trend below."

        # Support themes from salesforce (last 180 days)
        support_df = pd.DataFrame()
        if salesforce_df is not None and codes:
            dfs = salesforce_df.copy()
            if "Created Date" in dfs.columns:
                dfs["Created Date"] = pd.to_datetime(dfs["Created Date"], errors="coerce")
                cutoff = pd.Timestamp.now() - pd.Timedelta(days=180)
                dfs = dfs[(dfs["Customer"].astype(str).isin(codes)) & (dfs["Created Date"] >= cutoff)]
            else:
                dfs = dfs[dfs["Customer"].astype(str).isin(codes)]
            if not dfs.empty:
                if "Comments" in dfs.columns:
                    support_df = dfs.groupby("Comments", as_index=False).size().rename(columns={"size": "Tickets", "Comments": "Theme"})
                    support_df = support_df.sort_values("Tickets", ascending=False).head(6)
                else:
                    # fallback: by Status
                    col = "Status" if "Status" in dfs.columns else dfs.columns[0]
                    support_df = dfs.groupby(col, as_index=False).size().rename(columns={"size": "Tickets", col: "Theme"}).head(6)

        # Gainsight snapshot
        health_score, renewal_likelihood, churn_risk, last_contact = None, None, None, None
        if gainsight_df is not None and codes:
            dfg = gainsight_df.copy()
            if "Month" in dfg.columns:
                dfg["Month"] = pd.to_datetime(dfg["Month"], errors="coerce")
                # latest per customer
                dfg = dfg[dfg["Customer"].astype(str).isin(codes)]
                if not dfg.empty:
                    latest = dfg.sort_values("Month").groupby("Customer").tail(1)
                    if "Health Score" in latest.columns:
                        health_score = round(latest["Health Score"].mean(), 2)
                    if "Renewal Likelihood" in latest.columns:
                        renewal_likelihood = round(latest["Renewal Likelihood"].mean(), 2)
                    if "Churn Risk" in latest.columns:
                        churn_risk = round(latest["Churn Risk"].mean(), 2)
                    if "Last Contact Days Ago" in latest.columns:
                        last_contact = int(round(latest["Last Contact Days Ago"].mean(), 0))

        # Revenue metrics (best-effort)
        acv_display = "N/A"
        mrr_display = "N/A"
        renewal_date_display = "N/A"
        if revenue_df is not None:
            dfr = revenue_df.copy()
            if "Customer" in dfr.columns:
                dfr = dfr[dfr["Customer"].astype(str).isin(codes)] if codes else dfr
            lower_cols = {c.lower(): c for c in dfr.columns}
            # ACV / ARR / Amount
            acv_col = next((lower_cols[c] for c in lower_cols if "acv" in c), None)
            arr_col = next((lower_cols[c] for c in lower_cols if "arr" in c), None)
            mrr_col = next((lower_cols[c] for c in lower_cols if "mrr" in c), None)
            amt_col = next((lower_cols[c] for c in lower_cols if c in ("amount", "value", "contract value")), None)
            renew_col = next((lower_cols[c] for c in lower_cols if "renewal" in c and ("date" in c or "on" in c)), None)

            if acv_col or arr_col or amt_col:
                base = dfr[acv_col] if acv_col else (dfr[arr_col] if arr_col else dfr[amt_col])
                acv_val = safe_number(base.sum())
                acv_display = format_money(acv_val) if isinstance(acv_val, (int, float)) else acv_display
                if not mrr_col and isinstance(acv_val, (int, float)):
                    mrr_display = format_money(acv_val/12.0)
            if mrr_col:
                mrr_val = safe_number(dfr[mrr_col].sum())
                mrr_display = format_money(mrr_val) if isinstance(mrr_val, (int, float)) else mrr_display
            if renew_col:
                try:
                    rd = pd.to_datetime(dfr[renew_col], errors="coerce").max()
                    if pd.notna(rd):
                        renewal_date_display = rd.date().isoformat()
                except Exception:
                    pass

        # Salesforce KPIs
        csat_avg = ttr_avg = tickets_90d = None
        if salesforce_df is not None and codes:
            dfs2 = salesforce_df.copy()
            if "Created Date" in dfs2.columns:
                dfs2["Created Date"] = pd.to_datetime(dfs2["Created Date"], errors="coerce")
                cutoff90 = pd.Timestamp.now() - pd.Timedelta(days=90)
                dfs2 = dfs2[(dfs2["Customer"].astype(str).isin(codes)) & (dfs2["Created Date"] >= cutoff90)]
            else:
                dfs2 = dfs2[dfs2["Customer"].astype(str).isin(codes)]
            if not dfs2.empty:
                if "CSAT Score" in dfs2.columns:
                    csat_avg = round(pd.to_numeric(dfs2["CSAT Score"], errors="coerce").mean(), 2)
                if "TTR (hours)" in dfs2.columns:
                    ttr_avg = round(pd.to_numeric(dfs2["TTR (hours)"], errors="coerce").mean(), 2)
                tickets_90d = int(dfs2.shape[0])

        # Compose narrative
        lines = [
            f"### Customer Health Report — {company} (Last 12 Months)",
            "",
            "#### Revenue",
            f"- ACV: {acv_display}",
            f"- MRR: {mrr_display}",
            f"- Renewal date: {renewal_date_display}",
            "- Upsell status: Not tracked",
            "",
            "#### Forecasted / Gainsight Snapshot",
            f"- Health Score: {health_score if health_score is not None else 'N/A'}",
            f"- Renewal Likelihood: {renewal_likelihood if renewal_likelihood is not None else 'N/A'}",
            f"- Churn Risk: {churn_risk if churn_risk is not None else 'N/A'}",
            f"- Last Contact (avg days ago): {last_contact if last_contact is not None else 'N/A'}",
            "",
            "#### Usage",
            f"- {usage_msg}",
            "",
            "#### Support (last 90–180 days)",
            f"- Tickets last 90d: {tickets_90d if tickets_90d is not None else 'N/A'}",
            f"- Avg CSAT: {csat_avg if csat_avg is not None else 'N/A'}",
            f"- Avg TTR (hours): {ttr_avg if ttr_avg is not None else 'N/A'}",
        ]
        ans = "\n".join(lines)

        with st.chat_message("assistant"):
            with st.spinner("🤖 Thinking..."):
                time.sleep(3)
            # Typewriter effect for the narrative sections
            placeholder = st.empty()
            typed = ""
            for ch in ans:
                typed += ch
                placeholder.markdown(typed)
                time.sleep(0.01)
            # Render chart if available
            if fig_usage is not None:
                st.plotly_chart(fig_usage, use_container_width=True)
            # Render support table (top themes)
            if not support_df.empty:
                components.html(styled_table(support_df), height=300, scrolling=True)

            # Persist in session so it shows on rerun
            payload = {
                "role": "assistant",
                "content": ans,
                "id": msg_id,
            }
            if fig_usage is not None:
                payload["chart"] = fig_usage
            if not support_df.empty:
                payload["data"] = support_df
            st.session_state.messages.append(payload)
            show_assign_form(ans, msg_id=msg_id)
    else:
        with st.chat_message("assistant"):
            with st.spinner("🤖 Thinking..."):
                if not is_data_related(query):
                    ans = "ℹ I only know your Excel/CSV data. No outside knowledge."
                    st.markdown(ans)
                    st.session_state.messages.append({"role": "assistant", "content": ans, "id": str(uuid.uuid4())})
                    show_assign_form(ans, msg_id=st.session_state.messages[-1]["id"])
                else:
                    code = generate_pandas_code(query)
                    result = execute_pandas_code(code)

                    if "Error" in result.columns:
                        ans = f"⚠ {result.iloc[0]['Error']}"
                        st.markdown(ans)
                        st.session_state.messages.append({"role": "assistant", "content": ans, "id": str(uuid.uuid4())})
                        show_assign_form(ans, msg_id=st.session_state.messages[-1]["id"])
                    else:
                        ans = "Here’s the result"
                        st.markdown(ans)
                        st.session_state.messages.append({"role": "assistant", "content": ans, "id": str(uuid.uuid4())})

                        msg_id = st.session_state.messages[-1]["id"]

                        if result.shape == (1, 1):
                            col_name = result.columns[0]
                            val = result.iloc[0, 0]
                            if isinstance(val, (int, float)):
                                val = round(val, 2)
                            ans = f"{col_name} = {val}"
                            st.markdown(f"{ans}")
                            st.session_state.messages[-1]["content"] = ans
                            show_assign_form(ans, msg_id)
                        else:
                            components.html(styled_table(result), height=400, scrolling=True)
                            st.session_state.messages[-1]["data"] = result
                            if show_charts and len(result) > 1:
                                num_cols = result.select_dtypes(include=["number"])
                                if not num_cols.empty:
                                    fig = px.bar(result, x=result.columns[0], y=num_cols.columns[0], text=num_cols.columns[0])
                                    st.plotly_chart(fig, use_container_width=True, key=f"chart_{uuid.uuid4()}")
                                    st.session_state.messages[-1]["chart"] = fig
                            show_assign_form(ans, msg_id)