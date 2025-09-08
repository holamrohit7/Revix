# revix_demo_final.py
import os
import pickle
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import streamlit.components.v1 as components
from dotenv import load_dotenv
from groq import Groq
import uuid
from datetime import date
import random
import time

# ------------------------
# Setup
# ------------------------
st.set_page_config(page_title="Revix Chatbot", page_icon="🤖", layout="wide")

# ------------------------
# Sidebar Navigation
# ------------------------
st.sidebar.header("⚙ Navigation")
page = st.sidebar.radio(
    "Go to",
    ["📊 KPI Dashboard", "💬 Chat", "✅ Action Item Tracker", "🔔 Smart Alerts"]
)

# ------------------------
# Load API Key
# ------------------------
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    st.error("❌ GROQ_API_KEY not found")
    st.stop()
client = Groq(api_key=GROQ_API_KEY)

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
if "plotly_dark" not in st.session_state:
    st.session_state.plotly_dark = False  # Light mode per request
if "alerts" not in st.session_state:
    st.session_state.alerts = []

# ------------------------
# Functions (keep originals)
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
        model="openai/gpt-oss-20b",
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": f"Question: {query}"}
        ],
        temperature=0,
    )
    return resp.choices[0].message.content.strip()

def execute_pandas_code(code: str):
    try:
        local_vars = {"pd": pd, "last_result": st.session_state.last_result, **dataframes}
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
# Assign Form (unchanged)
# ------------------------
def show_assign_form(answer_text, msg_id, chart_title=None):
    with st.form(f"assign_form_{msg_id}"):
        st.markdown("### 📤 Assign as an action item")

        to = st.text_input("Assign to (email or name)", key=f"to_{msg_id}")
        due = st.date_input("Due Date", min_value=date.today(), key=f"due_{msg_id}")
        msg = st.text_area("Message", "Hi, Check this data and take necessary action.", key=f"msg_{msg_id}")
        
        if chart_title:
            st.text_input("Related Chart/Gauge", value=chart_title, disabled=True, key=f"title_{msg_id}")

        submitted = st.form_submit_button("✅ Confirm Assign")
        if submitted:
            action = {
                "id": str(uuid.uuid4())[:6],   # shorter ID
                "to": to,
                "due": str(due),
                "msg": msg,
                "answer": answer_text,
                "chart_title": chart_title if chart_title else "",
                "status": "Pending"
            }
            st.session_state.actions.append(action)
            st.success("✅ Action Assigned!")
            st.rerun()

# ------------------------
# Demo AI responses (scripted)
# ------------------------
def demo_investigation_response():
    # Realistic, human-like phrasing with modest confidence
    return (
        "Looking at the last 30 days, the revenue forecast decline is mostly explained by a ~10% drop in product usage. "
        "Nearly 80% of that decline stems from Alpha Solutions Inc. At the same time, Alpha accounts for roughly 65% of the increase "
        "in support tickets. Other accounts show only small, expected fluctuations — so Alpha is the main driver here. "
        "I’m about 85% confident in this assessment based on the last 30 days. Would you like a full customer health report for Alpha?"
    )

def demo_customer_health_report_text():
    # Multi-line text used in chat; we'll also create structured displays below
    return (
        "**Customer Health Report — Alpha Solutions Inc.**\n\n"
        "- **Status:** HIGH churn risk\n"
        "- **ACV:** $250,000\n"
        "- **MRR:** $20,800\n"
        "- **Renewal:** In 60 days\n"
        "- **Renewal probability:** 40% (down from 70% last quarter)\n"
        "- **Benchmark:** Usage ~60% below peers of similar size\n"
        "- **Primary issues:** API integration failures & reduced feature adoption\n"
        "- **Support:** 18 tickets in last 30 days (majority tagged API issues)\n"
        "- **Sentiment:** Shift from 'Satisfied' → 'Frustrated'\n\n"
        "Key recommendation: immediate cross-functional rescue — product to triage API errors, CSM to run an executive call, sales to align on retention plan."
    )

# ------------------------
# Chat Page (modified to use scripted responses for demo prompts)
# ------------------------
if page == "💬 Chat":
    st.title("🤖 Agent Revix (Chat Mode)")

    show_charts = st.sidebar.checkbox("📊 Show Charts Automatically", value=True)

    # Render past messages
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.chat_message("user").markdown(msg["content"])
        else:
            st.chat_message("assistant").markdown(msg["content"])
            if "data" in msg:
                components.html(styled_table(msg["data"]), height=280, scrolling=True)
            if "chart" in msg:
                fig = msg["chart"]
                if st.session_state.plotly_dark:
                    fig.update_layout(template="plotly_dark", margin=dict(l=6, r=6, b=10, t=30))
                else:
                    fig.update_layout(margin=dict(l=6, r=6, b=10, t=30))
                st.plotly_chart(fig, use_container_width=True, key=f"chart_{msg['id']}")

            # assign expander (unchanged)
            with st.expander("📤 Assign as an action item", expanded=False):
                with st.form(f"assign_form_{msg['id']}", clear_on_submit=True):
                    to = st.text_input("Assign to (email or name)", key=f"to_{msg['id']}")
                    due = st.date_input("Due Date", min_value=date.today(), key=f"due_{msg['id']}")
                    priority = st.selectbox("Priority", ["Low", "Medium", "High"], index=1, key=f"priority_{msg['id']}")
                    msg_text = st.text_area("Message", "Hi, Check this data and take necessary action.", key=f"msg_{msg['id']}")
                    submitted = st.form_submit_button("✅ Confirm Assign")
                    if submitted:
                        action = {
                            "id": str(uuid.uuid4())[:8],
                            "to": to.strip(),
                            "due": str(due),
                            "priority": priority,
                            "msg": msg_text,
                            "answer": msg["content"],
                            "priority": "Medium",
                            "status": "Assigned"
                        }
                        st.session_state.actions.append(action)
                        st.success("✅ Action Assigned!")
                        st.rerun()

    # Chat input -- keep it free for you to paste the demo prompt
    query = st.chat_input("Ask about your data...")

    if query:
        # append user message
        msg_id_user = str(uuid.uuid4())[:8]
        st.session_state.messages.append({"role": "user", "content": query, "id": msg_id_user})
        st.chat_message("user").markdown(query)

        # Provide scripted demo responses for the two demo prompts:
        lower_q = query.lower().strip()

        with st.chat_message("assistant"):
            with st.spinner("🤖 Thinking..."):
                time.sleep(1.2)  # short analysis delay

            # If user asked the demo investigation question (match key phrases)
            if ("mrr forecast" in lower_q and "usage" in lower_q and "tickets" in lower_q) or (
                "mrr forecast" in lower_q and "support tickets" in lower_q and "usage" in lower_q
            ):
                response_text = demo_investigation_response()

                # typing effect
                placeholder = st.empty()
                typed = ""
                for char in response_text:
                    typed += char
                    placeholder.markdown(typed + "▌")
                    time.sleep(0.008)
                placeholder.markdown(response_text)

                st.session_state.messages.append({"role": "assistant", "content": response_text, "id": str(uuid.uuid4())[:8]})

            # If user asked for the Alpha customer health report (match keywords)
            elif "customer health report" in lower_q and "alpha" in lower_q:
                summary = demo_customer_health_report_text()

                # typing effect for summary
                placeholder = st.empty()
                typed = ""
                for char in summary:
                    typed += char
                    placeholder.markdown(typed + "▌")
                    time.sleep(0.006)
                placeholder.markdown(summary)

                # Also render structured components (table + small usage chart) inside chat
                # Create small KPI table
                kpi_df = pd.DataFrame({
                    "Metric": ["ACV", "MRR", "Renewal Probability", "Benchmark Usage"],
                    "Value": ["$250,000", "$20,800", "40% (down from 70%)", "60% below median"]
                })
                components.html(styled_table(kpi_df), height=200, scrolling=True)

                # usage chart for 12 months (fabricated)
                months_12 = pd.date_range(end=pd.Timestamp.today().normalize(), periods=12, freq='M').strftime('%b %Y').tolist()
                usage_vals = [1200, 1180, 1150, 1120, 1100, 1080, 1050, 980, 940, 900, 860, 520]  # last strong drop
                fig_usage = px.bar(x=months_12, y=usage_vals, labels={"x":"Month","y":"Logins"}, title="12-month Login Trend (Alpha Solutions)")
                fig_usage.update_layout(margin=dict(l=6, r=6, b=10, t=30))
                st.plotly_chart(fig_usage, use_container_width=True)

                # Support breakdown table
                support_df = pd.DataFrame({
                    "Theme": ["API issues", "Performance", "Uptime", "Other"],
                    "Tickets (30 days)": [12, 3, 2, 1]
                })
                components.html(styled_table(support_df), height=180, scrolling=True)

                # Add 4 feedback quotes
                st.subheader("Representative feedback (from support logs & survey)")
                st.info('"Integration keeps breaking, impacting our deliveries."')
                st.info('"We face frequent API timeouts during peak hours."')
                st.info('"Support takes too long to respond to critical issues."')
                st.info('"We also observed intermittent uptime problems last month."')

                st.session_state.messages.append({"role": "assistant", "content": summary, "id": str(uuid.uuid4())[:8]})

            else:
                # Non-demo path: fall back to original behavior (if data-related, run LLM->pandas; else show demo fallback)
                if not is_data_related(query):
                    ans = "ℹ This demo chat has limited scripted responses. Try the curated demo prompts during the presentation."
                    st.markdown(ans)
                    st.session_state.messages.append({"role": "assistant", "content": ans, "id": str(uuid.uuid4())[:8]})
                else:
                    # Keep original pipeline for data-related queries using Groq -> pandas code generator
                    try:
                        with st.spinner("🧠 Generating query..."):
                            code = generate_pandas_code(query)
                        result = execute_pandas_code(code)
                        if "Error" in result.columns:
                            ans = f"⚠ {result.iloc[0]['Error']}"
                            st.markdown(ans)
                            st.session_state.messages.append({"role": "assistant", "content": ans, "id": str(uuid.uuid4())[:8]})
                        else:
                            ans = "Here’s the result"
                            st.markdown(ans)
                            st.session_state.messages.append({"role": "assistant", "content": ans, "id": str(uuid.uuid4())[:8]})

                            if result.shape == (1, 1):
                                col_name = result.columns[0]
                                val = result.iloc[0, 0]
                                if isinstance(val, (int, float)):
                                    val = round(val, 2)
                                ans = f"{col_name} = {val}"
                                st.markdown(ans)
                                st.session_state.messages[-1]["content"] = ans
                            else:
                                components.html(styled_table(result), height=280, scrolling=True)
                                st.session_state.messages[-1]["data"] = result
                                if show_charts and len(result) > 1:
                                    num_cols = result.select_dtypes(include=["number"])
                                    if not num_cols.empty:
                                        fig = px.bar(result, x=result.columns[0], y=num_cols.columns[0],
                                                     text=num_cols.columns[0],
                                                     title=f"{num_cols.columns[0]} by {result.columns[0]}")
                                        fig.update_layout(margin=dict(l=6, r=6, b=10, t=30))
                                        st.plotly_chart(fig, use_container_width=True)
                                        st.session_state.messages[-1]["chart"] = fig
                    except Exception as e:
                        ans = f"⚠ Demo fallback: {str(e)}"
                        st.markdown(ans)
                        st.session_state.messages.append({"role":"assistant","content":ans,"id":str(uuid.uuid4())[:8]})

# ------------------------
# KPI Dashboard (simplified per spec)
# ------------------------
elif page == "📊 KPI Dashboard":
    st.title("📊 KPI Dashboard")

    # Hardcoded / simulated KPI data
    csat_score = 4.3  # out of 5
    uptime_score = 91  # percent

    # MRR history (6 months) + 3-month forecast (we'll show 6 months actual)
    months_short = pd.date_range(end=pd.Timestamp.today().normalize() - pd.Timedelta(days=30), periods=6, freq='M').strftime('%b %Y').tolist()
    # realistic numbers between 100k - 130k starting around 120k trending down
    mrr_vals = [125000, 123000, 121000, 118000, 113000, 105000]  # last 6 months
    # product usage (DAU), 10% drop last month compared to previous
    usage_vals = [1100, 1080, 1060, 1030, 1000, 900]  # last is ~10% drop vs prior
    # support tickets last 6 months - last month higher
    tickets_vals = [25, 22, 28, 30, 27, 45]

    # Top KPI row
    k1, k2, k3, k4 = st.columns([1,1,2,2])
    with k1:
        st.markdown("**CSAT**")
        st.metric(label="", value=f"{csat_score} / 5")
    with k2:
        st.markdown("**Uptime**")
        # uptime gauge using indicator
        fig_uptime = go.Figure(go.Indicator(
            mode="gauge+number",
            value=uptime_score,
            title={'text': "Uptime %"},
            gauge={'axis': {'range': [0, 100]}}
        ))
        fig_uptime.update_layout(height=180, margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig_uptime, use_container_width=True)
    with k3:
        st.markdown("**MRR (6 months)**")
        fig_mrr = px.line(x=months_short, y=mrr_vals, markers=True, labels={"x":"Month","y":"MRR ($)"})
        fig_mrr.update_layout(margin=dict(l=6, r=6, t=30, b=10))
        st.plotly_chart(fig_mrr, use_container_width=True)
    with k4:
        st.markdown("**Product Usage (DAU)**")
        fig_usage = px.bar(x=months_short, y=usage_vals, labels={"x":"Month","y":"DAU"})
        fig_usage.update_layout(margin=dict(l=6, r=6, t=30, b=10))
        st.plotly_chart(fig_usage, use_container_width=True)

    # Support tickets (separate row)
    st.markdown("**Support Tickets (last 6 months)**")
    fig_tickets = px.bar(x=months_short, y=tickets_vals, labels={"x":"Month","y":"Tickets"})
    fig_tickets.update_layout(margin=dict(l=6, r=6, t=30, b=10))
    st.plotly_chart(fig_tickets, use_container_width=True)

    # Small note and action creation hint
    st.info("Sarah notices uptime drop and creates an action item for the SRE Head to investigate uptime decrease.")

# ------------------------
# Action Item Tracker (unchanged)
# ------------------------
elif page == "✅ Action Item Tracker":
    st.title("✅ Action Item Tracker")

    # Normalize statuses
    for act in st.session_state.actions:
        if act.get("status") == "Pending":
            act["status"] = "Assigned"

    # Filters
    st.subheader("🔎 Filters")
    f1, f2, f3 = st.columns(3)
    filter_priority = f1.selectbox("Priority", ["All", "Low", "Medium", "High"])
    filter_status = f2.selectbox("Status", ["All", "Assigned", "Work in Progress", "Done"])
    filter_id = f3.text_input("Search by ID (partial allowed)")

    # Apply filters
    filtered_actions = []
    for act in st.session_state.actions:
        if filter_priority != "All" and act.get("priority", "Medium") != filter_priority:
            continue
        if filter_status != "All" and act.get("status", "Assigned") != filter_status:
            continue
        if filter_id and filter_id.lower() not in act["id"].lower():
            continue
        filtered_actions.append(act)

    if len(filtered_actions) > 0:
        st.subheader("📄 Action Details")

        # Header
        header = st.columns([1, 2, 2, 2, 2, 3, 1])
        header[0].markdown("🆔 ID")
        header[1].markdown("👤 To")
        header[2].markdown("📅 Due Date")
        header[3].markdown("⭐ Priority")
        header[4].markdown("📊 Status")
        header[5].markdown("💬 Message")
        header[6].markdown("🗑")

        for i, act in enumerate(filtered_actions):
            cols = st.columns([1, 2, 2, 2, 2, 3, 1])

            # ID
            cols[0].write(act["id"][:6])

            # To / Due Date
            cols[1].write(act["to"])
            cols[2].write(act["due"])

            # Priority selector
            new_priority = cols[3].selectbox(
                "", ["Low", "Medium", "High"],
                index=["Low", "Medium", "High"].index(act.get("priority", "Medium")),
                key=f"priority_{i}",
                label_visibility="collapsed"
            )

            # Status selector
            new_status = cols[4].selectbox(
                "", ["Assigned", "Work in Progress", "Done"],
                index=["Assigned", "Work in Progress", "Done"].index(act.get("status", "Assigned")),
                key=f"status_{i}",
                label_visibility="collapsed"
            )

            # Save live updates
            act["priority"] = new_priority
            act["status"] = new_status

            # Full message in expander if long
            if len(act["msg"]) > 40:
                with cols[5].expander("View Message"):
                    st.write(act["msg"])
            else:
                cols[5].write(act["msg"])

            # Delete
            if cols[6].button("🗑", key=f"delete_{i}"):
                st.session_state.actions.remove(act)
                st.rerun()

        # Recalc KPIs AFTER updates
        assigned = sum(1 for a in filtered_actions if a["status"] == "Assigned")
        wip = sum(1 for a in filtered_actions if a["status"] == "Work in Progress")
        done = sum(1 for a in filtered_actions if a["status"] == "Done")

        # KPIs
        st.subheader("📊 Summary")
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("📌 Total Actions", len(filtered_actions))
        k2.metric("📝 Assigned", assigned)
        k3.metric("⚙ Work in Progress", wip)
        k4.metric("✅ Done", done)

        # Chart
        chart_data = pd.DataFrame({
            "Status": ["Assigned", "Work in Progress", "Done"],
            "Count": [assigned, wip, done]
        })
        fig = px.bar(chart_data, x="Status", y="Count", text="Count", color="Status",
                     title="📊 Actions by Status")
        fig.update_layout(yaxis=dict(title="Count"))
        st.plotly_chart(fig, use_container_width=True)

        # Export
        st.download_button(
            "📥 Export Actions as CSV",
            pd.DataFrame(filtered_actions).to_csv(index=False).encode("utf-8"),
            "actions.csv",
            "text/csv"
        )

    else:
        st.info("No actions match your filters.")

# ------------------------
# Smart Alerts (keep NLP builder; add scripted test fallback)
# ------------------------
elif page == "🔔 Smart Alerts":
    st.title("🔔 Smart Alerts")

    # Ensure alerts state exists
    if "alerts" not in st.session_state:
        st.session_state.alerts = []

    # Handle reset flag BEFORE rendering text input
    if "reset_alert_input" in st.session_state and st.session_state.reset_alert_input:
        st.session_state.new_alert_query = ""
        st.session_state.reset_alert_input = False

    # Create New Alert
    st.subheader("➕ Create a New Alert")
    query = st.text_input("Define a new alert (natural language):", key="new_alert_query")

    if st.button("🧪 Test Logic") and query:
        qlow = query.lower()
        # Scripted test result when query clearly matches demo condition
        if ("acv" in qlow or "100k" in qlow) and ("usage" in qlow or "usage drop" in qlow) and ("ticket" in qlow):
            st.warning("⚠ Alert condition triggered on sample data!")
            sample = pd.DataFrame([{
                "Account": "Alpha Solutions Inc",
                "ACV": "$250,000",
                "Usage Drop (14d)": "25%",
                "Tickets (30d)": 18,
                "Match": "Yes"
            }])
            components.html(styled_table(sample), height=200, scrolling=True)
            st.caption("Scripted demo match: Alpha Solutions Inc flagged.")
            st.session_state.test_alert = {"query": query, "code": "DEMO_SCRIPTED_MATCH"}
        else:
            # Try original path (generate code -> execute) - may produce results depending on your dataframes
            try:
                code = generate_pandas_code(query)
                st.session_state.test_alert = {"query": query, "code": code}
                result = execute_pandas_code(code)
                if not result.empty:
                    st.warning("⚠ Alert condition triggered on sample data!")
                    components.html(styled_table(result.head(10)), height=400, scrolling=True)
                    st.caption(f"Showing {min(10, len(result))} of {len(result)} rows matched.")
                else:
                    st.success("✅ No issues found on sample data.")
            except Exception as e:
                st.error(f"Test failed: {e}")

    # Confirm Alert
    if "test_alert" in st.session_state:
        st.markdown("### 📧 Configure Notification")
        with st.form("confirm_alert_form", clear_on_submit=True):
            emails = st.text_input("Send to (comma separated emails)")
            message = st.text_area("Custom Message", f"Alert Triggered: {st.session_state.test_alert['query']}")
            submitted = st.form_submit_button("✅ Confirm & Create Alert")
            if submitted:
                new_alert = {
                    "id": str(uuid.uuid4())[:6],
                    "query": st.session_state.test_alert["query"],
                    "code": st.session_state.test_alert["code"],
                    "emails": [e.strip() for e in emails.split(",") if e.strip()],
                    "message": message
                }
                st.session_state.alerts.append(new_alert)
                del st.session_state.test_alert
                st.session_state.reset_alert_input = True
                st.success("✅ Alert created successfully!")
                st.rerun()

    # Manage Active Alerts
    st.markdown("### 📂 Active Alerts")
    if not st.session_state.alerts:
        st.info("No alerts defined yet.")
    else:
        for i, alert in enumerate(st.session_state.alerts):
            with st.expander(f"🔔 {alert['query']}"):
                st.write(f"*Query:* {alert['query']}")
                st.write(f"*Recipients:* {', '.join(alert.get('emails', []))}")
                st.write(f"*Message:* {alert.get('message', '')}")

                cols = st.columns([1,1,1])
                if cols[0].button("▶ Test Now", key=f"test_{i}"):
                    # If coded as DEMO_SCRIPTED_MATCH show the scripted sample
                    if alert.get("code") == "DEMO_SCRIPTED_MATCH":
                        sample = pd.DataFrame([{
                            "Account": "Alpha Solutions Inc",
                            "ACV": "$250,000",
                            "Usage Drop (14d)": "25%",
                            "Tickets (30d)": 18,
                            "Match": "Yes"
                        }])
                        st.warning("⚠ Alert triggered!")
                        components.html(styled_table(sample), height=200, scrolling=True)
                        st.caption("Scripted demo match: Alpha Solutions Inc flagged.")
                    else:
                        # run the stored code (if any) against local dataframes
                        try:
                            result = execute_pandas_code(alert["code"])
                            if not result.empty:
                                st.warning("⚠ Alert triggered!")
                                components.html(styled_table(result.head(10)), height=400, scrolling=True)
                                st.caption(f"Showing {min(10, len(result))} of {len(result)} rows matched.")
                            else:
                                st.success("✅ No issues found.")
                        except Exception as e:
                            st.error(f"Test failed: {e}")

                if cols[1].button("✏ Edit", key=f"edit_{i}"):
                    st.session_state.test_alert = alert
                    st.session_state.alerts.pop(i)
                    st.rerun()

                if cols[2].button("❌ Delete", key=f"del_{i}"):
                    st.session_state.alerts.pop(i)
                    st.rerun()
