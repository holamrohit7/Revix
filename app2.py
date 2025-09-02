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
    ["📊 KPI Dashboard", "💬 Chat", "✅ Action Item Tracker", "🔔 Smart Alerts"]  # 👈 Dashboard comes first now
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
# Assign Form
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


if "last_result" not in st.session_state:
    st.session_state.last_result = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "actions" not in st.session_state:
    st.session_state.actions = []
if "assign_open_for" not in st.session_state:
    st.session_state.assign_open_for = None
if "plotly_dark" not in st.session_state:   # 🔥 FIX HERE
    st.session_state.plotly_dark = True     # default to dark theme
if "alerts" not in st.session_state:
    st.session_state.alerts = []   # store active alert rules

if page == "💬 Chat":
    st.title("🤖 Agent Revix (Chat Mode)")

    show_charts = st.sidebar.checkbox("📊 Show Charts Automatically", value=True)

    # Render past messages
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            
            st.chat_message("user").markdown(msg["content"])
        else:
            st.chat_message("assistant").markdown(msg["content"])

            # --- table ---
            if "data" in msg:
                components.html(styled_table(msg["data"]), height=280, scrolling=True)

            # --- chart ---
            if "chart" in msg:
                fig = msg["chart"]
                if st.session_state.plotly_dark:
                    fig.update_layout(template="plotly_dark", margin=dict(l=6, r=6, b=10, t=30))
                st.plotly_chart(fig, use_container_width=True, key=f"chart_{msg['id']}")

            # --- assign expander (always visible for each assistant response) ---
            with st.expander("📤 Assign as an action item", expanded=False):
                with st.form(f"assign_form_{msg['id']}", clear_on_submit=True):
                    to = st.text_input("Assign to (email or name)", key=f"to_{msg['id']}")
                    due = st.date_input("Due Date", min_value=date.today(), key=f"due_{msg['id']}")
                    priority = st.selectbox("Priority", ["Low", "Medium", "High"], index=1, key=f"priority_{msg['id']}")  # ✅ Added
                    msg_text = st.text_area("Message", "Hi, Check this data and take necessary action.", key=f"msg_{msg['id']}")
                    submitted = st.form_submit_button("✅ Confirm Assign")

                    if submitted:
                        action = {
                            "id": str(uuid.uuid4())[:8],
                            "to": to.strip(),
                            "due": str(due),
                            "priority": priority,  # ✅ Store priority
                            "msg": msg_text,
                            "answer": msg["content"],
                            "priority": "Medium",
                            "status": "Assigned"
                        }
                        st.session_state.actions.append(action)
                        st.success("✅ Action Assigned!")
                        st.rerun()

    # --- new query ---
    query = st.chat_input("Ask about your data...")

    if query:
        st.session_state.messages.append({"role": "user", "content": query, "id": str(uuid.uuid4())[:8]})
        st.chat_message("user").markdown(query)

        with st.chat_message("assistant"):
            with st.spinner("🤖 Thinking..."):
                if not is_data_related(query):
                    ans = "ℹ I only know your Excel/CSV data. No outside knowledge."
                    msg_id = str(uuid.uuid4())[:8]
                    st.markdown(ans)
                    st.session_state.messages.append({"role": "assistant", "content": ans, "id": msg_id})

                    # show assign form immediately
                    with st.expander("📤 Assign as an action item", expanded=False):
                        with st.form(f"assign_form_{msg_id}", clear_on_submit=True):
                            to = st.text_input("Assign to (email or name)", key=f"to_{msg_id}")
                            due = st.date_input("Due Date", min_value=date.today(), key=f"due_{msg_id}")
                            msg_text = st.text_area("Message", "Hi, Check this data and take necessary action.", key=f"msg_{msg_id}")
                            submitted = st.form_submit_button("✅ Confirm Assign")

                            if submitted:
                                action = {
                                    "id": str(uuid.uuid4())[:8],
                                    "to": to.strip(),
                                    "due": str(due),
                                    "msg": msg_text,
                                    "answer": ans,
                                    "priority": "Medium",
                                    "status": "Assigned"
                                }
                                st.session_state.actions.append(action)
                                st.success("✅ Action Assigned!")
                                st.experimental_rerun()

                else:
                    code = generate_pandas_code(query)
                    result = execute_pandas_code(code)

                    if "Error" in result.columns:
                        ans = f"⚠ {result.iloc[0]['Error']}"
                        msg_id = str(uuid.uuid4())[:8]
                        st.markdown(ans)
                        st.session_state.messages.append({"role": "assistant", "content": ans, "id": msg_id})

                        # assign form immediately
                        with st.expander("📤 Assign as an action item", expanded=False):
                            with st.form(f"assign_form_{msg_id}", clear_on_submit=True):
                                to = st.text_input("Assign to (email or name)", key=f"to_{msg_id}")
                                due = st.date_input("Due Date", min_value=date.today(), key=f"due_{msg_id}")
                                priority = st.selectbox("Priority", ["Low", "Medium", "High"], index=1, key=f"priority_{msg['id']}")  # ✅ Added

                                msg_text = st.text_area("Message", "Hi, Check this data and take necessary action.", key=f"msg_{msg_id}")
                                submitted = st.form_submit_button("✅ Confirm Assign")

                                if submitted:
                                    action = {
                                        "id": str(uuid.uuid4())[:8],
                                        "to": to.strip(),
                                        "due": str(due),
                                     "priority": priority,  # ✅ Store priority

                                        "msg": msg_text,
                                        "answer": ans,
                                        "priority": "Medium",
                                        "status": "Assigned"
                                    }
                                    st.session_state.actions.append(action)
                                    st.success("✅ Action Assigned!")
                                    st.experimental_rerun()

                    else:
                        ans = "Here’s the result"
                        msg_id = str(uuid.uuid4())[:8]
                        st.markdown(ans)
                        st.session_state.messages.append({"role": "assistant", "content": ans, "id": msg_id})

                        if result.shape == (1, 1):
                            col_name = result.columns[0]
                            val = result.iloc[0, 0]
                            if isinstance(val, (int, float)):
                                val = round(val, 2)
                            ans = f"{col_name} = {val}"
                            st.markdown(f"{ans}")
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
                                    if st.session_state.plotly_dark:
                                        fig.update_layout(template="plotly_dark", margin=dict(l=6, r=6, b=10, t=30))
                                    st.plotly_chart(fig, use_container_width=True)
                                    st.session_state.messages[-1]["chart"] = fig

                        # assign form immediately
                        with st.expander("📤 Assign as an action item", expanded=False):
                            with st.form(f"assign_form_{msg_id}", clear_on_submit=True):
                                to = st.text_input("Assign to (email or name)", key=f"to_{msg_id}")
                                due = st.date_input("Due Date", min_value=date.today(), key=f"due_{msg_id}")
                                priority = st.selectbox("Priority", ["Low", "Medium", "High"], index=1, key=f"priority_{msg['id']}")  # ✅ Added

                                msg_text = st.text_area("Message", "Hi, Check this data and take necessary action.", key=f"msg_{msg_id}")
                                submitted = st.form_submit_button("✅ Confirm Assign")

                                if submitted:
                                    action = {
                                        "id": str(uuid.uuid4())[:8],
                                        "to": to.strip(),
                                        "due": str(due),
                                        "priority": priority,  # ✅ Store priority

                                        "msg": msg_text,
                                        "answer": ans,
                                        "priority": "Medium",
                                        "status": "Assigned"
                                    }
                                    st.session_state.actions.append(action)
                                    st.success("✅ Action Assigned!")
                                    st.experimental_rerun()


# ------------------------
# KPI Dashboard
# ------------------------


elif page == "📊 KPI Dashboard":
    st.title("📊 KPI Dashboard")

    # ----- Sample Data -----
    random.seed(42)
    months = pd.date_range(start="2025-01-01", periods=12, freq="M")
    df_kpi = pd.DataFrame({
        "Month": months.strftime("%b"),
        "Product": [random.choice(["FNA", "FNB", "FNC"]) for _ in months],
        "Revenue": [random.randint(50000, 150000) for _ in months],
        "Customer Satisfaction": [random.randint(80, 100) for _ in months],
        "Product Uptime": [random.randint(95, 100) for _ in months],
        "Bug Fix Rate": [random.randint(85, 100) for _ in months],
        "Tickets Resolved": [random.randint(50, 200) for _ in months]
    })

    # ===== Filters =====
    st.subheader("🔎 Filters")
    col1, col2 = st.columns(2)

    with col1:
        selected_product = st.selectbox(
            "Select Product",
            ["All"] + sorted(df_kpi["Product"].unique().tolist())
        )

    with col2:
        selected_month = st.selectbox(
            "Select Period (Month)",
            ["All"] + df_kpi["Month"].unique().tolist()
        )

    # Apply filters
    df_filtered = df_kpi.copy()
    if selected_product != "All":
        df_filtered = df_filtered[df_filtered["Product"] == selected_product]
    if selected_month != "All":
        df_filtered = df_filtered[df_filtered["Month"] == selected_month]

    # ===== Gauges (2 per row) =====
    st.subheader("📊 KPI Gauges")
    chart_titles = []  # store all chart names

    g1, g2 = st.columns(2)
    with g1:
        title = "Customer Satisfaction %"
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=df_filtered["Customer Satisfaction"].mean(),
            title={'text': title},
            gauge={'axis': {'range': [0, 100]}}
        ))
        st.plotly_chart(fig, use_container_width=True)
        chart_titles.append(title)
    with g2:
        title = "Product Uptime %"
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=df_filtered["Product Uptime"].mean(),
            title={'text': title},
            gauge={'axis': {'range': [0, 100]}}
        ))
        st.plotly_chart(fig, use_container_width=True)
        chart_titles.append(title)

    # ===== Charts (2 per row) =====
    st.subheader("📈 KPI Charts")

    charts = [
        ("Monthly Revenue Trend", px.line(df_filtered, x="Month", y="Revenue", markers=True,
                                          title="Monthly Revenue Trend")),
        ("Customer Satisfaction by Month", px.bar(df_filtered, x="Month", y="Customer Satisfaction",
                                                 text="Customer Satisfaction", title="Customer Satisfaction by Month")),
        ("Product Uptime Trend", px.line(df_filtered, x="Month", y="Product Uptime", markers=True,
                                         title="Product Uptime Trend")),
        ("Bug Fix Rate by Month", px.bar(df_filtered, x="Month", y="Bug Fix Rate", text="Bug Fix Rate",
                                         title="Bug Fix Rate by Month")),
        ("Tickets Resolved Trend", px.line(df_filtered, x="Month", y="Tickets Resolved", markers=True,
                                           title="Tickets Resolved Trend")),
        ("Revenue vs Satisfaction vs Tickets", px.scatter(
            df_filtered, x="Revenue", y="Customer Satisfaction",
            size="Tickets Resolved", color="Month",
            title="Revenue vs Satisfaction vs Tickets"
        ))
    ]

    # store all chart titles
    for t, _ in charts:
        chart_titles.append(t)

    for i in range(0, len(charts), 2):
        a, b = st.columns(2)
        with a:
            st.plotly_chart(charts[i][1], use_container_width=True)
        if i + 1 < len(charts):
            with b:
                st.plotly_chart(charts[i + 1][1], use_container_width=True)

    # ===== Assign Form (at bottom) =====
    # ===== Assign Form (at bottom) =====
    st.markdown("---")
    st.subheader("📤 Assign action item")

    with st.form("assign_dashboard_form", clear_on_submit=True):   # ✅ clears after submit
        to = st.text_input("Assign to (email or name)")
        due = st.date_input("Due Date", min_value=date.today())
        chart_choice = st.selectbox("Select Chart/Gauge", chart_titles)
        priority = st.selectbox("Priority", ["Low", "Medium", "High"], index=1)  # ✅ new
        msg = st.text_area("Message", "Hi, Check this data and take necessary action.")
        submitted = st.form_submit_button("✅ Confirm Assign")

        if submitted:
            action = {
                "id": str(uuid.uuid4())[:6],
                "to": to,
                "due": str(due),
                "priority": priority,   # ✅ added
                "msg": msg,
                "answer": f"Review Dashboard KPI: {chart_choice}",
                "chart_title": chart_choice,
                "status": "Pending"
            }
            st.session_state.actions.append(action)
            st.success(f"✅ Assigned: {chart_choice}")
            st.rerun()   # ✅ refresh to clear




# ------------------------
# Action Item Tracker
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

        # 🔥 Now recalc KPIs AFTER updates
        assigned = sum(1 for a in filtered_actions if a["status"] == "Assigned")
        wip = sum(1 for a in filtered_actions if a["status"] == "Work in Progress")
        done = sum(1 for a in filtered_actions if a["status"] == "Done")

        # KPIs (move here so they are live!)
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
        fig.update_layout(template="plotly_dark", yaxis=dict(title="Count"))
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
# Smart Alerts
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

    # ------------------------
    # Step 1: Create New Alert
    # ------------------------
    st.subheader("➕ Create a New Alert")
    query = st.text_input("Define a new alert (natural language):", key="new_alert_query")

    if st.button("🧪 Test Logic") and query:
        # Generate pandas code from NLP
        code = generate_pandas_code(query)
        st.session_state.test_alert = {"query": query, "code": code}
        result = execute_pandas_code(code)

        if not result.empty:
            st.warning("⚠ Alert condition triggered on sample data!")
            components.html(styled_table(result.head(10)), height=400, scrolling=True)
            st.caption(f"Showing {min(10, len(result))} of {len(result)} rows matched.")
        else:
            st.success("✅ No issues found on sample data.")

    # ------------------------
    # Step 2: Confirm Alert
    # ------------------------
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

                # 🔥 Reset so input clears for new NLP query
                del st.session_state.test_alert
                st.session_state.reset_alert_input = True
                st.success("✅ Alert created successfully!")
                st.rerun()

    # ------------------------
    # Step 3: Manage Active Alerts
    # ------------------------
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
                    result = execute_pandas_code(alert["code"])
                    if not result.empty:
                        st.warning("⚠ Alert triggered!")
                        components.html(styled_table(result.head(10)), height=400, scrolling=True)
                        st.caption(f"Showing {min(10, len(result))} of {len(result)} rows matched.")
                    else:
                        st.success("✅ No issues found.")

                if cols[1].button("✏ Edit", key=f"edit_{i}"):
                    st.session_state.test_alert = alert
                    st.session_state.alerts.pop(i)
                    st.rerun()

                if cols[2].button("❌ Delete", key=f"del_{i}"):
                    st.session_state.alerts.pop(i)
                    st.rerun()
