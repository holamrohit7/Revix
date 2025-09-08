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

# # ------------------------
# # Setup
# # ------------------------
# st.set_page_config(page_title="Revix Chatbot", page_icon="🤖", layout="wide")

# # ------------------------
# # Sidebar Navigation
# # ------------------------
# st.sidebar.header("⚙ Navigation")
# page = st.sidebar.radio(
#     "Go to",
#     ["📊 KPI Dashboard", "💬 Chat", "✅ Action Item Tracker", "🔔 Smart Alerts"]
# )

------------------------
# Sidebar with Logo at Top (no extra space above)
------------------------
st.markdown(
    """
    <style>
    /* remove the default top padding/space in sidebar */
    section[data-testid="stSidebar"] > div:first-child {
        padding-top: 0rem;
    }
    .logo-card {
        text-align: center;
        padding: 10px 5px 20px 5px;
    }
    .logo-title {
        font-size: 16px;
        font-weight: 800;
        color: #FFFFFF;
        margin-top: 8px;
        margin-bottom: 2px;
    }
    .logo-sub {
        font-size: 14px;
        font-weight: 400;
        color: #AFC6D9;
        margin: 0;
    }
    .nav-section h3 {
        font-size: 15px;
        margin-bottom: 6px;
        color: #FFFFFF;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    # --- Logo card ---
    st.markdown('<div class="logo-card">', unsafe_allow_html=True)
    st.image("Images/logo.jpg", use_container_width=True)  # 👈 adjust width if needed
    st.markdown("<div class='logo-title'>Revolutionize Data & Execute Precisely</div>", unsafe_allow_html=True)
    st.markdown("<div class='logo-sub'>AI-Powered Analytics</div>", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # --- Navigation ---
    st.markdown('<div class="nav-section">', unsafe_allow_html=True)
    st.markdown("### ⚙ Navigation", unsafe_allow_html=True)
    page = st.radio(
        "",
        ["📊 KPI Dashboard", "💬 Chat", "✅ Action Item Tracker", "🔔 Smart Alerts"],
        index=0,
        key="main_nav",
    )
    st.markdown('</div>', unsafe_allow_html=True)


# --- Account section (clean, no avatar) ---
st.markdown(
    """
    <style>

    .account-card {
        margin-top: auto;
        background: linear-gradient(180deg, rgba(255,255,255,0.03), rgba(255,255,255,0.01));
        border-radius: 12px;
        padding: 14px 16px;
        margin: 12px 8px 0px 8px;   /* 👈 set bottom margin = 0 */
        color: #E6EEF3;
        box-shadow: 0 2px 8px rgba(2,6,18,0.6);
    }
        
    .acct-name {
        font-size:15px;
        font-weight:700;
        margin:0 0 4px 0;
        color:#ffffff;
    }
    .acct-email {
        font-size:13px;
        color:#9fb2c8;
        margin:0;
    }
    .acct-role {
        font-size:12px;
        color:#9fb2c8;
        margin:4px 0 10px 0;
    }
    .acct-actions {
        display:flex;
        gap:10px;
    }
    .acct-btn {
        background:#1a2538;
        border:1px solid rgba(255,255,255,0.08);
        color:#cfe6ff;
        padding:6px 12px;
        border-radius:8px;
        font-size:12px;
        cursor:pointer;
        transition: all 0.2s ease;
    }
    .acct-btn:hover {
        background:#22314a;
        border-color: rgba(255,255,255,0.2);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown(
        """
        <div class="account-card">
            <p class="acct-name">John Cena</p>
            <p class="acct-email">John@revix.com</p>
            <p class="acct-role">VP Sales</p>
            <div class="acct-actions">
                <button class="acct-btn">Manage</button>
                <button class="acct-btn">Sign out</button>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
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
        "Customer Health Report — Alpha Solutions Inc.\n\n"
        "- Status: HIGH churn risk\n"
        "- ACV: $250,000\n"
        "- MRR: $20,800\n"
        "- Renewal: In 60 days\n"
        "- Renewal probability: 40% (down from 70% last quarter)\n"
        "- Benchmark: Usage ~60% below peers of similar size\n"
        "- Primary issues: API integration failures & reduced feature adoption\n"
        "- Support: 18 tickets in last 30 days (majority tagged API issues)\n"
        "- Sentiment: Shift from 'Satisfied' → 'Frustrated'\n\n"
        "Key recommendation: immediate cross-functional rescue — product to triage API errors, CSM to run an executive call, sales to align on retention plan."
    )

# ------------------------
# Chat Page (modified to use scripted responses for demo prompts)
# ------------------------
if page == "💬 Chat":
    st.title("🤖 Agent Revix (Chat Mode)")

    # show_charts = st.sidebar.checkbox("📊 Show Charts Automatically", value=True)
    show_charts = True
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
    
    # ===== Gauges (3 fixed KPIs in one row) =====
    st.subheader("📊 KPI Gauges")
    chart_titles = []  # keep collecting titles for the assign form

    # --- Fixed values you provided (no randomness / no dataframe dependence) ---
    # Revenue: show in thousands with $ prefix
    revenue_val_raw = 125000
    revenue_display_val = int(revenue_val_raw / 1000)           # 125
    revenue_target_raw = 130000
    revenue_target_display = int(revenue_target_raw / 1000)     # 130

    # Customer Satisfaction: 0-5 scale
    csat_val = 4.2
    csat_target = 4.4

    # Product Uptime: percent scale
    uptime_val = 95.0
    uptime_target = 99.0

    col_r, col_c, col_u = st.columns(3)

    with col_r:
        title = "Revenue"
        fig_rev = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=revenue_display_val,
            title={'text': title},
            delta={'reference': revenue_target_display, 'relative': False, 'position': "bottom", 'valueformat': ",d"},
            number={'prefix': "$", 'suffix': "k", 'valueformat': ",d"},
            gauge={
                'axis': {'range': [0, max(revenue_target_display, revenue_display_val) * 1.2]},
                'bar': {'thickness': 0.25},
                'steps': [
                    {'range': [0, revenue_target_display * 0.6], 'color': "rgba(200,200,200,0.15)"},
                    {'range': [revenue_target_display * 0.6, revenue_target_display], 'color': "rgba(0,200,120,0.12)"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': revenue_target_display
                }
            }
        ))
        fig_rev.update_layout(height=240, margin=dict(l=6, r=6, t=24, b=6))
        st.plotly_chart(fig_rev, use_container_width=True)
        chart_titles.append(title)

    with col_c:
        title = "Customer Satisfaction"
        fig_csat = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=round(csat_val, 1),
            title={'text': title},
            delta={'reference': round(csat_target, 1), 'relative': False, 'position': "bottom", 'valueformat': ".1f"},
            number={'valueformat': ".1f"},
            gauge={
                'axis': {'range': [0, 5]},
                'bar': {'thickness': 0.25},
                'steps': [
                    {'range': [0, 3], 'color': "rgba(200,200,200,0.15)"},
                    {'range': [3, 4.5], 'color': "rgba(255,200,0,0.12)"},
                    {'range': [4.5, 5], 'color': "rgba(0,200,120,0.12)"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 3},
                    'thickness': 0.75,
                    'value': round(csat_target, 1)
                }
            }
        ))
        fig_csat.update_layout(height=240, margin=dict(l=6, r=6, t=24, b=6))
        st.plotly_chart(fig_csat, use_container_width=True)
        chart_titles.append(title)

    with col_u:
        title = "Product Uptime"
        fig_uptime = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=round(uptime_val, 1),
            title={'text': title},
            delta={'reference': round(uptime_target, 1), 'relative': False, 'position': "bottom", 'valueformat': ".1f"},
            number={'suffix': " %", 'valueformat': ".1f"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'thickness': 0.25},
                'steps': [
                    {'range': [0, 95], 'color': "rgba(240,120,120,0.12)"},    # red-ish below 95
                    {'range': [95, 99], 'color': "rgba(255,200,0,0.12)"},     # yellow 95-99
                    {'range': [99, 100], 'color': "rgba(0,200,120,0.12)"}     # green >=99
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': round(uptime_target, 1)
                }
            }
        ))
        fig_uptime.update_layout(height=240, margin=dict(l=6, r=6, t=24, b=6))
        st.plotly_chart(fig_uptime, use_container_width=True)
        chart_titles.append(title)


    # ===== Charts (2 per row) =====
    # charts = [
    #     ("Monthly Revenue Trend", px.line(df_filtered, x="Month", y="Revenue", markers=True,
    #                                       title="Monthly Revenue Trend")),
    #     ("Customer Satisfaction by Month", px.bar(df_filtered, x="Month", y="Customer Satisfaction",
    #                                              text="Customer Satisfaction", title="Customer Satisfaction by Month")),
    #     ("Product Uptime Trend", px.line(df_filtered, x="Month", y="Product Uptime", markers=True,
    #                                      title="Product Uptime Trend")),
    #     ("Bug Fix Rate by Month", px.bar(df_filtered, x="Month", y="Bug Fix Rate", text="Bug Fix Rate",
    #                                      title="Bug Fix Rate by Month")),
    #     ("Tickets Resolved Trend", px.line(df_filtered, x="Month", y="Tickets Resolved", markers=True,
    #                                        title="Tickets Resolved Trend")),
    #     ("Revenue vs Satisfaction vs Tickets", px.scatter(
    #         df_filtered, x="Revenue", y="Customer Satisfaction",
    #         size="Tickets Resolved", color="Month",
    #         title="Revenue vs Satisfaction vs Tickets"
    #     ))
    # ]

    # # store all chart titles
    # for t, _ in charts:
    #     chart_titles.append(t)

    # for i in range(0, len(charts), 2):
    #     a, b = st.columns(2)
    #     with a:
    #         st.plotly_chart(charts[i][1], use_container_width=True)
    #     if i + 1 < len(charts):
    #         with b:
    #             st.plotly_chart(charts[i + 1][1], use_container_width=True)
    
    # ===== Four KPI Charts (2x2 layout, tighter y-axes, product usage as line) =====
    st.subheader("📈 KPI Charts")
    chart_titles = chart_titles if 'chart_titles' in locals() else []  # preserve existing list

    # --- Prepare month labels (6 months + 3-month projection for MRR) ---
    months_6 = pd.date_range(end=pd.Timestamp.today().normalize() - pd.Timedelta(days=30), periods=6, freq='M').strftime('%b %Y').tolist()
    months_9 = pd.date_range(end=pd.Timestamp.today().normalize() - pd.Timedelta(days=30), periods=9, freq='M').strftime('%b %Y').tolist()

    # --- Data (exact values you provided) ---
    mrr_hist = [115000, 116000, 117000, 119000, 121000, 125000]   # last 6 months
    mrr_proj = [122000, 121000, 120000]                            # next 3 months
    mrr_vals = mrr_hist + mrr_proj

    usage_vals = [1100, 1150, 1200, 1210, 1150, 1100]
    tickets_vals = [55, 56, 53, 60, 63, 65]
    uptime_vals = [97, 98, 99, 98, 97, 95]

    # --- helper to compute nice y-range with padding ---
    def padded_range(vals, pad_frac=0.18, floor=None, ceiling=None):
        ymin = min(vals)
        ymax = max(vals)
        span = ymax - ymin
        if span == 0:
            span = max(1, abs(ymax) * 0.05)
        pad = span * pad_frac
        lower = ymin - pad
        upper = ymax + pad
        if floor is not None:
            lower = max(lower, floor)
        if ceiling is not None:
            upper = min(upper, ceiling)
        return [lower, upper]

    # Build a 2x2 layout
    row1_col1, row1_col2 = st.columns(2)
    row2_col1, row2_col2 = st.columns(2)

    # ===== 1) MRR (row1_col1) - line with projection (solid history, dashed projection) =====
    with row1_col1:
        fig_mrr = go.Figure()
        # historical
        fig_mrr.add_trace(go.Scatter(x=months_9[:6], y=mrr_hist, mode='lines+markers', name='Historical MRR', line=dict(width=3)))
        # projection (start from last historical point for continuity)
        proj_x = months_9[5:]  # last historical month + next 3
        proj_y = [mrr_hist[-1]] + mrr_proj
        fig_mrr.add_trace(go.Scatter(x=proj_x, y=proj_y, mode='lines+markers', name='Projection', line=dict(dash='dash', width=2)))
        # highlight latest historical point
        fig_mrr.add_trace(go.Scatter(x=[months_9[5]], y=[mrr_hist[-1]], mode='markers+text', name='Latest', text=["Latest"], textposition="top center", marker=dict(size=10)))
        yrange_mrr = padded_range(mrr_vals, pad_frac=0.10, floor=0)
        fig_mrr.update_layout(title="MRR — Last 6 months + 3-month Projection", xaxis_title="Month", yaxis_title="MRR ($)",
                            yaxis=dict(range=yrange_mrr, tickformat=",d"), margin=dict(l=6, r=6, t=36, b=10), height=340)
        st.plotly_chart(fig_mrr, use_container_width=True)
        chart_titles.append("MRR — Last 6 months + 3-month Projection")

    # ===== 2) Product Usage Report (row1_col2) - line chart =====
    with row1_col2:
        fig_usage = go.Figure()
        fig_usage.add_trace(go.Scatter(x=months_6, y=usage_vals, mode='lines+markers', name='DAU', line=dict(width=3)))
        yrange_usage = padded_range(usage_vals, pad_frac=0.14, floor=0)
        fig_usage.update_layout(title="Product Usage (Last 6 Months)", xaxis_title="Month", yaxis_title="Daily Active Users",
                                yaxis=dict(range=yrange_usage), margin=dict(l=6, r=6, t=36, b=10), height=340)
        st.plotly_chart(fig_usage, use_container_width=True)
        chart_titles.append("Product Usage (Last 6 Months)")

    # ===== 3) Support Tickets Raised (row2_col1) - bar + trend line =====
    with row2_col1:
        fig_tickets = go.Figure()
        fig_tickets.add_trace(go.Bar(x=months_6, y=tickets_vals, name='Tickets', marker_line_width=0))
        # 3-month rolling average trend line for clarity
        trend_y = pd.Series(tickets_vals).rolling(3, min_periods=1).mean().tolist()
        fig_tickets.add_trace(go.Scatter(x=months_6, y=trend_y, mode='lines', name='3-mo MA', line=dict(width=2)))
        yrange_tickets = padded_range(tickets_vals, pad_frac=0.18, floor=0)
        fig_tickets.update_layout(title="Support Tickets Raised (Last 6 Months)", xaxis_title="Month", yaxis_title="Tickets",
                                yaxis=dict(range=yrange_tickets), margin=dict(l=6, r=6, t=36, b=10), height=340)
        st.plotly_chart(fig_tickets, use_container_width=True)
        chart_titles.append("Support Tickets Raised (Last 6 Months)")

    # ===== 4) Product Uptime (row2_col2) - area/line with tight y-axis =====
    with row2_col2:
        fig_uptime = go.Figure()
        fig_uptime.add_trace(go.Scatter(x=months_6, y=uptime_vals, mode='lines+markers', fill='tozeroy', name='Uptime', line=dict(width=2)))
        # keep y-axis tight around the values but within 0-100
        yrange_uptime = padded_range(uptime_vals, pad_frac=0.06, floor=90, ceiling=100)
        fig_uptime.update_layout(title="Product Uptime (Last 6 Months)", xaxis_title="Month", yaxis_title="Uptime %",
                                yaxis=dict(range=yrange_uptime), margin=dict(l=6, r=6, t=36, b=10), height=340)
        st.plotly_chart(fig_uptime, use_container_width=True)
        chart_titles.append("Product Uptime (Last 6 Months)")



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
                st.write(f"Query: {alert['query']}")
                st.write(f"Recipients: {', '.join(alert.get('emails', []))}")
                st.write(f"Message: {alert.get('message', '')}")

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
