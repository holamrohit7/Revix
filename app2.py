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
# Chat Page
# ------------------------
if page == "💬 Chat":
    st.title("🤖 Agent Revix (Chat Mode)")

    show_charts = st.sidebar.checkbox("📊 Show Charts Automatically", value=True)

    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.chat_message("user").markdown(msg["content"])
        else:
            st.chat_message("assistant").markdown(msg["content"])

            if "data" in msg:
                components.html(styled_table(msg["data"]), height=280, scrolling=True)

            if "chart" in msg:
                fig = msg["chart"]
                fig.update_layout(template="plotly_dark", margin=dict(l=6, r=6, b=10, t=30))
                st.plotly_chart(fig, use_container_width=True, key=f"chart_{msg['id']}")

    query = st.chat_input("Ask about your data...")

    if query:
        st.session_state.messages.append({"role": "user", "content": query, "id": str(uuid.uuid4())[:8]})
        st.chat_message("user").markdown(query)

        with st.chat_message("assistant"):
            with st.spinner("🤖 Thinking..."):
                if not is_data_related(query):
                    ans = "ℹ I only know your Excel/CSV data."
                    msg_id = str(uuid.uuid4())[:8]
                    st.markdown(ans)
                    st.session_state.messages.append({"role": "assistant", "content": ans, "id": msg_id})
                else:
                    code = generate_pandas_code(query)
                    result = execute_pandas_code(code)

                    if "Error" in result.columns:
                        ans = f"⚠ {result.iloc[0]['Error']}"
                        msg_id = str(uuid.uuid4())[:8]
                        st.markdown(ans)
                        st.session_state.messages.append({"role": "assistant", "content": ans, "id": msg_id})
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
                                    fig.update_layout(template="plotly_dark", margin=dict(l=6, r=6, b=10, t=30))
                                    st.plotly_chart(fig, use_container_width=True)
                                    st.session_state.messages[-1]["chart"] = fig

# ------------------------
# KPI Dashboard
# ------------------------
elif page == "📊 KPI Dashboard":
    st.title("📊 KPI Dashboard")

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

    st.subheader("🔎 Filters")
    col1, col2 = st.columns(2)

    with col1:
        selected_product = st.selectbox("Select Product", ["All"] + sorted(df_kpi["Product"].unique().tolist()))
    with col2:
        selected_month = st.selectbox("Select Period (Month)", ["All"] + df_kpi["Month"].unique().tolist())

    df_filtered = df_kpi.copy()
    if selected_product != "All":
        df_filtered = df_filtered[df_filtered["Product"] == selected_product]
    if selected_month != "All":
        df_filtered = df_filtered[df_filtered["Month"] == selected_month]

    st.subheader("📊 KPI Gauges")
    g1, g2 = st.columns(2)
    with g1:
        fig = go.Figure(go.Indicator(mode="gauge+number", value=df_filtered["Customer Satisfaction"].mean(),
                                     title={'text': "Customer Satisfaction %"}, gauge={'axis': {'range': [0, 100]}}))
        st.plotly_chart(fig, use_container_width=True)
    with g2:
        fig = go.Figure(go.Indicator(mode="gauge+number", value=df_filtered["Product Uptime"].mean(),
                                     title={'text': "Product Uptime %"}, gauge={'axis': {'range': [0, 100]}}))
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("📈 KPI Charts")
    charts = [
        ("Monthly Revenue Trend", px.line(df_filtered, x="Month", y="Revenue", markers=True)),
        ("Customer Satisfaction by Month", px.bar(df_filtered, x="Month", y="Customer Satisfaction")),
        ("Product Uptime Trend", px.line(df_filtered, x="Month", y="Product Uptime", markers=True)),
        ("Bug Fix Rate by Month", px.bar(df_filtered, x="Month", y="Bug Fix Rate")),
        ("Tickets Resolved Trend", px.line(df_filtered, x="Month", y="Tickets Resolved", markers=True)),
        ("Revenue vs Satisfaction vs Tickets", px.scatter(df_filtered, x="Revenue", y="Customer Satisfaction", size="Tickets Resolved", color="Month"))
    ]
    for i in range(0, len(charts), 2):
        a, b = st.columns(2)
        with a:
            st.plotly_chart(charts[i][1], use_container_width=True)
        if i+1 < len(charts):
            with b:
                st.plotly_chart(charts[i+1][1], use_container_width=True)
