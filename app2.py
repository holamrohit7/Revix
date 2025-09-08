import os
import pickle
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import streamlit.components.v1 as components
from dotenv import load_dotenv
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
# Smart Memory
# ------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "actions" not in st.session_state:
    st.session_state.actions = []

# ------------------------
# 💬 Chat Window (Step 2: Investigation)
# ------------------------
if page == "💬 Chat":
    st.title("🤖 Agent Revix (Chat Mode)")

    # Predefined sample prompts
    st.markdown("### 💡 Sample Prompts")
    st.markdown("- Why is our MRR forecast, and usage is dropping while support tickets are increasing?")

    # Render past messages
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.chat_message("user").markdown(msg["content"])
        else:
            st.chat_message("assistant").markdown(msg["content"])

    # --- new query ---
    query = st.chat_input("Ask about your data...")

    if query:
        st.session_state.messages.append({"role": "user", "content": query})
        st.chat_message("user").markdown(query)

        with st.chat_message("assistant"):
            with st.spinner("🤖 Thinking..."):
                time.sleep(2)  # simulate AI analysis delay
                if "mrr" in query.lower() and "usage" in query.lower() and "tickets" in query.lower():
                    response = (
                        "Looking at the last 30 days, the revenue forecast decline is mostly explained by a 10% drop "
                        "in usage. Nearly 80% of that drop comes from **Alpha Solutions Inc**. What stands out is that "
                        "Alpha also accounts for about 65% of the overall rise in support tickets. Other accounts show "
                        "only small, expected fluctuations — so Alpha is clearly the main driver here."
                    )
                else:
                    response = "This is a demo mode. Please try the sample prompt for full scripted response."

                # fake typing effect
                placeholder = st.empty()
                typed = ""
                for char in response:
                    typed += char
                    placeholder.markdown(typed)
                    time.sleep(0.01)

                st.session_state.messages.append({"role": "assistant", "content": response})

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

    df_filtered = df_kpi.copy()
    if selected_product != "All":
        df_filtered = df_filtered[df_filtered["Product"] == selected_product]
    if selected_month != "All":
        df_filtered = df_filtered[df_filtered["Month"] == selected_month]

    st.subheader("📊 KPI Gauges")
    g1, g2 = st.columns(2)
    with g1:
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=df_filtered["Customer Satisfaction"].mean(),
            title={'text': "Customer Satisfaction %"},
            gauge={'axis': {'range': [0, 100]}}
        ))
        st.plotly_chart(fig, use_container_width=True)
    with g2:
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=df_filtered["Product Uptime"].mean(),
            title={'text': "Product Uptime %"},
            gauge={'axis': {'range': [0, 100]}}
        ))
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("📈 KPI Charts")
    charts = [
        ("Monthly Revenue Trend", px.line(df_filtered, x="Month", y="Revenue", markers=True)),
        ("Customer Satisfaction by Month", px.bar(df_filtered, x="Month", y="Customer Satisfaction", text="Customer Satisfaction")),
        ("Product Uptime Trend", px.line(df_filtered, x="Month", y="Product Uptime", markers=True)),
        ("Bug Fix Rate by Month", px.bar(df_filtered, x="Month", y="Bug Fix Rate", text="Bug Fix Rate")),
        ("Tickets Resolved Trend", px.line(df_filtered, x="Month", y="Tickets Resolved", markers=True)),
        ("Revenue vs Satisfaction vs Tickets", px.scatter(
            df_filtered, x="Revenue", y="Customer Satisfaction",
            size="Tickets Resolved", color="Month"))
    ]
    for i in range(0, len(charts), 2):
        a, b = st.columns(2)
        with a:
            st.plotly_chart(charts[i][1], use_container_width=True)
        if i + 1 < len(charts):
            with b:
                st.plotly_chart(charts[i + 1][1], use_container_width=True)

# ------------------------
# Action Item Tracker
# ------------------------
elif page == "✅ Action Item Tracker":
    st.title("✅ Action Item Tracker")
    st.info("Demo only — action items will appear here when assigned.")

# ------------------------
# Smart Alerts
# ------------------------
elif page == "🔔 Smart Alerts":
    st.title("🔔 Smart Alerts")
    st.info("Demo only — alert rules can be added here.")
