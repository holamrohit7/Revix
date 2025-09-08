import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import time
import pandas as pd

st.set_page_config(page_title="Revix Demo", page_icon="🤖", layout="wide")

# -------------------
# Sidebar Navigation
# -------------------
page = st.sidebar.radio("Go to", ["📊 KPI Dashboard", "💬 Chat", "📄 Customer Health", "✅ Action Items", "🔔 Smart Alerts"])

# -------------------
# 1. KPI Dashboard
# -------------------
if page == "📊 KPI Dashboard":
    st.title("📊 KPI Dashboard")

    # Hardcoded KPI values
    csat = 4.3
    uptime = 91
    mrr = [120, 130, 128, 122, 110, 105]
    usage = [1000, 980, 950, 900, 870, 810]  # 10% drop
    tickets = [30, 32, 29, 28, 27, 45]  # spike last month
    months = ["Mar", "Apr", "May", "Jun", "Jul", "Aug"]

    # KPI Gauges
    col1, col2 = st.columns(2)
    with col1:
        fig = go.Figure(go.Indicator(mode="gauge+number", value=csat, title={'text': "CSAT Score"}, gauge={'axis': {'range': [0, 5]}}))
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = go.Figure(go.Indicator(mode="gauge+number", value=uptime, title={'text': "Uptime %"}, gauge={'axis': {'range': [0, 100]}}))
        st.plotly_chart(fig, use_container_width=True)

    # Charts
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(px.line(x=months, y=mrr, markers=True, title="MRR (with 3M forecast drop)"), use_container_width=True)
    with c2:
        st.plotly_chart(px.bar(x=months, y=usage, title="Daily Active Users (-10% decline)"), use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        st.plotly_chart(px.bar(x=months, y=tickets, title="Support Tickets (spike last month)"), use_container_width=True)
    with c4:
        st.plotly_chart(px.line(x=months, y=[10,12,11,9,10,8], markers=True, title="Random Trend"), use_container_width=True)

    st.info("Sarah creates action item for SRE Head to investigate uptime drop.")

# -------------------
# 2. AI Chat
# -------------------
elif page == "💬 Chat":
    st.title("💬 AI Chat Simulation")

    st.markdown("**User Prompt:** Why is our MRR forecast and usage dropping while support tickets are increasing?")

    # Typing effect
    response = (
        "Looking at the last 30 days, the revenue forecast decline is mostly explained by a 10% drop in usage. "
        "Nearly 80% of that drop comes from Alpha Solutions Inc. "
        "What stands out is that Alpha also accounts for about 65% of the overall rise in support tickets. "
        "Other accounts show only small, expected fluctuations — so Alpha is clearly the main driver here."
    )

    placeholder = st.empty()
    full_text = ""
    for char in response:
        full_text += char
        placeholder.markdown(full_text)
        time.sleep(0.01)

# -------------------
# 3. Customer Health Report
# -------------------
elif page == "📄 Customer Health":
    st.title("📄 Customer Health Report - Alpha Solutions Inc")

    st.subheader("Summary")
    st.write("Alpha Solutions shows declining engagement, usage drop, and high support burden.")

    st.subheader("KPIs")
    data = {
        "Metric": ["ACV", "MRR", "Renewal Probability", "Benchmark Usage", "Next Renewal Date"],
        "Value": ["$250k", "$20k", "40% (down from 70%)", "60% below median", "Dec 2025"]
    }
    st.table(pd.DataFrame(data))

    st.subheader("Engagement Timeline")
    st.write("Salesforce notes: 3 calls last quarter, 1 exec escalation.")

    st.subheader("Usage Trend")
    st.line_chart([1200, 1150, 1100, 950, 870, 810])

    st.subheader("Support Themes")
    st.write("- 15 API-related tickets\n- 5 uptime complaints")

    st.subheader("Customer Feedback")
    st.info("\"API integration keeps breaking, slowing our team.\"")
    st.info("\"We logged multiple API issues last month.\"")
    st.info("\"Overall support experience has been slow.\"")
    st.info("\"We also faced uptime problems recently.\"")

# -------------------
# 4. Action Items
# -------------------
elif page == "✅ Action Items":
    st.title("✅ Action Item Tracker")

    actions = [
        {"Owner": "Product Head & Engineering Head", "Action": "Investigate recurring API issues"},
        {"Owner": "CSM Head", "Action": "Schedule call with Alpha Solutions"},
        {"Owner": "Sales Head", "Action": "Plan strategy for renewal risk (40% probability)"},
        {"Owner": "SRE Head", "Action": "Investigate uptime drop"},
    ]

    df = pd.DataFrame(actions)
    st.table(df)

    st.success("Action items assigned across functions.")

# -------------------
# 5. Smart Alerts
# -------------------
elif page == "🔔 Smart Alerts":
    st.title("🔔 Smart Alert Builder")

    st.subheader("Example Alert Rule")
    st.code("If Account Value > $100k AND Usage Drop > 20% AND Tickets Increase > 20%")

    st.write("Configured Recipients: Product Head, Engineering Head, CSM Head")

    st.warning("This alert will automatically flag risky enterprise accounts like Alpha Solutions.")
