import os
import pickle
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import calendar
import numpy as np
import streamlit.components.v1 as components
from dotenv import load_dotenv
from groq import Groq
import uuid
from datetime import date

# ------------------------
# Custom CSS
# ------------------------
st.markdown("""
<style>
    /* Page background & font */
    html, body, [class*="css"]  {
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial;
        background: linear-gradient(180deg, #f8fafc 0%, #ffffff 60%);
    }

    /* Main Layout */
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 1rem;
    }

    /* Headers */
    .main-header {
        font-size: 2.5rem;
        color: #0f172a; /* deep slate */
        text-align: center;
        margin-bottom: 0.2rem;
        font-weight: 700;
    }
    .main-subtitle {
        color: #475569; /* muted slate */
        text-align: center;
        font-size: 1.05rem;
        margin-bottom: 1.6rem;
    }

    /* KPI Cards */
    .kpi-card {
        background: linear-gradient(180deg, #ffffff, #fbfdff);
        border-radius: 0.6rem;
        padding: 1.25rem;
        box-shadow: 0 6px 18px rgba(2,6,23,0.08);
        text-align: center;
        height: 100%;
    }
    .kpi-card h3 {
        margin: 0;
        font-size: 0.9rem;
        color: #64748b;
    }
    .kpi-value {
        font-size: 1.6rem;
        font-weight: 700;
        color: #0f172a;
        margin: 0.4rem 0;
    }
    .kpi-target {
        font-size: 0.85rem;
        color: #94a3b8;
    }

    /* Charts */
    .chart-container {
        background: white;
        border-radius: 0.5rem;
        padding: 0.85rem;
        box-shadow: 0 6px 18px rgba(2,6,23,0.04);
        margin-bottom: 1rem;
    }

    /* Form */
    .form-container {
        background: white;
        border-radius: 0.5rem;
        padding: 1rem;
        box-shadow: 0 6px 18px rgba(2,6,23,0.04);
    }
    .stButton > button {
        width: 100%;
        background-color: #0ea5a4; /* teal accent */
        color: white;
        border: none;
    }

    /* Small tweaks to make plotly cards blend */
    .plotly-graph-div .main-svg { background: transparent !important; }
</style>
""", unsafe_allow_html=True)

# ------------------------
# Setup
# ------------------------
st.set_page_config(page_title="Revix Analytics", page_icon="🤖", layout="wide")


# ------------------------
# Sidebar Navigation
# ------------------------
with st.sidebar:
    # Professional Header with Enhanced Styling
    st.markdown("""
        <div style='padding: 1.2rem; background: linear-gradient(135deg, #1e293b, #0f172a); 
             border-radius: 12px; margin-bottom: 1.5rem; text-align: center; 
             box-shadow: 0 4px 12px rgba(2,6,23,0.15)'>
            <h2 style='margin:0; color: #ffffff; font-size: 1.6rem; font-weight: 600;
                      text-shadow: 0 2px 4px rgba(0,0,0,0.1);'>
                🤖 Revix Analytics
            </h2>
            <p style='margin:6px 0 0 0; color: #94a3b8; font-size: 0.95rem;
                     letter-spacing: 0.5px;'>
                AI-Powered Analytics
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # Enhanced Navigation Menu
    st.markdown("""
        <style>
            div[data-testid="stRadio"] > label {
                font-weight: 600;
                color: #1e293b;
                font-size: 1.05rem;
                padding: 0.3rem 0;
            }
            div[data-testid="stRadio"] > div[role="radiogroup"] > label {
                padding: 0.6rem;
                border-radius: 8px;
                transition: all 0.2s;
            }
            div[data-testid="stRadio"] > div[role="radiogroup"] > label:hover {
                background: #f1f5f9;
            }
        </style>
    """, unsafe_allow_html=True)
    
    selected_page = st.radio(
        "Navigation",
        ["📊 KPI Dashboard", "💬 Chat", "✅ Action Item Tracker", "🔔 Smart Alerts"],
        label_visibility="collapsed"
    )
    
    # Enhanced Dashboard Settings
    if selected_page == "📊 KPI Dashboard":
        st.divider()
        with st.expander("⚙️ Dashboard Settings", expanded=True):
            st.markdown("""
                <style>
                    div[data-testid="stExpander"] {
                        border: none;
                        box-shadow: 0 2px 8px rgba(2,6,23,0.08);
                        border-radius: 8px;
                    }
                </style>
            """, unsafe_allow_html=True)
            
            # KPI Dashboard submenu with session state
            st.markdown("##### 📊 Display Options")
            if 'show_kpi_charts' not in st.session_state:
                st.session_state.show_kpi_charts = True
            if 'show_debug' not in st.session_state:
                st.session_state.show_debug = False
                
            st.session_state.show_kpi_charts = st.checkbox("📈 Show KPI Charts", 
                value=st.session_state.show_kpi_charts,
                help="Toggle visibility of performance charts below the KPI gauges")
            
            st.markdown("##### 🛠️ Advanced")
            st.session_state.show_debug = st.checkbox("🔍 Debug Mode",
                value=st.session_state.show_debug,
                help="Show additional debugging information")
    
    st.divider()
    
    # Enhanced Help/Support footer
    st.markdown("""
        <div style='margin-top: 2rem; padding: 1rem; 
             background: linear-gradient(180deg, #f8fafc, #f1f5f9);
             border-radius: 8px; text-align: center;
             box-shadow: 0 2px 6px rgba(2,6,23,0.05)'>
            <p style='margin:0; font-size: 0.9rem; color: #475569; font-weight: 500;'>
                Need help? 
                <span style='color: #0ea5e9; text-decoration: underline; cursor: pointer;'>
                    Contact support
                </span>
            </p>
        </div>
    """, unsafe_allow_html=True)

# Set active page
page = selected_page
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

# Initialize session state keys used throughout the app to avoid AttributeError
if 'plotly_dark' not in st.session_state:
    st.session_state.plotly_dark = True
if 'last_result' not in st.session_state:
    st.session_state.last_result = None
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'actions' not in st.session_state:
    st.session_state.actions = []
if 'alerts' not in st.session_state:
    st.session_state.alerts = []


if page == "📊 KPI Dashboard":
    # Header
    st.markdown("<h1 class='main-header'>Revix Analytics</h1>", unsafe_allow_html=True)
    st.markdown("<p class='main-subtitle'>AI-Powered Data Insights</p>", unsafe_allow_html=True)
    
    # Filters
    st.subheader("🔎 Filters")
    col1, col2 = st.columns(2)

    # Build product options from actual DataFrame 'Product' columns (ignore datasource tokens)
    def gather_products(dfs, ignore_tokens=None):
        ignore_tokens = set([t.lower() for t in (ignore_tokens or [])])
        prods = set()
        # Prefer explicit 'Product' column when present
        for name, df in dfs.items():
            for col in df.columns:
                if col.lower() == "product":
                    vals = df[ col ].dropna().astype(str).str.strip()
                    for v in vals.unique():
                        if v and v.lower() not in ignore_tokens:
                            prods.add(v)
                    break

        # Fallback: look for other candidate columns if nothing found
        if not prods:
            candidates = ["product", "prod", "product_code", "product_name", "sku", "item", "plan"]
            for name, df in dfs.items():
                for c in df.columns:
                    if c.lower() in candidates:
                        vals = df[c].dropna().astype(str).str.strip()
                        for v in vals.unique():
                            if v and v.lower() not in ignore_tokens:
                                prods.add(v)
        return sorted(prods)

    datasource_tokens = ["salesforce", "revenue", "gainsight", "usage", "jira"]
    product_options = ["All"] + gather_products(dataframes, ignore_tokens=datasource_tokens)

    with col1:
        product = st.selectbox("Product", options=product_options, key="product_main")

    with col2:
        month = st.selectbox("Month", options=["All", "January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"], key="month_main")

    # KPI Gauges (dynamic from selected Product / Month)
    st.subheader("📊 KPI Gauges")
    kpi_cols = st.columns(5)

    # Helpers to filter by product and month
    month_name_to_num = {m: i for i, m in enumerate(calendar.month_name) if m}

    def df_matches_month(series, month_name):
        # Try parsing datelike values, else try extracting MM from YYYY-MM
        try:
            parsed = pd.to_datetime(series, errors='coerce')
            if parsed.notna().any():
                return parsed.dt.month == month_name_to_num.get(month_name, -1)
        except Exception:
            pass
        # fallback: strings like YYYY-MM or YYYY-M
        s = series.astype(str).str.strip()
        mm = s.str.extract(r"-(\d{1,2})$")
        if not mm.empty:
            mm = mm[0].astype(float).fillna(-1).astype(int)
            return mm == month_name_to_num.get(month_name, -1)
        return pd.Series([False] * len(series), index=series.index)

    def filter_df_for_selection(df, sel_product, sel_month):
        d = df.copy()
        # product filter
        prod_cols = [c for c in d.columns if c.lower() == 'product']
        if sel_product and sel_product != 'All' and prod_cols:
            col = prod_cols[0]
            d = d[d[col].astype(str).str.strip().str.lower() == sel_product.strip().lower()]

        # month filter
        if sel_month and sel_month != 'All':
            month_cols = [c for c in d.columns if c.lower() == 'month' or 'date' in c.lower()]
            if month_cols:
                mcol = month_cols[0]
                mask = df_matches_month(d[mcol], sel_month)
                d = d[mask]
        return d

    # Build combined frame from all DataFrames that have a Product column
    frames = []
    for name, df in dataframes.items():
        if any(c.lower() == 'product' for c in df.columns):
            try:
                sub = filter_df_for_selection(df, product, month)
                if not sub.empty:
                    sub['_source_df'] = name
                    frames.append(sub)
            except Exception:
                continue

    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    # Compute dynamic KPIs where possible
    def safe_sum_cols(df, candidates):
        for c in df.columns:
            if any(k in c.lower() for k in candidates):
                try:
                    return df[c].dropna().astype(float).sum()
                except Exception:
                    continue
        return None

    revenue_val = None
    health_val = None
    uptime_val = None
    bugfix_val = None
    tickets_val = None

    # When both filters are "All", show 100% values
    if product == "All" and month == "All":
        revenue_val = 150000  # Target revenue
        health_val = 100     # Perfect satisfaction
        uptime_val = 100     # Perfect uptime
        bugfix_val = 100     # Perfect fix rate
        tickets_val = 200    # Target tickets resolved
    elif not combined.empty:
        # Revenue-like columns
        revenue_val = safe_sum_cols(combined, ['revenue', 'amount', 'price', 'sales'])
        # Health/CSAT like
        for c in combined.columns:
            if 'health' in c.lower() or 'csat' in c.lower() or 'satisfaction' in c.lower():
                try:
                    health_val = float(combined[c].dropna().astype(float).mean())
                    break
                except Exception:
                    continue
        # Uptime (heuristic: uptime/availability columns)
        def parse_percent_like(series):
            """Try to coerce values like '99.5%', '0.995', '99.5' into a 0-100 float percent."""
            s = series.dropna().astype(str).str.strip()
            if s.empty:
                return None
            # strip percent sign
            s_nopct = s.str.rstrip('%')
            # try float conversion
            try:
                vals = pd.to_numeric(s_nopct, errors='coerce').dropna()
                if vals.empty:
                    return None
                # if values mostly <=1 assume fractional (0.995) => convert to percent
                med = vals.median()
                if med <= 1.0:
                    vals = vals * 100.0
                return float(vals.mean())
            except Exception:
                return None

        uptime_val = None
        for c in combined.columns:
            if any(k in c.lower() for k in ['uptime', 'availability', 'service_level']):
                try:
                    parsed = parse_percent_like(combined[c])
                    if parsed is not None:
                        uptime_val = parsed
                        break
                except Exception:
                    continue
        # fallback: try searching other dataframes for uptime-like columns
        if uptime_val is None:
            for name, df in dataframes.items():
                for c in df.columns:
                    if any(k in c.lower() for k in ['uptime', 'availability', 'service_level']):
                        try:
                            parsed = parse_percent_like(df[c])
                            if parsed is not None:
                                uptime_val = parsed
                                break
                        except Exception:
                            continue
                if uptime_val is not None:
                    break

        # Secondary fallback: use Health Score or CSAT Score as a proxy for uptime (0-100 scale expected)
        if uptime_val is None:
            for name, df in dataframes.items():
                for c in df.columns:
                    if any(k in c.lower() for k in ['health score', 'health', 'csat', 'csat score', 'satisfaction']):
                        try:
                            vals = pd.to_numeric(df[c].dropna(), errors='coerce').dropna()
                            if not vals.empty:
                                uptime_val = float(vals.mean())
                                break
                        except Exception:
                            continue
                if uptime_val is not None:
                    break

        # Tertiary fallback: derive a resolution-rate proxy from jira/salesforce (closed/resolved ratio)
        if uptime_val is None:
            # check jira
            if 'jira' in dataframes:
                jdf = dataframes['jira']
                status_cols = [c for c in jdf.columns if 'status' in c.lower() or 'resolution' in c.lower() or 'resolved' in c.lower()]
                if status_cols:
                    try:
                        s = jdf[status_cols[0]].astype(str).str.lower()
                        closed = s.isin(['closed', 'resolved', 'done']).sum()
                        total = len(s)
                        if total > 0:
                            uptime_val = round((closed / total) * 100.0, 2)
                    except Exception:
                        pass
            # check salesforce if still none
            if uptime_val is None and 'salesforce' in dataframes:
                sdf = dataframes['salesforce']
                status_cols = [c for c in sdf.columns if 'status' in c.lower() or 'resolved' in c.lower()]
                if status_cols:
                    try:
                        s = sdf[status_cols[0]].astype(str).str.lower()
                        closed = s.isin(['closed', 'resolved', 'done']).sum()
                        total = len(s)
                        if total > 0:
                            uptime_val = round((closed / total) * 100.0, 2)
                    except Exception:
                        pass
        # Tickets resolved: from salesforce-like support rows
        if 'Status' in combined.columns:
            try:
                tickets_val = int((combined['Status'].astype(str).str.lower() == 'closed').sum())
            except Exception:
                tickets_val = None

        # Bug Fix Rate heuristic: look for resolved/closed vs open/total from issue-like datasets
        # Strategy: if there are columns like 'status' or 'resolved' or 'issue status', compute closed/total
        bug_fix_rate = None
        status_cols = [c for c in combined.columns if 'status' in c.lower()]
        if status_cols:
            try:
                s = combined[status_cols[0]].astype(str).str.lower()
                closed = s.isin(['closed', 'resolved', 'done']).sum()
                total = len(s)
                if total > 0:
                    bugfix_val = round((closed / total) * 100, 2)
            except Exception:
                bugfix_val = None
        else:
            # fallback: look for numeric columns with 'fix' keyword
            for c in combined.columns:
                if any(k in c.lower() for k in ['fix', 'fix_rate', 'fixes', 'resolved']):
                    try:
                        bugfix_val = float(combined[c].dropna().astype(float).mean())
                        break
                    except Exception:
                        continue

    # Fallback static defaults when calculation not possible
    kpi_data = [
        {"title": "Revenue", "value": revenue_val if revenue_val is not None else "-", "target": 150000, "color": "#10B981"},
        {"title": "Customer Satisfaction", "value": health_val if health_val is not None else "-", "target": 100, "color": "#3B82F6"},
        {"title": "Product Uptime", "value": uptime_val if uptime_val is not None else "-", "target": 100, "color": "#6366F1"},
        {"title": "Bug Fix Rate", "value": bugfix_val if bugfix_val is not None else "-", "target": 100, "color": "#F59E0B"},
        {"title": "Tickets Resolved", "value": tickets_val if tickets_val is not None else "-", "target": 200, "color": "#EF4444"}
    ]

    for col, kpi in zip(kpi_cols, kpi_data):
        with col:
            # Prepare a user-friendly display value with one decimal when numeric
            if isinstance(kpi['value'], (int, float)):
                if kpi['target'] == 100:
                    display_value = f"{kpi['value']:.1f}%"
                else:
                    display_value = f"{kpi['value']:.1f}"
            else:
                display_value = kpi['value']

            st.markdown(f"""
                <div class='kpi-card'>
                    <h3>{kpi['title']}</h3>
                    <div class='kpi-value'>{display_value}</div>
                    <div class='kpi-target'>/ {kpi['target']:,}</div>
                </div>
            """, unsafe_allow_html=True)

            # Gauge uses a numeric percent between 0 and 100
            if isinstance(kpi['value'], (int, float)) and isinstance(kpi['target'], (int, float)) and kpi['target']:
                pct = min(100.0, max(0.0, (float(kpi['value']) / float(kpi['target'])) * 100.0))
            else:
                pct = 0.0

            fig = go.Figure(go.Indicator(mode="gauge+number", value=round(pct, 1), gauge={'axis': {'range': [None, 100]}, 'bar': {'color': kpi['color']}}))
            fig.update_layout(height=140, margin=dict(l=8, r=8, t=8, b=8))
            st.plotly_chart(fig, use_container_width=True)

    # KPI Charts (reflect selection)
    # Handle data availability message
    if combined.empty and selected_page == "📊 KPI Dashboard":
        st.info('No data matches the selected Product/Month. Try selecting a different product or "All".')
    
    # Show charts section if enabled
    if st.session_state.show_kpi_charts and selected_page == "📊 KPI Dashboard" and not combined.empty:
        st.subheader("📈 KPI Charts")
        
        # Show debug info if enabled
        if st.session_state.show_debug:
            with st.expander("🔍 Debug Information", expanded=True):
                st.write("Active Filters:")
                st.json({
                    "Product": product,
                    "Month": month,
                    "Data Rows": len(combined),
                    "Available Metrics": [col for col in combined.columns]
                })

        # Create chart columns for layout
        chart_cols = st.columns(2)
        # Build four KPI line charts from combined data
        # ensure we have a datetime column
        time_cols = [c for c in combined.columns if 'date' in c.lower() or c.lower() == 'month']
        if time_cols:
            tcol = time_cols[0]
            ts = combined.copy()
            ts['_dt'] = pd.to_datetime(ts[tcol], errors='coerce')
            ts = ts.dropna(subset=['_dt'])
            if not ts.empty:
                # normalize month bucket
                ts['month'] = ts['_dt'].dt.to_period('M').dt.to_timestamp()

                # Monthly Revenue trend
                rev_col = None
                for c in ts.columns:
                    if any(k in c.lower() for k in ['revenue', 'amount', 'sales']):
                        rev_col = c
                        break
                if rev_col:
                    rev_trend = ts.groupby('month')[rev_col].sum().reset_index()
                    fig_rev = px.line(rev_trend, x='month', y=rev_col, title='Monthly Revenue Trend')
                    # single smooth trend line, no markers
                    fig_rev.update_traces(mode='lines', line=dict(width=3, color='#10B981'))
                    fig_rev.update_layout(yaxis_title='Revenue', xaxis_title='Month')
                    # sleek transparent background and minimal grid
                    fig_rev.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
                    fig_rev.update_xaxes(showgrid=False)
                    fig_rev.update_yaxes(showgrid=False)
                    if st.session_state.plotly_dark:
                        fig_rev.update_layout(template='plotly_dark')
                    chart_cols[0].plotly_chart(fig_rev, use_container_width=True)
                else:
                    # Fallback: use first numeric column as revenue proxy
                    num_cols = ts.select_dtypes(include=['number']).columns.tolist()
                    if num_cols:
                        rev_col = num_cols[0]
                        rev_trend = ts.groupby('month')[rev_col].sum().reset_index()
                        fig_rev = px.line(rev_trend, x='month', y=rev_col, title=f'Monthly Revenue Trend ({rev_col})')
                        fig_rev.update_traces(mode='lines', line=dict(width=3, color='#10B981'))
                        fig_rev.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
                        fig_rev.update_xaxes(showgrid=False)
                        fig_rev.update_yaxes(showgrid=False)
                        if st.session_state.plotly_dark:
                            fig_rev.update_layout(template='plotly_dark')
                        chart_cols[0].plotly_chart(fig_rev, use_container_width=True)
                    else:
                        chart_cols[0].info('Monthly Revenue trend not available for selection.')

                # Product Uptime Trend (use parse_percent_like on candidate columns or proxy)
                uptime_series = None
                for c in ts.columns:
                    if any(k in c.lower() for k in ['uptime', 'availability', 'service_level']):
                        # per-month parsed percent
                        def _month_pct(d):
                            return parse_percent_like(d[c])
                        uptime_series = ts.groupby('month').apply(lambda d: _month_pct(d)).reset_index(name='uptime')
                        break
                if uptime_series is None:
                    # try health/csat as proxy
                    for c in ts.columns:
                        if any(k in c.lower() for k in ['health', 'csat', 'satisfaction']):
                            uptime_series = ts.groupby('month')[c].mean().reset_index().rename(columns={c: 'uptime'})
                            break
                if uptime_series is not None:
                    fig_up = px.line(uptime_series, x='month', y='uptime', title='Product Uptime Trend')
                    fig_up.update_traces(mode='lines', line=dict(width=3, color='#6366F1'))
                    fig_up.update_layout(yaxis_title='Uptime (%)', xaxis_title='Month')
                    fig_up.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
                    fig_up.update_xaxes(showgrid=False)
                    fig_up.update_yaxes(showgrid=False)
                    if st.session_state.plotly_dark:
                        fig_up.update_layout(template='plotly_dark')
                    chart_cols[1].plotly_chart(fig_up, use_container_width=True)
                else:
                    chart_cols[1].info('Product Uptime trend not available for selection.')

                # Customer Satisfaction by month
                csat_col = None
                for c in ts.columns:
                    if any(k in c.lower() for k in ['csat', 'satisfaction', 'health score', 'health']):
                        csat_col = c
                        break
                if csat_col:
                    csat_trend = ts.groupby('month')[csat_col].mean().reset_index()
                    fig_cs = px.line(csat_trend, x='month', y=csat_col, title='Customer Satisfaction by Month')
                    fig_cs.update_traces(mode='lines', line=dict(width=3, color='#3B82F6'))
                    fig_cs.update_layout(yaxis_title='Satisfaction (score)', xaxis_title='Month')
                    fig_cs.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
                    fig_cs.update_xaxes(showgrid=False)
                    fig_cs.update_yaxes(showgrid=False)
                    if st.session_state.plotly_dark:
                        fig_cs.update_layout(template='plotly_dark')
                    chart_cols[0].plotly_chart(fig_cs, use_container_width=True)
                else:
                    chart_cols[0].info('Customer Satisfaction by month not available for selection.')

                # Bug Fix Rate by month
                bug_col = None
                status_cols = [c for c in ts.columns if 'status' in c.lower()]
                if status_cols:
                    s = ts.copy()
                    sc = status_cols[0]
                    s['is_closed'] = s[sc].astype(str).str.lower().isin(['closed', 'resolved', 'done']).astype(int)
                    bug_trend = s.groupby('month')['is_closed'].mean().reset_index()
                    bug_trend['fix_rate'] = bug_trend['is_closed'] * 100.0
                    fig_bug = px.line(bug_trend, x='month', y='fix_rate', title='Bug Fix Rate by Month')
                    fig_bug.update_traces(mode='lines', line=dict(width=3, color='#F59E0B'))
                    fig_bug.update_layout(yaxis_title='Fix Rate (%)', xaxis_title='Month')
                    fig_bug.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
                    fig_bug.update_xaxes(showgrid=False)
                    fig_bug.update_yaxes(showgrid=False)
                    if st.session_state.plotly_dark:
                        fig_bug.update_layout(template='plotly_dark')
                    chart_cols[1].plotly_chart(fig_bug, use_container_width=True)
                else:
                    # fallback to numeric fix-like columns
                    for c in ts.columns:
                        if any(k in c.lower() for k in ['fix', 'fix_rate', 'fixes', 'resolved']):
                            bug_col = c
                            break
                    if bug_col:
                        bug_trend = ts.groupby('month')[bug_col].mean().reset_index()
                        fig_bug2 = px.line(bug_trend, x='month', y=bug_col, title='Bug Fix Rate by Month')
                        fig_bug2.update_traces(mode='lines', line=dict(width=3, color='#F59E0B'))
                        fig_bug2.update_layout(yaxis_title='Fix Rate', xaxis_title='Month')
                        fig_bug2.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
                        fig_bug2.update_xaxes(showgrid=False)
                        fig_bug2.update_yaxes(showgrid=False)
                        if st.session_state.plotly_dark:
                            fig_bug2.update_layout(template='plotly_dark')
                        chart_cols[1].plotly_chart(fig_bug2, use_container_width=True)
                    else:
                        chart_cols[1].info('Bug Fix Rate by month not available for selection.')

                # Tickets Resolved by month (new chart)
                # Tickets Resolved by month
                tickets_trend = None
                status_candidates = [c for c in ts.columns if 'status' in c.lower() or 'resolved' in c.lower()]
                if status_candidates:
                    status_col = status_candidates[0]
                    tdf = ts.copy()
                    tdf['is_resolved'] = tdf[status_col].astype(str).str.lower().isin(['closed', 'resolved', 'done']).astype(int)
                    tickets_trend = tdf.groupby('month')['is_resolved'].sum().reset_index().rename(columns={'is_resolved': 'tickets_resolved'})
                else:
                    # fallback: look for 'resolved'/'closed' across original dataframes
                    for name, odf in dataframes.items():
                        if any('status' in c.lower() or 'resolved' in c.lower() for c in odf.columns):
                            oc = [c for c in odf.columns if 'status' in c.lower() or 'resolved' in c.lower()][0]
                            odf2 = odf.copy()
                            # attempt to find a date/month column
                            timecols = [c for c in odf2.columns if 'date' in c.lower() or c.lower() == 'month']
                            if timecols:
                                dtc = timecols[0]
                                odf2['_dt'] = pd.to_datetime(odf2[dtc], errors='coerce')
                                odf2 = odf2.dropna(subset=['_dt'])
                                odf2['month'] = odf2['_dt'].dt.to_period('M').dt.to_timestamp()
                                odf2['is_resolved'] = odf2[oc].astype(str).str.lower().isin(['closed', 'resolved', 'done']).astype(int)
                                tickets_trend = odf2.groupby('month')['is_resolved'].sum().reset_index().rename(columns={'is_resolved': 'tickets_resolved'})
                                break

                if tickets_trend is not None and not tickets_trend.empty:
                    fig_tickets = px.bar(tickets_trend, x='month', y='tickets_resolved', title='Tickets Resolved by Month', color='tickets_resolved', color_continuous_scale=px.colors.sequential.Viridis)
                    fig_tickets.update_layout(yaxis_title='Tickets Resolved', xaxis_title='Month')
                    fig_tickets.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
                    fig_tickets.update_xaxes(showgrid=False)
                    fig_tickets.update_yaxes(showgrid=False)
                    if st.session_state.plotly_dark:
                        fig_tickets.update_layout(template='plotly_dark')
                    chart_cols[0].plotly_chart(fig_tickets, use_container_width=True)
                else:
                    chart_cols[0].info('Tickets Resolved by month not available for selection.')

                # Combined: Revenue vs Satisfaction vs Tickets (multi-axis / area)
                # Build a combined monthly DF with revenue, satisfaction, tickets
                combined_metrics = []
                if rev_col:
                    revm = ts.groupby('month')[rev_col].sum().reset_index().rename(columns={rev_col: 'revenue'})
                    combined_metrics.append(revm.set_index('month'))
                if csat_col:
                    csatm = ts.groupby('month')[csat_col].mean().reset_index().rename(columns={csat_col: 'satisfaction'})
                    combined_metrics.append(csatm.set_index('month'))
                if 'is_resolved' in locals() or 'is_resolved' in ts.columns:
                    # ensure tickets_trend exists
                    try:
                        tmetric = tickets_trend.set_index('month')
                        combined_metrics.append(tmetric)
                    except Exception:
                        pass

                if combined_metrics:
                    # merge all on month
                    df_comb = pd.concat(combined_metrics, axis=1).reset_index()
                    # normalize scales for plotting: revenue on secondary axis
                    fig_combo = go.Figure()
                    if 'revenue' in df_comb.columns:
                        fig_combo.add_trace(go.Bar(x=df_comb['month'], y=df_comb['revenue'], name='Revenue', marker_color='#10B981', yaxis='y1', opacity=0.9))
                    if 'satisfaction' in df_comb.columns:
                        fig_combo.add_trace(go.Scatter(x=df_comb['month'], y=df_comb['satisfaction'], name='Satisfaction', mode='lines', line=dict(color='#3B82F6', width=3), yaxis='y2'))
                    if 'tickets_resolved' in df_comb.columns:
                        fig_combo.add_trace(go.Scatter(x=df_comb['month'], y=df_comb['tickets_resolved'], name='Tickets Resolved', mode='lines', line=dict(color='#6366F1', width=3, dash='dash'), yaxis='y2'))

                    # Layout with two y axes, transparent
                    fig_combo.update_layout(title='Revenue vs Satisfaction vs Tickets', xaxis=dict(title='Month', showgrid=False), yaxis=dict(title='Revenue', side='left', showgrid=False), yaxis2=dict(title='Score / Tickets', overlaying='y', side='right', showgrid=False))
                    fig_combo.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
                    if st.session_state.plotly_dark:
                        fig_combo.update_layout(template='plotly_dark')
                    chart_cols[1].plotly_chart(fig_combo, use_container_width=True)
            else:
                chart_cols[0].info('No parsable date/month column found for KPI charts.')
        else:
            chart_cols[0].info('No date/month column found for KPI charts.')
    else:
        st.info('No data matches the selected Product/Month. Try selecting a different product or "All".')

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
            # assistant message
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
                            "status": "Assigned"
                        }
                        st.session_state.actions.append(action)
                        st.success("✅ Action Assigned!")
                        st.rerun()

    # --- new query ---
    query = st.chat_input("Ask about your data...", key="chat_input_main")

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
                                priority = st.selectbox("Priority", ["Low", "Medium", "High"], index=1, key=f"priority_{msg_id}")  # ✅ Added

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
                                priority = st.selectbox("Priority", ["Low", "Medium", "High"], index=1, key=f"priority_{msg_id}")  # ✅ Added

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
    pass

    # Build chart_titles dynamically from session actions, messages, and dataframe names/columns
    chart_titles = []

    # 1) from previously created actions (chart_title field)
    for act in st.session_state.get("actions", []):
        ct = act.get("chart_title")
        if ct:
            chart_titles.append(ct)

    # 2) from recent assistant messages that attached charts (if they include a 'chart_title')
    for m in st.session_state.get("messages", []):
        if m.get("role") == "assistant" and m.get("chart") and m.get("id"):
            # if the message stored a title earlier
            ct = m.get("chart_title")
            if ct:
                chart_titles.append(ct)

    # 3) from dataframes: add dataframe names and a couple of numeric columns
    try:
        for name, df in list(dataframes.items())[:8]:
            chart_titles.append(name)
            num_cols = df.select_dtypes(include=["number"]).columns.tolist()
            for c in num_cols[:2]:
                chart_titles.append(f"{name} - {c}")
    except Exception:
        # if dataframes unavailable or malformed, skip
        pass

    # unique and limit
    chart_titles = [t for t in dict.fromkeys(chart_titles) if t]
    if not chart_titles:
        chart_titles = ["Dashboard"]

    # ===== Assign Form (at bottom) =====
    st.markdown("---")
    st.subheader("📤 Assign action item")

    with st.form("assign_dashboard_form", clear_on_submit=True):   # ✅ clears after submit
        to = st.text_input("Assign to (email or name)", key="assign_to_dashboard")
        due = st.date_input("Due Date", min_value=date.today(), key="due_dashboard")
        chart_choice = st.selectbox("Select Chart/Gauge", chart_titles, key="chart_choice_assign")
        priority = st.selectbox("Priority", ["Low", "Medium", "High"], index=1, key="priority_assign_dashboard")
        msg = st.text_area("Message", "Hi, Check this data and take necessary action.", key="msg_dashboard")
        submitted = st.form_submit_button("✅ Confirm Assign")

        if submitted:
            action = {
                "id": str(uuid.uuid4())[:6],
                "to": to,
                "due": str(due),
                "priority": priority,
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
    filter_priority = f1.selectbox("Priority", ["All", "Low", "Medium", "High"], key="filter_priority_actions")
    filter_status = f2.selectbox("Status", ["All", "Assigned", "Work in Progress", "Done"], key="filter_status_actions")
    filter_id = f3.text_input("Search by ID (partial allowed)", key="filter_id_actions")

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

        # Use stable keys based on action id to avoid shifting keys when list changes
        for act in filtered_actions:
            aid = act.get("id")
            cols = st.columns([1, 2, 2, 2, 2, 3, 1])

            # ID
            cols[0].write(aid[:6])

            # To / Due Date
            cols[1].write(act["to"])
            cols[2].write(act["due"])

            # Priority selector (stable key)
            priority_key = f"priority_{aid}"
            status_key = f"status_{aid}"
            delete_key = f"delete_{aid}"

            new_priority = cols[3].selectbox(
                "", ["Low", "Medium", "High"],
                index=["Low", "Medium", "High"].index(act.get("priority", "Medium")),
                key=priority_key,
                label_visibility="collapsed"
            )

            # Status selector (stable key)
            new_status = cols[4].selectbox(
                "", ["Assigned", "Work in Progress", "Done"],
                index=["Assigned", "Work in Progress", "Done"].index(act.get("status", "Assigned")),
                key=status_key,
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

            # Delete (stable key)
            if cols[6].button("🗑", key=delete_key):
                # find and remove the exact action by id from the master session list
                for sact in list(st.session_state.actions):
                    if sact.get("id") == aid:
                        st.session_state.actions.remove(sact)
                        break
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
            emails = st.text_input("Send to (comma separated emails)", key="alert_emails")
            message = st.text_area("Custom Message", f"Alert Triggered: {st.session_state.test_alert['query']}", key="alert_custom_message")
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





