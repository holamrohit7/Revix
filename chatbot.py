import os
import pickle
import pandas as pd
import streamlit as st
import plotly.express as px
import streamlit.components.v1 as components
from dotenv import load_dotenv
from groq import Groq
import uuid
from datetime import date

# ------------------------
# Setup
# ------------------------
st.set_page_config(page_title="Revix Chatbot", page_icon="🤖", layout="wide")
st.title("🤖 Agent Revix")

# ------------------------
# Sidebar Options
# ------------------------
st.sidebar.header("⚙ Options")
show_charts = st.sidebar.checkbox("📊 Show Charts Automatically", value=True)

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