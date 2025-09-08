📊 Revix Chatbot

Revix is a Streamlit-based intelligent chatbot that lets you chat with your Excel/CSV data.
You can ask natural questions, and the bot will generate Pandas code, execute it, and show results as beautiful tables and charts.

⚡ Features

🤖 Chat with your data

Ask natural language questions.

Auto-converts queries → Pandas code → Results.

Smart query memory (last_result) for follow-ups.

📑 Clean Output

Auto-tables with centered styling (no index).

Auto-charts with Plotly.

Rounds numbers to 2 decimals.

🔒 Data-Only Answers

Works only with your Excel/CSV files.

No outside-world knowledge.



🚀 Usage
Step 1: Build embeddings (only once)
python build_embeddings.py


This will load your Excel/CSV files, build cached DataFrames, and save dataframes.pkl.

Step 2: Run chatbot
streamlit run app2.py

Step 3: Ask questions

Examples:

"Top 5 customers by Annual Revenue"

"Product wise CSAT score"

"Show Alpha customers revenue"

"Summarize Salesforce data"

"Show me tickets count by product"

📷 Example

Query: Product wise CSAT score

Output:

Product	Count	Sum	Avg
FNA	6716	20234	3.01
FNB	6610	19834	3.00
FNC	6674	19820	2.97

(Plus a chart if enabled ✅)

🧩 Requirements

Python 3.9+

Streamlit

Pandas

Plotly

Groq API client

python-dotenv
