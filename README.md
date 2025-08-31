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

🛠️ Installation
1. Clone repo
git clone https://github.com/Harsh8793/Chat_with_excel_or_csv.git

2. Create virtual environment
python -m venv venv
source venv/bin/activate   # Mac/Linux
venv\Scripts\activate      # Windows

3. Install dependencies
pip install -r requirements.txt

4. Environment variables

Create a .env file:

GROQ_API_KEY=your_groq_api_key_here

🚀 Usage
Step 1: Build embeddings (only once)
python build_embeddings.py


This will load your Excel/CSV files, build cached DataFrames, and save dataframes.pkl.

Step 2: Run chatbot
streamlit run chatbot.py

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
