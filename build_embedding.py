# build_embeddings.py
import os
import glob
import pickle
import re
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer

DATA_FOLDER = "D:\Dev_space\Dev1\Revix\excel"
CACHE_FILE = os.path.join(DATA_FOLDER, "faiss_index.pkl")
DF_FILE = os.path.join(DATA_FOLDER, "dataframes.pkl")

def normalize_name(filename):
    """Normalize file names to consistent short keys"""
    name = os.path.splitext(filename)[0].lower()
    name = re.sub(r"[^a-z0-9]", "_", name)  # replace spaces, symbols
    if "salesforce" in name: return "salesforce"
    if "revenue" in name: return "revenue"
    if "gainsight" in name: return "gainsight"
    if "usage" in name: return "usage"
    if "jira" in name: return "jira"
    return name

def load_files(folder):
    excel_files = glob.glob(os.path.join(folder, "*.xlsx"))
    csv_files = glob.glob(os.path.join(folder, "*.csv"))
    all_files = excel_files + csv_files

    dataframes = {}
    for file in all_files:
        try:
            if file.endswith(".xlsx"):
                df = pd.read_excel(file, engine="openpyxl")
            else:
                df = pd.read_csv(file)
            norm_name = normalize_name(os.path.basename(file))
            dataframes[norm_name] = df
            print(f"✅ Loaded {file} as {norm_name} ({df.shape[0]} rows)")
        except Exception as e:
            print(f"⚠️ Could not read {file}: {e}")
    return dataframes

print("📂 Loading files...")
dataframes = load_files(DATA_FOLDER)
if not dataframes:
    print("❌ No files found.")
    exit()

print("⚡ Building embeddings...")
model = SentenceTransformer("all-MiniLM-L6-v2")
docs, metadata = [], []

# Chunk rows (50 at a time for speed)
for fname, df in dataframes.items():
    chunk_size = 50
    for start in range(0, len(df), chunk_size):
        chunk = df.iloc[start:start+chunk_size]
        row_text = "\n".join(
            [" | ".join([f"{col}: {str(row[col])}" for col in df.columns]) for _, row in chunk.iterrows()]
        )
        docs.append(row_text[:2000])  # truncate if too long
        metadata.append({"file": fname, "rows": f"{start}-{start+len(chunk)-1}"})

if docs:
    embeddings = model.encode(docs, convert_to_tensor=False, show_progress_bar=True)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    with open(CACHE_FILE, "wb") as f:
        pickle.dump((index, docs, metadata, model), f)
    print(f"💾 Saved embeddings → {CACHE_FILE}")
else:
    print("⚠️ No docs created, skipping embeddings.")

# Always save DataFrames
with open(DF_FILE, "wb") as f:
    pickle.dump(dataframes, f)
print(f"💾 Saved raw dataframes → {DF_FILE}")
