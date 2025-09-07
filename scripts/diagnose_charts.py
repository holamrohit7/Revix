import pickle
import os
import pandas as pd

DATA_FOLDER = os.path.join(os.path.dirname(__file__), '..', 'excel')
DF_FILE = os.path.join(DATA_FOLDER, 'dataframes.pkl')

if not os.path.exists(DF_FILE):
    print('dataframes.pkl not found:', DF_FILE)
    raise SystemExit(1)

with open(DF_FILE, 'rb') as f:
    dfs = pickle.load(f)

candidates = ['revenue', 'amount', 'sales', 'price', 'csat', 'health', 'uptime', 'availability', 'status', 'resolved', 'date', 'month']

for name, df in dfs.items():
    print('\n---', name, '---')
    print('shape:', df.shape)
    print('columns:', list(df.columns))
    print('dtypes:')
    print(df.dtypes)
    print('\nSample rows:')
    print(df.head(5))

    # detect candidate columns
    found = {k: [] for k in candidates}
    for col in df.columns:
        low = col.lower()
        for k in candidates:
            if k in low:
                found[k].append(col)
    for k, cols in found.items():
        if cols:
            print(f"Found {k} columns: {cols}")

    # quick check for numeric revenue-like
    numcols = df.select_dtypes(include=['number']).columns.tolist()
    if numcols:
        print('Numeric columns:', numcols[:5])

print('\nDiagnosis complete')
