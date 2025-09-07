import pickle
import os
import pandas as pd

DATA_FOLDER = os.path.join(os.path.dirname(__file__), '..', 'excel')
DF_FILE = os.path.join(DATA_FOLDER, 'dataframes.pkl')

if not os.path.exists(DF_FILE):
    print('dataframes.pkl not found at', DF_FILE)
    raise SystemExit(1)

with open(DF_FILE, 'rb') as f:
    dataframes = pickle.load(f)

candidates = []
for name, df in dataframes.items():
    print('\n===', name, '===')
    print('shape:', df.shape)
    print('columns:', df.columns.tolist())
    # find uptime-like columns
    for c in df.columns:
        if any(k in c.lower() for k in ['uptime', 'availability', 'service_level', 'availability_pct', 'avail']):
            print('\n-- Candidate uptime column:', c)
            s = df[c].dropna().astype(str).str.strip()
            print('sample values:', s.head(10).tolist())
            # try parse
            s_nopct = s.str.rstrip('%')
            numeric = pd.to_numeric(s_nopct, errors='coerce').dropna()
            print('parsed numeric sample:', numeric.head(10).tolist())
            if not numeric.empty:
                med = numeric.median()
                mean = numeric.mean()
                if med <= 1.0:
                    print('median <=1.0, treating as fraction; mean percent ~', mean * 100.0)
                else:
                    print('mean percent ~', mean)
            candidates.append((name, c))

print('\nFound candidate uptime columns:', candidates)
