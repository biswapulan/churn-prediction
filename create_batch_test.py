# create_batch.py
# Run this with: python create_batch.py

import pandas as pd
from pathlib import Path

# No streamlit imports - pure python script
BASE_DIR = Path(r'C:\Users\bisu2\Desktop\churn-prediction')

# Load original data
df = pd.read_csv(
    BASE_DIR / 'data' / 'raw' / 'WA_Fn-UseC_-Telco-Customer-Churn.csv'
)

# Drop columns not needed
df = df.drop(columns=['customerID', 'Churn'])

# Fix TotalCharges
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'] = df['TotalCharges'].fillna(0)

# Take 100 random customers
batch_test = df.sample(100, random_state=42)

# Save
output_path = BASE_DIR / 'data' / 'processed' / 'batch_test.csv'
batch_test.to_csv(output_path, index=False)

# Verify
if output_path.exists():
    print(f"✅ File created successfully!")
    print(f"📁 Location: {output_path}")
    print(f"👥 Customers: {len(batch_test)}")
    print(f"📊 Columns: {list(batch_test.columns)}")
else:
    print("❌ File creation failed!")