import pandas as pd
import os
from pathlib import Path

# Define file paths
paths = [
    "/Users/similovesyou/Desktop/qts/simian-behavior/5-csrt/derivatives/raw-modeling-data-elo-min-attempts.csv",
    "/Users/similovesyou/Desktop/qts/simian-behavior/dms/derivatives/raw-modeling-data-elo-min-attempts.csv",
    "/Users/similovesyou/Desktop/qts/simian-behavior/TAVo-tevas/derivatives/raw-modeling-data-elo-min-attempts.csv"
]

# Dictionary to store loaded dataframes
dfs = {}

# Load each file with enhanced error handling
for path in paths:
    try:
        # Extract a short name from the path
        name = Path(path).parents[2].name  # Gets '5-csrt', 'dms', or 'TAVo-tevas'
        
        # Check if file exists and is not empty
        if not os.path.exists(path):
            print(f"File not found: {path}")
            continue
            
        if os.path.getsize(path) == 0:
            print(f"Empty file: {path}")
            continue
            
        # Try reading with different encodings if needed
        try:
            df = pd.read_csv(path)
        except UnicodeDecodeError:
            df = pd.read_csv(path, encoding='latin1')
            
        dfs[name] = df
        print(f"\nSuccessfully loaded {name} data")
        print(f"File size: {os.path.getsize(path)} bytes")
        
    except Exception as e:
        print(f"\nError loading {path}: {str(e)}")

# If all DataFrames are empty, check the files manually
if all(df.empty for df in dfs.values()):
    print("\nAll loaded DataFrames are empty. Possible issues:")
    print("1. The files might actually be empty")
    print("2. The files might contain only headers")
    print("3. There might be a parsing issue")
    print("\nPlease verify the file contents manually.")
else:
    # Print column information for non-empty dataframes
    print("\nColumn Information:")
    for name, df in dfs.items():
        if not df.empty:
            print(f"\n{name} dataset:")
            print(f"Shape: {df.shape}")
            print("Columns:")
            print(df.columns.tolist())
            print("\nFirst 3 rows:")
            print(df.head(3))
            print("\nMissing values per column:")
            print(df.isnull().sum())
        else:
            print(f"\n{name} dataset is empty")

# Additional diagnostics
print("\nRunning additional diagnostics:")
for path in paths:
    try:
        print(f"\nFile: {path}")
        print(f"Exists: {os.path.exists(path)}")
        if os.path.exists(path):
            print(f"Size: {os.path.getsize(path)} bytes")
            with open(path, 'r') as f:
                first_lines = [next(f) for _ in range(3)]
            print("First 3 lines:")
            for line in first_lines:
                print(line.strip())
    except Exception as e:
        print(f"Error examining {path}: {str(e)}")