import os
import pandas as pd
import random

def balance_csv_dataset(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    csv_files = [f for f in os.listdir(input_folder) if f.endswith('.csv') and "(" in f]

    lengths = {}
    for file in csv_files:
        df = pd.read_csv(os.path.join(input_folder, file))
        lengths[file] = len(df)

    if not lengths:
        print("No CSV files found.")
        return

    min_len = min(lengths.values())
    print(f"Minimum number of entries across files: {min_len}")

    for file in csv_files:
        path = os.path.join(input_folder, file)
        df = pd.read_csv(path)
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # Shuffle
        df_balanced = df.iloc[:min_len]
        df_balanced.to_csv(os.path.join(output_folder, file), index=False)
        print(f"{file}: trimmed to {min_len} entries")

balance_csv_dataset("ProcessedData/", "BalancedData/")