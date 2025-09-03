import os
import pandas as pd
from collections import Counter, defaultdict
import numpy as np

def find_top_sas_with_threshold(csv_folder, threshold_ratio=0.7):
    file_paths = [os.path.join(csv_folder, f) for f in os.listdir(csv_folder) if f.endswith('.csv')]
    
    sa_counter = Counter()
    file_presence = defaultdict(int)
    rssi_per_sa = defaultdict(list)
    total_files = len(file_paths)

    for path in file_paths:
        try:
            df = pd.read_csv(path)
            df['wlan.sa'] = df['wlan.sa'].dropna().str.strip().str.lower()
            df = df.dropna(subset=['wlan.sa', 'radiotap.dbm_antsignal'])

            sa_set = set(df['wlan.sa'])
            for sa in sa_set:
                file_presence[sa] += 1

            sa_counter.update(df['wlan.sa'])

            for _, row in df.iterrows():
                sa = row['wlan.sa']
                rssi = row['radiotap.dbm_antsignal']
                try:
                    rssi_per_sa[sa].append(float(rssi))
                except:
                    continue

        except Exception as e:
            print(f"Error in file {path}: {e}")

    min_required = int(threshold_ratio * total_files)
    valid_sas = {sa for sa, count in file_presence.items() if count >= min_required}

    top5 = Counter({sa: count for sa, count in sa_counter.items() if sa in valid_sas}).most_common(5)

    print(f"\nTop 5 Source Addresses (present in at least {threshold_ratio*100:.0f}% of files):")
    for sa, count in top5:
        files_present = file_presence[sa]
        rssi_values = rssi_per_sa[sa]
        if len(rssi_values) > 1:
            variance = np.var(rssi_values)
            print(f"{sa} - {count} total entries, present in {files_present}/{total_files} files, RSSI variance: {variance:.2f} dBm²")
        else:
            print(f"{sa} - {count} total entries, present in {files_present}/{total_files} files, RSSI variance: insufficient data")

find_top_sas_with_threshold("SampleData/", threshold_ratio=0.7)
