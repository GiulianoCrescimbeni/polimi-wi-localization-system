import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re

def heatmap_rssi_for_sa(cartella_csv, target_sa):
    grid = np.full((6, 4), np.nan)
    counts = np.zeros((6, 4), dtype=int)

    for file in os.listdir(cartella_csv):
        if file.endswith('.csv'):
            match = re.match(r"\((\d),(\d)\)\.csv", file)
            if match:
                x, y = int(match.group(1)), int(match.group(2))
                path = os.path.join(cartella_csv, file)
                try:
                    df = pd.read_csv(path)
                    df['wlan.sa'] = df['wlan.sa'].str.strip().str.lower()
                    df = df[df['wlan.sa'] == target_sa.lower()]
                    rssi_values = df['radiotap.dbm_antsignal'].dropna().astype(float)
                    if not rssi_values.empty:
                        grid[x, y] = rssi_values.mean()
                        counts[x, y] = len(rssi_values)
                except Exception as e:
                    print(f"Errore in {file}: {e}")

    total_samples = int(np.sum(counts))
    print(f"{target_sa} → {total_samples} RSSI samples used")

    plt.imshow(grid.T, origin='lower', cmap='viridis', interpolation='nearest')
    plt.title(f"Mean RSSI for SA: {target_sa}")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.colorbar(label='RSSI (dBm)')
    plt.xticks(range(6))
    plt.yticks(range(4))

    for x in range(6):
        for y in range(4):
            if counts[x, y] > 0:
                plt.text(x, y, f"{counts[x, y]}", ha='center', va='center', fontsize=8, color='white')

    plt.grid(False)
    plt.show()

dataFolder = "BalancedData"

heatmap_rssi_for_sa(dataFolder, "f2:42:89:7e:73:bd")
heatmap_rssi_for_sa(dataFolder, "40:ee:dd:0e:38:54")
heatmap_rssi_for_sa(dataFolder, "3a:07:16:cb:a7:f4")
heatmap_rssi_for_sa(dataFolder, "c4:ea:1d:53:75:c3")