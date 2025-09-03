import os
import pandas as pd
import numpy as np
from collections import Counter
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# ---- Data Loading ----
def load_dataset(folder, sa_list):
    X, y = [], []
    label_map = {}
    current_label = 0

    for file in sorted(os.listdir(folder)):
        if file.endswith('.csv') and "(" in file:
            try:
                coord = eval(file.replace(".csv", ""))
                if coord not in label_map:
                    label_map[coord] = current_label
                    current_label += 1

                df = pd.read_csv(os.path.join(folder, file))
                df['wlan.sa'] = df['wlan.sa'].str.lower().str.strip()

                grouped = df[df['wlan.sa'].isin(sa_list)].groupby('wlan.sa')

                rows_per_file = {}
                for sa in sa_list:
                    rows_per_file[sa] = df[df['wlan.sa'] == sa]['radiotap.dbm_antsignal'].astype(float).values

                min_len = min(len(rows_per_file[sa]) for sa in sa_list)
                if min_len == 0:
                    continue

                for i in range(min_len):
                    row = [rows_per_file[sa][i] for sa in sa_list]
                    X.append(row)
                    y.append(label_map[coord])

            except Exception as e:
                print(f"Error in {file}: {e}")

    return np.array(X), np.array(y), label_map

# ---- Parameters ----
target_sas = [
    "f2:42:89:7e:73:bd",
    "40:ee:dd:0e:38:54",
    "3a:07:16:cb:a7:f4",
    "c4:ea:1d:53:75:c3"
]

# ---- Load and Split ----
X, y, label_map = load_dataset("BalancedData", target_sas)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# --- Label Map ---
print("\nLabel Map:")
for i, pos in label_map.items():
    print(f"{i}: {pos}")

# ---- Train Random Forest ----
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

# ---- Reverse label map for coordinate display ----
reverse_label_map = {v: k for k, v in label_map.items()}

# Compute average localization error
true_coords = np.array([reverse_label_map[i] for i in y_test])
pred_coords = np.array([reverse_label_map[i] for i in y_pred])
distances = np.linalg.norm(true_coords - pred_coords, axis=1)
avg_error = np.mean(distances)
print("Random Forest Accuracy:", accuracy_score(y_test, y_pred))
print(f"Average Localization Error: {avg_error:.2f} metri")

# ---- Confusion Matrix with coordinate labels ----
fig, ax = plt.subplots(figsize=(8, 8))
ConfusionMatrixDisplay.from_predictions(
    y_test, y_pred,
    display_labels=[reverse_label_map[i] for i in range(len(reverse_label_map))],
    xticks_rotation=90,
    ax=ax,
    normalize='true'
)
plt.title("Confusion Matrix - Random Forest")
plt.tight_layout()
plt.show()