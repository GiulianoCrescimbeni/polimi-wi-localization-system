import os
import numpy as np
import pandas as pd
from collections import defaultdict, Counter
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# ---- Load dataset ----
def load_full_dataset(folder, sa_list, block_size=5):
    sa_list = [sa.lower() for sa in sa_list]
    X, y = [], []
    label_map = {}
    label_counter = 0

    for file in sorted(os.listdir(folder)):
        if file.endswith('.csv') and '(' in file:
            try:
                coord = eval(file.split(".")[0])
                if coord not in label_map:
                    label_map[coord] = label_counter
                    label_counter += 1

                df = pd.read_csv(os.path.join(folder, file))
                df['wlan.sa'] = df['wlan.sa'].astype(str).str.strip().str.lower()

                for start in range(0, len(df), block_size):
                    block = df.iloc[start:start + block_size]
                    if len(block) < block_size:
                        continue

                    features = []
                    for sa in sa_list:
                        rssi_values = block[block['wlan.sa'] == sa]['radiotap.dbm_antsignal'].astype(float)
                        features.append(rssi_values.mean() if not rssi_values.empty else -100.0)

                    if not any(pd.isna(features)):
                        X.append(features)
                        y.append(label_map[coord])
            except Exception as e:
                print(f"Error reading {file}: {e}")
    return np.array(X), np.array(y), label_map

# ---- Parameters ----
target_sas = [
    "f2:42:89:7e:73:bd",
    "40:ee:dd:0e:38:54",
    "3a:07:16:cb:a7:f4",
    "c4:ea:1d:53:75:c3"
]

X, y, label_map = load_full_dataset("BalancedData", target_sas, block_size=5)
print(f"Total samples: {len(X)}")
print(f"Feature shape: {X.shape}")
print(f"Label map: {label_map}")

# ---- Stratified split ----
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
for train_index, test_index in sss.split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

# ---- Horus (conditional probability) ----
def predict_horus(X_train, y_train, X_test):
    y_pred = []
    label_classes = np.unique(y_train)

    # Model construction: per-class distributions
    distributions = defaultdict(list)
    for features, label in zip(X_train, y_train):
        distributions[label].append(features)
    
    for test_sample in X_test:
        probs = {}
        for label in label_classes:
            samples = np.array(distributions[label])
            # Inverse Euclidean distance as probability proxy
            dists = np.linalg.norm(samples - test_sample, axis=1)
            eps = 1e-6  # to avoid division by zero
            weights = 1 / (dists + eps)
            probs[label] = np.sum(weights)
        predicted_label = max(probs, key=probs.get)
        y_pred.append(predicted_label)
    return y_pred

# ---- Prediction ----
y_pred = predict_horus(X_train, y_train, X_test)

# ---- Accuracy ----
accuracy = accuracy_score(y_test, y_pred)
print("Horus Accuracy:", accuracy)

# ---- Reverse label map to display coordinates ----
reverse_label_map = {v: k for k, v in label_map.items()}

# ---- Compute average localization error ----
true_coords = np.array([reverse_label_map[i] for i in y_test])
pred_coords = np.array([reverse_label_map[i] for i in y_pred])
distances = np.linalg.norm(true_coords - pred_coords, axis=1)
avg_error = np.mean(distances)
print(f"Average Localization Error: {avg_error:.2f}")

# ---- Confusion Matrix with coordinate labels ----
fig, ax = plt.subplots(figsize=(8, 8))
ConfusionMatrixDisplay.from_predictions(
    y_test, y_pred,
    display_labels=[reverse_label_map[i] for i in range(len(reverse_label_map))],
    xticks_rotation=90,
    ax=ax,
    normalize='true'
)
ax.set_title("Confusion Matrix - Horus")
plt.tight_layout()
plt.show()