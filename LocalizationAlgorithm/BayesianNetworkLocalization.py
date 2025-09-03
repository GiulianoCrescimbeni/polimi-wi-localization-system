import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination

# ----Load dataset ----
def load_data(folder, sas, max_entries=None):
    X, y = [], []
    label_map = {}
    label_idx = 0
    for file in sorted(os.listdir(folder)):
        if file.endswith('.csv') and '(' in file:
            label = eval(file.replace('.csv', ''))
            if label not in label_map:
                label_map[label] = label_idx
                label_idx += 1
            df = pd.read_csv(os.path.join(folder, file))
            df['wlan.sa'] = df['wlan.sa'].astype(str).str.lower().str.strip()
            row_per_sa = []
            for sa in sas:
                rssi_values = df[df['wlan.sa'] == sa]['radiotap.dbm_antsignal'].astype(float).values
                if max_entries:
                    np.random.shuffle(rssi_values)
                    rssi_values = rssi_values[:max_entries]
                if len(rssi_values) == 0:
                    rssi_values = [-100] * (max_entries or 1)
                row_per_sa.append(rssi_values)
            rows = list(zip(*row_per_sa))
            X.extend(rows)
            y.extend([label_map[label]] * len(rows))
    return np.array(X), np.array(y), label_map

# ---- Parameters ----
target_sas = [
    "f2:42:89:7e:73:bd",
    "40:ee:dd:0e:38:54",
    "3a:07:16:cb:a7:f4",
    "c4:ea:1d:53:75:c3"
]

# ---- Load data ----
X, y, label_map = load_data("BalancedData", target_sas, max_entries=50)
print("Samples:", len(X))
print("Feature shape:", X.shape)
print("Label map:", label_map)

# ---- Preprocessing: categorized dataframe ----
df = pd.DataFrame(X, columns=[f"rssi_{i}" for i in range(X.shape[1])])
df['location'] = y

# ---- Discretize RSSI (simplification) ----
for col in df.columns[:-1]:
    df[col] = pd.qcut(df[col], q=4, labels=False, duplicates='drop')

# ---- Split ----
train_df, test_df = train_test_split(df, test_size=0.3, random_state=42)

# ---- Create Bayesian model ----
features = [f"rssi_{i}" for i in range(X.shape[1])]
edges = [(f, 'location') for f in features]
model = BayesianNetwork(edges)
model.fit(train_df, estimator=MaximumLikelihoodEstimator)

# ---- Inference ----
infer = VariableElimination(model)

# ---- Prediction ----
y_pred = []
for _, row in test_df.iterrows():
    evidence = {f: row[f] for f in features}
    q = infer.map_query(variables=['location'], evidence=evidence)
    y_pred.append(q['location'])

# ---- Accuracy ----
acc = accuracy_score(test_df['location'], y_pred)
print(f"Bayesian Network Accuracy: {acc:.4f}")

# ---- Reverse label map ----
reverse_label_map = {v: k for k, v in label_map.items()}

# ---- Average localization error ----
real_coords = np.array([reverse_label_map[i] for i in test_df['location']])
pred_coords = np.array([reverse_label_map[i] for i in y_pred])

distances = np.linalg.norm(real_coords - pred_coords, axis=1)
mean_error = np.mean(distances)
print(f"Average Localization Error: {mean_error:.2f}")

# ---- Confusion Matrix with readable labels ----
fig, ax = plt.subplots(figsize=(10, 10))
ConfusionMatrixDisplay.from_predictions(
    test_df['location'], y_pred,
    display_labels=[reverse_label_map[i] for i in range(len(reverse_label_map))],
    xticks_rotation=90,
    ax=ax,
    normalize='true'
)
plt.title("Confusion Matrix - Bayesian Network")
plt.tight_layout()
plt.show()