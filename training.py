# fall_detection_randomforest.py

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib

# 1️⃣ Dataset Path
BASE_PATH = "dataset/"  # change this to your dataset directory

# 2️⃣ Define Fall and Non-Fall Classes
FALL_CLASSES = ['freefall', 'runfall', 'walkfall']
NON_FALL_CLASSES = ['downsit', 'runsit', 'walksit']

# 3️⃣ Load and Label Data
all_data = []

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().lower().replace(" ", "_").replace("-", "_") for c in df.columns]
    return df

def pick_acc_columns(df: pd.DataFrame):
    cols = set(df.columns)
    # Prefer explicit iOS-style names first
    if {"accelerationx", "accelerationy", "accelerationz"}.issubset(cols):
        return "accelerationx", "accelerationy", "accelerationz"
    # Common variants
    for triplet in [("acc_x", "acc_y", "acc_z"), ("ax", "ay", "az"), ("x", "y", "z")]:
        if set(triplet).issubset(cols):
            return triplet
    return None

for folder in os.listdir(BASE_PATH):
    folder_path = os.path.join(BASE_PATH, folder)
    if os.path.isdir(folder_path):
        folder_key = folder.strip().lower()
        label = 1 if folder_key in FALL_CLASSES else 0 if folder_key in NON_FALL_CLASSES else None
        if label is None:
            continue
        for file in os.listdir(folder_path):
            if file.lower().endswith(".csv"):
                file_path = os.path.join(folder_path, file)
                try:
                    # Auto-detect delimiter (handles ';' and ',')
                    df = pd.read_csv(file_path, sep=None, engine='python')
                except Exception:
                    continue
                df = normalize_columns(df)
                cols = pick_acc_columns(df)
                if cols is None:
                    continue
                x_col, y_col, z_col = cols
                sub = df[[x_col, y_col, z_col]].copy()
                sub.rename(columns={x_col: 'acc_x', y_col: 'acc_y', z_col: 'acc_z'}, inplace=True)
                sub['label'] = label
                all_data.append(sub)

# Combine all data
if len(all_data) == 0:
    print("No usable data files found. Ensure folders and CSV columns are correct.")
    raise SystemExit(1)
else:
    data = pd.concat(all_data, ignore_index=True)


# 4️⃣ Feature Engineering
# Compute resultant acceleration and simple absolute-value features
data['resultant'] = np.sqrt(data['acc_x']**2 + data['acc_y']**2 + data['acc_z']**2)
data['abs_x'] = np.abs(data['acc_x'])
data['abs_y'] = np.abs(data['acc_y'])
data['abs_z'] = np.abs(data['acc_z'])

X = data[['acc_x', 'acc_y', 'acc_z', 'resultant', 'abs_x', 'abs_y', 'abs_z']]
y = data['label']

# 5️⃣ Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

"""Random Forest does not require feature scaling; keep it simple."""

# 6️⃣ Random Forest Training (simple, but stronger fixed params)
rf = RandomForestClassifier(
    n_estimators=400,
    max_depth=None,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features='sqrt',
    bootstrap=True,
    class_weight='balanced_subsample',
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train, y_train)

# 7️⃣ Model Evaluation
y_pred = rf.predict(X_test)

print("\n Model Evaluation Results:")
print("Accuracy:", round(accuracy_score(y_test, y_pred) * 100, 2), "%")
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# 8️⃣ Save Model
os.makedirs("models", exist_ok=True)
joblib.dump(rf, os.path.join("models", "fall_detection_rf_model.pkl"))

print("\n Model saved to models/fall_detection_rf_model.pkl")