from pathlib import Path
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

CSV_PATH = Path("/home/fiorello/master_thesis/fem/grid_data.csv")

# 1. carica dati
df = pd.read_csv(CSV_PATH)

# 2. tieni solo righe valide
df = df[df["status"] == "ok"].copy()

# 3. target intero
df["n_bubbles"] = df["n_bubbles"].astype(int)

# 4. feature e target
feature_cols = ["n_pores_effective", "k_spacing", "width", "height"]
X = df[feature_cols].to_numpy()
y = df["n_bubbles"].to_numpy()

# 5. modello
rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    class_weight="balanced",
    random_state=42,
    n_jobs=4,
)

# 6. cross-validation stratificata
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# 7. predizioni out-of-fold
y_pred = cross_val_predict(rf, X, y, cv=cv)

# 8. metriche
acc = accuracy_score(y, y_pred)
bacc = balanced_accuracy_score(y, y_pred)
macro_f1 = f1_score(y, y_pred, average="macro")
cm = confusion_matrix(y, y_pred)

print("=== Random Forest baseline ===")
print(f"Accuracy          : {acc:.4f}")
print(f"Balanced accuracy : {bacc:.4f}")
print(f"Macro F1          : {macro_f1:.4f}")

print("\nDistribuzione classi reali:")
print(df["n_bubbles"].value_counts().sort_index())

print("\nClassification report:")
print(classification_report(y, y_pred))

print("\nConfusion matrix:")
print(cm)