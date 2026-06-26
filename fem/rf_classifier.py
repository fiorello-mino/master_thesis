from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

CSV_PATH = Path("grid_data.csv")

df = pd.read_csv(CSV_PATH)

df = df[df["status"] == "ok"].copy()

df["n_bubbles"] = df["n_bubbles"].astype(int)

feature_cols = ["n_pores_effective", "k_spacing", "width", "height"]
X = df[feature_cols].to_numpy()
y = df["n_bubbles"].to_numpy()

rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    class_weight="balanced",
    random_state=42,
    n_jobs=4,
)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

y_pred = cross_val_predict(rf, X, y, cv=cv)

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

detail = pd.DataFrame({
    "true":    y,
    "pred":    y_pred,
    "correct": y == y_pred,
    **{col: df[col].values for col in feature_cols},
})

# Casi sbagliati ordinati per errore assoluto
errors = detail[~detail["correct"]].copy()
errors["abs_err"] = (errors["true"] - errors["pred"]).abs()
errors = errors.sort_values("abs_err", ascending=False)

print(f"\nErrori totali: {len(errors)} / {len(detail)}")

print("\nErrori per classe reale:")
print(detail.groupby("true")["correct"].agg(
    total="count",
    correct="sum",
    wrong=lambda x: (~x).sum(),
    accuracy=lambda x: x.mean()
))

labels = sorted(np.unique(y))
cm_df = pd.DataFrame(cm, index=labels, columns=labels)

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues", ax=ax)
ax.set_xlabel("Predicted n_bubbles")
ax.set_ylabel("True n_bubbles")
ax.set_title("Confusion Matrix — RF")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
plt.show()