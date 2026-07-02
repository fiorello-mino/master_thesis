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

# valutazione con CV
y_pred_cv = cross_val_predict(rf, X, y, cv=cv)

acc = accuracy_score(y, y_pred_cv)
bacc = balanced_accuracy_score(y, y_pred_cv)
macro_f1 = f1_score(y, y_pred_cv, average="macro")
cm = confusion_matrix(y, y_pred_cv)

print("=== Random Forest baseline ===")
print(f"Accuracy          : {acc:.4f}")
print(f"Balanced accuracy : {bacc:.4f}")
print(f"Macro F1          : {macro_f1:.4f}")

print("\nDistribuzione classi reali:")
print(df["n_bubbles"].value_counts().sort_index())

print("\nClassification report:")
print(classification_report(y, y_pred_cv))

print("\nConfusion matrix:")
print(cm)

# Confusion matrix plot come prima
labels = sorted(np.unique(y))
cm_df = pd.DataFrame(cm, index=labels, columns=labels)

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues", ax=ax)
ax.set_xlabel("Predicted n_bubbles")
ax.set_ylabel("True n_bubbles")
ax.set_title("Confusion Matrix — RF (CV)")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
#plt.show()

# modello finale addestrato su tutto X,y
rf.fit(X, y)
print("Modello RF addestrato su tutto il dataset di training.")



TEST_CSV = Path("test_init_summary.csv")

df_test = pd.read_csv(TEST_CSV)

feature_cols = ["n_pores_effective", "k_spacing", "width", "height"]
X_test = df_test[feature_cols].to_numpy()

y_pred_test = rf.predict(X_test)

df_test["n_bubbles_pred"] = y_pred_test

out_test_csv = Path("test_init_with_predictions.csv")
df_test.to_csv(out_test_csv, index=False)

print(f"Predizioni completate su {len(df_test)} righe.")
print(f"CSV con n_bubbles_pred salvato in: {out_test_csv}")

print("\nDistribuzione delle classi predette sul test:")
print(df_test["n_bubbles_pred"].value_counts().sort_index())
