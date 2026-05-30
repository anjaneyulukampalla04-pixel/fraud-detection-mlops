import os
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve
)

from imblearn.over_sampling import SMOTE

# -----------------------------
# CONFIG
# -----------------------------
RANDOM_STATE = 42

OUTPUT_DIR = "outputs"
PLOTS_DIR = "plots"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

# -----------------------------
# LOAD DATASET
# -----------------------------
df = pd.read_csv("archive/archive/fraudTrain.csv")

print("Dataset Shape:", df.shape)

# -----------------------------
# TARGET COLUMN
# -----------------------------
TARGET_COL = "is_fraud"

# -----------------------------
# PREPROCESSING
# -----------------------------
X = df.drop(columns=[TARGET_COL])
y = df[TARGET_COL].astype(int)

# Remove non-numeric columns
non_numeric = X.select_dtypes(include=['object']).columns

if len(non_numeric) > 0:
    print("Dropping non-numeric columns:")
    print(non_numeric.tolist())

    X = X.drop(columns=non_numeric)

# -----------------------------
# FEATURE COLUMNS
# -----------------------------
feature_columns = X.columns.tolist()

# Save feature columns
joblib.dump(feature_columns, "outputs/feature_columns.joblib")

# -----------------------------
# SCALING
# -----------------------------
scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

# Save scaler
joblib.dump(scaler, "outputs/scaler.joblib")

# -----------------------------
# TRAIN TEST SPLIT
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.2,
    random_state=RANDOM_STATE,
    stratify=y
)

# -----------------------------
# HANDLE IMBALANCE
# -----------------------------
print("Applying SMOTE...")

smote = SMOTE(random_state=RANDOM_STATE)

X_train_resampled, y_train_resampled = smote.fit_resample(
    X_train,
    y_train
)

# -----------------------------
# MODELS
# -----------------------------
models = {
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "DecisionTree": DecisionTreeClassifier(random_state=RANDOM_STATE),
    "RandomForest": RandomForestClassifier(
        n_estimators=100,
        random_state=RANDOM_STATE
    )
}

results = {}

# -----------------------------
# TRAINING
# -----------------------------
for name, model in models.items():

    with mlflow.start_run(run_name=name):

        print(f"\nTraining {name}...")

        model.fit(X_train_resampled, y_train_resampled)

        y_pred = model.predict(X_test)

        y_proba = model.predict_proba(X_test)[:, 1]

        auc = roc_auc_score(y_test, y_proba)

        mlflow.log_param("model_name", name)
        mlflow.log_metric("roc_auc", auc)

        mlflow.sklearn.log_model(
            model,
            artifact_path=name
        )

    print(classification_report(y_test, y_pred))

    print("ROC-AUC:", auc)

    # Save model
    model_path = f"outputs/{name}.joblib"

    joblib.dump(model, model_path)

    print(f"Saved model: {model_path}")

    # Store results
    results[name] = {
        "auc": auc,
        "y_proba": y_proba
    }

# -----------------------------
# ROC CURVE
# -----------------------------
plt.figure(figsize=(8, 6))

for name, result in results.items():

    fpr, tpr, _ = roc_curve(y_test, result["y_proba"])

    plt.plot(
        fpr,
        tpr,
        label=f"{name} (AUC={result['auc']:.3f})"
    )

plt.plot([0, 1], [0, 1], linestyle='--')

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

plt.title("ROC Curves")

plt.legend()

roc_path = "plots/roc_curves.png"

plt.savefig(roc_path)
mlflow.log_artifact(roc_path)

print(f"ROC curve saved: {roc_path}")

# -----------------------------
# FINISHED
# -----------------------------
print("\nTraining Complete")
print("All models saved in outputs/")

