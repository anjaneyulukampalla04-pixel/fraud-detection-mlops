"""
Credit Card Fraud Detection
Single-file script for loading the uploaded dataset, exploratory data analysis,
preprocessing, training three models (Logistic Regression, Decision Tree, Random Forest),
handling class imbalance (SMOTE), evaluation (confusion matrix, classification report, ROC-AUC),
and saving trained models.

This script is written to work with the uploaded zip at: /mnt/data/archive.zip
It will automatically extract CSV files from the archive and attempt to load the most
likely fraud dataset. If you already have the CSV path, set CSV_PATH.

Usage:
    python credit_card_fraud_detection.py

Requirements:
    pandas, numpy, scikit-learn, imbalanced-learn, matplotlib, joblib
    Install with: pip install pandas numpy scikit-learn imbalanced-learn matplotlib joblib

"""

import os
import zipfile
import glob
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, precision_recall_fscore_support
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import joblib

# ---------- CONFIG ----------
ZIP_PATH = "C:\\Users\\rajua\\OneDrive\Desktop\\project . ML\\archive.zip"    # path to uploaded archive (provided in conversation)
CSV_PATH = None                       # set to a path if you know the CSV (overrides ZIP_PATH)
RANDOM_STATE = 42
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------- HELPERS ----------

def extract_csv_from_zip(zip_path, out_dir="data"):
    """Extracts zip and returns a list of CSV file paths found inside."""
    csv_files = []
    if not os.path.exists(zip_path):
        print(f"Zip not found at {zip_path}")
        return csv_files
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(out_dir)
        for name in z.namelist():
            if name.lower().endswith('.csv'):
                csv_files.append(os.path.join(out_dir, name))
    return csv_files


def find_best_csv(csv_files):
    """Heuristic: prefer files named 'credit', 'fraud', or containing many rows/columns."""
    if not csv_files:
        return None
    # prioritize filenames
    for fname in csv_files:
        if any(key in os.path.basename(fname).lower() for key in ['credit', 'fraud', 'transaction']):
            return fname
    # otherwise take largest file by size
    csv_files_sorted = sorted(csv_files, key=lambda p: os.path.getsize(p), reverse=True)
    return csv_files_sorted[0]


# ---------- LOAD DATA ----------
if CSV_PATH and os.path.exists(CSV_PATH):
    data_path = CSV_PATH
else:
    print('Looking for CSV files inside the provided zip...')
    csvs = extract_csv_from_zip(ZIP_PATH, out_dir='/mnt/data')
    if not csvs:
        raise FileNotFoundError('No CSV files found in the provided archive. Place a CSV or update CSV_PATH.')
    data_path = find_best_csv(csvs)

print(f'Using data file: {data_path}')

df = pd.read_csv(data_path)
print('\nDataset shape:', df.shape)
print('\nColumns:', df.columns.tolist())

# Quick peek
print('\nFirst 5 rows:')
print(df.head())

# ---------- SIMPLE EDA ----------
print('\nChecking for missing values:')
print(df.isnull().sum().sort_values(ascending=False).head(20))

# Find target column — common names: 'Class', 'isFraud', 'fraud', 'target'
possible_targets = ['Class', 'isFraud', 'fraud', 'is_fraud', 'target', 'label']
cols_lower = [c.lower() for c in df.columns]
TARGET_COL = None
for t in possible_targets:
    if t.lower() in cols_lower:
        TARGET_COL = df.columns[cols_lower.index(t.lower())]
        break

if TARGET_COL is None:
    # ask heuristic: if there is a column with only 0/1
    for c in df.columns:
        unique_vals = set(df[c].dropna().unique())
        if unique_vals <= {0, 1}:
            TARGET_COL = c
            break

if TARGET_COL is None:
    raise ValueError('Could not find a binary target column automatically. Please set TARGET_COL manually.')

print(f'Using target column: {TARGET_COL}')

# Show class distribution
print('\nClass distribution:')
print(df[TARGET_COL].value_counts())

# ---------- PREPROCESS ----------
# Separate features and target
X = df.drop(columns=[TARGET_COL])
y = df[TARGET_COL].astype(int)

# Drop non-numeric columns if any (simple approach) — user can refine
non_numeric = X.select_dtypes(include=['object', 'category']).columns.tolist()
if non_numeric:
    print(f"Dropping non-numeric columns: {non_numeric} (you can preprocess them instead)")
    X = X.drop(columns=non_numeric)

# If there is an 'Time' column and 'Amount', typical creditcard data has these — we scale Amount
if 'Amount' in X.columns:
    # scale Amount separately
    scaler_amount = StandardScaler()
    X['Amount_scaled'] = scaler_amount.fit_transform(X[['Amount']])
    X = X.drop(columns=['Amount'])

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)
print('\nTrain positive ratio:', y_train.mean(), 'Test positive ratio:', y_test.mean())

# Handle class imbalance using SMOTE on training set
print('\nPerforming SMOTE on training set...')
sm = SMOTE(random_state=RANDOM_STATE)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)
print('After resampling, class counts:', np.bincount(y_train_res))

# ---------- MODELS ----------
models = {
    'LogisticRegression': LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
    'DecisionTree': DecisionTreeClassifier(random_state=RANDOM_STATE),
    'RandomForest': RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1)
}

results = {}

for name, model in models.items():
    print(f'\nTraining {name}...')
    model.fit(X_train_res, y_train_res)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else model.decision_function(X_test)

    report = classification_report(y_test, y_pred, digits=4)
    auc = roc_auc_score(y_test, y_proba)
    cm = confusion_matrix(y_test, y_pred)
    results[name] = {
        'model': model,
        'report': report,
        'auc': auc,
        'cm': cm,
        'y_pred': y_pred,
        'y_proba': y_proba
    }
    print(report)
    print(f'ROC-AUC: {auc:.4f}')
    print('Confusion matrix:\n', cm)

    # save model
    model_path = os.path.join(OUTPUT_DIR, f'{name}.joblib')
    joblib.dump(model, model_path)
    print(f'Model saved to: {model_path}')

# Save scalers too
joblib.dump(scaler, os.path.join(OUTPUT_DIR, 'scaler.joblib'))
if 'scaler_amount' in globals():
    joblib.dump(scaler_amount, os.path.join(OUTPUT_DIR, 'scaler_amount.joblib'))
print('\nSaved scalers to output folder.')

# ---------- PLOT ROC CURVES ----------
plt.figure(figsize=(8, 6))
for name, res in results.items():
    fpr, tpr, _ = roc_curve(y_test, res['y_proba'])
    plt.plot(fpr, tpr, label=f'{name} (AUC = {res["auc"]:.3f})')
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves')
plt.legend()
roc_path = os.path.join(OUTPUT_DIR, 'roc_curves.png')
plt.savefig(roc_path, bbox_inches='tight')
print(f'ROC plot saved to: {roc_path}')

# ---------- SUMMARY ----------
print('\nSummary of model AUCs:')
for name, res in results.items():
    print(f'{name}: AUC = {res["auc"]:.4f}')

print(f'All outputs (models, scalers, plots) are in: {OUTPUT_DIR}')

# End of script
