"""
Phase 1 baseline: binary network intrusion detection on CIC-IDS2017.

Pipeline:
  1. Load and concatenate all MachineLearningCSV files.
  2. Clean the data (column names, inf/NaN, duplicates).
  3. Build a binary target: BENIGN (0) vs attack (1).
  4. Train/test split with stratification.
  5. Scale features and train a Random Forest baseline.
  6. Evaluate with the metrics that actually matter for an IDS.

Usage:
  Put the CIC-IDS2017 CSV files in ./data/ , then run:
      python train_baseline.py
"""

import glob
import os

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

DATA_DIR = "data"
RANDOM_STATE = 42


def load_data(data_dir: str) -> pd.DataFrame:
    """Load and concatenate every CSV in the data directory."""
    csv_paths = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
    if not csv_paths:
        raise FileNotFoundError(
            f"No CSV files found in '{data_dir}'. "
            "Put the CIC-IDS2017 MachineLearningCSV files there."
        )

    frames = []
    for path in csv_paths:
        print(f"  loading {os.path.basename(path)} ...")
        # low_memory=False avoids mixed-type warnings on these files.
        frames.append(pd.read_csv(path, low_memory=False))

    df = pd.concat(frames, ignore_index=True)
    print(f"  combined shape: {df.shape}")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Fix the well-known quirks of the CIC-IDS2017 CSV files."""
    # 1. Column names have leading/trailing spaces in this dataset.
    df.columns = df.columns.str.strip()

    # 2. Some flow features contain inf (e.g. division by zero in rates).
    df = df.replace([np.inf, -np.inf], np.nan)

    # 3. Drop rows with missing values (a small fraction of the data).
    before = len(df)
    df = df.dropna()
    print(f"  dropped {before - len(df)} rows with NaN/inf")

    # 4. Remove exact duplicate flows.
    before = len(df)
    df = df.drop_duplicates()
    print(f"  dropped {before - len(df)} duplicate rows")

    return df


def build_features_and_target(df: pd.DataFrame):
    """Separate features from the label and build a binary target."""
    # The label column is named 'Label' after stripping whitespace.
    label_col = "Label"

    # Binary target: anything that is not BENIGN counts as an attack.
    y = (df[label_col].str.upper() != "BENIGN").astype(int)

    # Drop the label, keep only numeric feature columns.
    X = df.drop(columns=[label_col])
    X = X.select_dtypes(include=[np.number])

    # A few columns are constant (all zeros); they carry no signal.
    constant_cols = X.columns[X.nunique() <= 1]
    if len(constant_cols) > 0:
        X = X.drop(columns=constant_cols)
        print(f"  dropped {len(constant_cols)} constant columns")

    print(f"  feature matrix: {X.shape}")
    print(f"  class balance -> benign: {(y == 0).sum()}, attack: {(y == 1).sum()}")
    return X, y


def main():
    print("[1/5] Loading data ...")
    df = load_data(DATA_DIR)

    print("[2/5] Cleaning data ...")
    df = clean_data(df)

    print("[3/5] Building features and target ...")
    X, y = build_features_and_target(df)

    print("[4/5] Splitting and scaling ...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,  # keep the same benign/attack ratio in both splits
    )

    # Random Forest does not strictly need scaling, but we scale here so the
    # exact same preprocessing carries over to the neural network later.
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    print("[5/5] Training Random Forest baseline ...")
    clf = RandomForestClassifier(
        n_estimators=100,
        n_jobs=-1,            # use all CPU cores
        class_weight="balanced",  # counter the benign-heavy imbalance
        random_state=RANDOM_STATE,
    )
    clf.fit(X_train, y_train)

    print("\n=== Results ===")
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]

    print("\nConfusion matrix (rows = actual, cols = predicted):")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification report:")
    print(classification_report(y_test, y_pred, target_names=["benign", "attack"]))

    print(f"ROC AUC: {roc_auc_score(y_test, y_prob):.4f}")

    # Feature importance: which columns drive the predictions?
    importances = pd.Series(clf.feature_importances_, index=X.columns)
    print("\nTop 15 features by importance:")
    print(importances.sort_values(ascending=False).head(15))


if __name__ == "__main__":
    main()
