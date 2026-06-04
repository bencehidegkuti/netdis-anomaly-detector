"""
Phase 1, multi-class analysis: per-attack-type detection on CIC-IDS2017.

Where the binary baseline answered "attack or not", this script answers
"which kind of attack", and more importantly shows which attack types the
model detects well and which it struggles with.

Pipeline:
  1. Load and concatenate all CSV files (same as the baseline).
  2. Clean the data, including the broken Web Attack label encoding.
  3. Keep the original attack labels as a multi-class target.
  4. Stratified train/test split, scale features.
  5. Train a Random Forest and evaluate PER CLASS.
  6. Save a row-normalized confusion matrix heatmap for the README.

Usage:
  python train_multiclass.py
"""

import glob
import os
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
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
        frames.append(pd.read_csv(path, low_memory=False))
    df = pd.concat(frames, ignore_index=True)
    print(f"  combined shape: {df.shape}")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Fix the well-known quirks of the CIC-IDS2017 CSV files."""
    # Column names have leading/trailing spaces in this dataset.
    df.columns = df.columns.str.strip()

    # Replace inf (division-by-zero in rate features) with NaN, then drop.
    df = df.replace([np.inf, -np.inf], np.nan)
    before = len(df)
    df = df.dropna()
    print(f"  dropped {before - len(df)} rows with NaN/inf")

    before = len(df)
    df = df.drop_duplicates()
    print(f"  dropped {before - len(df)} duplicate rows")
    return df


def normalize_labels(series: pd.Series) -> pd.Series:
    """
    Clean the label text.

    The Web Attack labels in CIC-IDS2017 contain a non-UTF-8 byte (a cp1252
    dash) that shows up as garbage. We replace any non-ASCII run with a plain
    dash and collapse whitespace, so labels like 'Web Attack <byte> Brute Force'
    become a clean 'Web Attack - Brute Force'.
    """
    def fix(s: str) -> str:
        s = re.sub(r"[^\x00-\x7F]+", "-", s)   # non-ASCII -> dash
        s = re.sub(r"\s+", " ", s).strip()       # collapse whitespace
        return s

    return series.astype(str).apply(fix)


def build_features_and_target(df: pd.DataFrame):
    """Separate numeric features from the multi-class label."""
    label_col = "Label"
    y = normalize_labels(df[label_col])

    X = df.drop(columns=[label_col])
    X = X.select_dtypes(include=[np.number])

    constant_cols = X.columns[X.nunique() <= 1]
    if len(constant_cols) > 0:
        X = X.drop(columns=constant_cols)
        print(f"  dropped {len(constant_cols)} constant columns")

    print(f"  feature matrix: {X.shape}")
    print("\n  class distribution:")
    counts = y.value_counts()
    for label, n in counts.items():
        print(f"    {label:<32} {n:>8}")
    return X, y


def main():
    print("[1/5] Loading data ...")
    df = load_data(DATA_DIR)

    print("[2/5] Cleaning data ...")
    df = clean_data(df)

    print("[3/5] Building features and multi-class target ...")
    X, y = build_features_and_target(df)

    print("\n[4/5] Splitting and scaling ...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,  # keep every attack type proportionally in both splits
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    print("[5/5] Training Random Forest (multi-class) ...")
    clf = RandomForestClassifier(
        n_estimators=100,
        n_jobs=-1,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )
    clf.fit(X_train, y_train)

    print("\n=== Per-class results ===\n")
    y_pred = clf.predict(X_test)

    # The per-class report is the heart of this analysis: precision/recall/F1
    # for each attack type separately. Low recall on a class means the model
    # misses that attack; low precision means it raises false alarms for it.
    print(classification_report(y_test, y_pred, zero_division=0))

    # Row-normalized confusion matrix: each row sums to 1, so the diagonal is
    # the per-class detection rate (recall). Saved as an image for the README.
    labels_sorted = sorted(y.unique())
    cm = confusion_matrix(y_test, y_pred, labels=labels_sorted, normalize="true")

    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm,
        annot=True, fmt=".2f", cmap="Blues",
        xticklabels=labels_sorted, yticklabels=labels_sorted,
        cbar_kws={"label": "fraction of true class"},
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("CIC-IDS2017 multi-class confusion matrix (row-normalized)")
    plt.tight_layout()
    out_path = "confusion_matrix_multiclass.png"
    plt.savefig(out_path, dpi=150)
    print(f"\nSaved confusion matrix heatmap to: {out_path}")


if __name__ == "__main__":
    main()
