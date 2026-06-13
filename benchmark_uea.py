"""
TSPulse UEA Benchmark
=====================
Benchmarks TSPulseClassifier on all 29 UEA multivariate datasets.

Usage
-----
    python benchmark_uea.py

Results are saved to tspulse_uea_results.csv
"""

import os
import time
import warnings
import traceback

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Import
# ---------------------------------------------------------------------------
from sktime.datasets import load_UCR_UEA_dataset
from sktime_ext.classification.deep_learning.tspulse import TSPulseClassifier

# ---------------------------------------------------------------------------
# All 29 UEA multivariate datasets
# ---------------------------------------------------------------------------
DATASETS = [
    "ArticularyWordRecognition",
    "AtrialFibrillation",
    "BasicMotions",
    "CharacterTrajectories",
    "Cricket",
    "DuckDuckGeese",
    "EigenWorms",
    "Epilepsy",
    "EthanolConcentration",
    "ERing",
    "FaceDetection",
    "FingerMovements",
    "HandMovementDirection",
    "Handwriting",
    "Heartbeat",
    "InsectWingbeat",
    "JapaneseVowels",
    "Libras",
    "LSST",
    "MotorImagery",
    "NATOPS",
    "PedestrianActivity",
    "PEMS-SF",
    "PhonemeSpectra",
    "RacketSports",
    "SelfRegulationSCP1",
    "SelfRegulationSCP2",
    "SpokenArabicDigits",
    "StandWalkJump",
    "UWaveGestureLibrary",
]

# ---------------------------------------------------------------------------
# Helper: DataFrame-of-Series → numpy3D (n_instances, n_channels, series_len)
# ---------------------------------------------------------------------------
def to_numpy3d(X_df: pd.DataFrame) -> np.ndarray:
    """
    Convert sktime nested DataFrame to numpy3D array.

    Handles both fixed-length and ragged series (pads ragged to max length).
    """
    n_instances = len(X_df)
    n_channels  = X_df.shape[1]

    # Collect all series as arrays
    series = [
        [np.asarray(X_df.iloc[i, c], dtype=np.float32) for c in range(n_channels)]
        for i in range(n_instances)
    ]

    # Determine max length (handles ragged series)
    max_len = max(s.shape[0] for row in series for s in row)

    # Build array, zero-padding if necessary
    out = np.zeros((n_instances, n_channels, max_len), dtype=np.float32)
    for i, row in enumerate(series):
        for c, s in enumerate(row):
            out[i, c, : len(s)] = s

    return out


# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
N_EPOCHS    = 20
BATCH_SIZE  = 16
RESULTS     = []

print("=" * 70)
print("TSPulse UEA Benchmark — 29 Multivariate Datasets")
print(f"n_epochs={N_EPOCHS}  batch_size={BATCH_SIZE}")
print("=" * 70)

# ---------------------------------------------------------------------------
# MAIN LOOP
# ---------------------------------------------------------------------------
for idx, dataset in enumerate(DATASETS, 1):
    print(f"\n[{idx:02d}/{len(DATASETS)}] {dataset}")
    print("-" * 50)

    try:
        # ---- Load --------------------------------------------------------
        t0 = time.time()
        X_train_df, y_train = load_UCR_UEA_dataset(
            name=dataset, split="train", return_X_y=True
        )
        X_test_df, y_test = load_UCR_UEA_dataset(
            name=dataset, split="test", return_X_y=True
        )
        load_time = time.time() - t0
        print(f"  Loaded in {load_time:.1f}s  |  "
              f"train={len(y_train)}  test={len(y_test)}  "
              f"channels={X_train_df.shape[1]}  "
              f"classes={len(np.unique(y_train))}")

        # ---- Convert to numpy3D ------------------------------------------
        X_train = to_numpy3d(X_train_df)
        X_test  = to_numpy3d(X_test_df)

        series_len = X_train.shape[2]
        if series_len > 512:
            print(f"  ⚠ Series length {series_len} > 512 — will be truncated by model")

        # ---- Train -------------------------------------------------------
        # Pass original string labels directly — BaseClassifier encodes them
        clf = TSPulseClassifier(
            n_epochs=N_EPOCHS,
            batch_size=BATCH_SIZE,
            freeze_backbone=False,
            random_state=42,
            verbose=False,
        )

        t0 = time.time()
        clf.fit(X_train, y_train)
        train_time = time.time() - t0

        # ---- Predict -----------------------------------------------------
        t0 = time.time()
        y_pred = clf.predict(X_test)
        pred_time = time.time() - t0

        # ---- Metrics  (compare in original label space) ------------------
        acc = accuracy_score(y_test, y_pred)
        f1  = f1_score(y_test, y_pred, average="macro", zero_division=0)

        print(f"  Accuracy : {acc:.4f}")
        print(f"  Macro F1 : {f1:.4f}")
        print(f"  Train time: {train_time:.1f}s  |  Predict time: {pred_time:.2f}s")

        RESULTS.append({
            "dataset":      dataset,
            "n_train":      len(y_train),
            "n_test":       len(y_test),
            "n_channels":   X_train.shape[1],
            "series_length": series_len,
            "n_classes":    len(np.unique(y_train)),
            "accuracy":     round(acc, 4),
            "macro_f1":     round(f1, 4),
            "train_time_s": round(train_time, 1),
            "predict_time_s": round(pred_time, 2),
            "status":       "ok",
        })

    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        traceback.print_exc()
        RESULTS.append({
            "dataset":      dataset,
            "n_train":      None,
            "n_test":       None,
            "n_channels":   None,
            "series_length": None,
            "n_classes":    None,
            "accuracy":     None,
            "macro_f1":     None,
            "train_time_s": None,
            "predict_time_s": None,
            "status":       f"FAILED: {e}",
        })
        continue

# ---------------------------------------------------------------------------
# SAVE & PRINT
# ---------------------------------------------------------------------------
df = pd.DataFrame(RESULTS)

out_path = "tspulse_uea_results.csv"
df.to_csv(out_path, index=False)

print("\n\n" + "=" * 70)
print("FINAL RESULTS")
print("=" * 70)

# Pretty print — only show key columns
display_cols = ["dataset", "n_train", "n_test", "n_channels",
                "n_classes", "accuracy", "macro_f1", "train_time_s", "status"]
print(df[display_cols].to_string(index=False))

# Summary stats (successful runs only)
ok = df[df["status"] == "ok"]
failed = df[df["status"] != "ok"]

print(f"\n{'=' * 70}")
print(f"Completed:  {len(ok)}/{len(DATASETS)} datasets")
print(f"Failed:     {len(failed)}/{len(DATASETS)} datasets")
if len(ok) > 0:
    print(f"Mean Accuracy : {ok['accuracy'].mean():.4f}")
    print(f"Mean Macro F1 : {ok['macro_f1'].mean():.4f}")
    print(f"Best  Accuracy: {ok['accuracy'].max():.4f}  ({ok.loc[ok['accuracy'].idxmax(), 'dataset']})")
    print(f"Worst Accuracy: {ok['accuracy'].min():.4f}  ({ok.loc[ok['accuracy'].idxmin(), 'dataset']})")
if len(failed) > 0:
    print(f"\nFailed datasets:")
    for _, row in failed.iterrows():
        print(f"  {row['dataset']}: {row['status']}")

print(f"\nResults saved to: {out_path}")
print("=" * 70)
