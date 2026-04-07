"""
Test TSPulse sktime integration on local multivariate .ts datasets
from data/Multivariate_ts/

Usage:
    python test_local_datasets.py                        # all datasets
    python test_local_datasets.py BasicMotions Epilepsy  # specific datasets
"""

import sys
import os
import time
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score, classification_report

warnings.filterwarnings("ignore")

from sktime_ext.classification.deep_learning.tspulse import TSPulseClassifier

DATA_ROOT = Path("data/Multivariate_ts")

DEFAULT_DATASETS = sorted(p.name for p in DATA_ROOT.iterdir() if p.is_dir())


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def load_ts_dataset(name: str):
    """Load a dataset from data/Multivariate_ts/<name>/<name>_TRAIN.ts"""
    from sktime.datasets import load_from_tsfile

    train_path = DATA_ROOT / name / f"{name}_TRAIN.ts"
    test_path  = DATA_ROOT / name / f"{name}_TEST.ts"

    if not train_path.exists():
        raise FileNotFoundError(f"Train file not found: {train_path}")
    if not test_path.exists():
        raise FileNotFoundError(f"Test file not found: {test_path}")

    X_train, y_train = load_from_tsfile(str(train_path), return_data_type="numpy3d")
    X_test,  y_test  = load_from_tsfile(str(test_path),  return_data_type="numpy3d")

    return X_train, y_train, X_test, y_test


def dataset_info(name, X_train, y_train, X_test, y_test):
    n_inst, n_dims, series_len = X_train.shape
    classes = np.unique(y_train)
    print(f"\n  Dataset      : {name}")
    print(f"  Train/Test   : {len(X_train)} / {len(X_test)}")
    print(f"  Dimensions   : {n_dims}")
    print(f"  Series length: {series_len}")
    print(f"  Classes ({len(classes)}): {classes}")


def run_dataset(name: str, n_epochs: int = 50, batch_size: int = 32,
                freeze_backbone: bool = False):
    """Train & evaluate TSPulse on one dataset. Returns result dict."""
    print(f"\n{'=' * 70}")
    print(f"  DATASET: {name}")
    print(f"{'=' * 70}")

    # ── Load ──────────────────────────────────────────────────────────────────
    try:
        X_train, y_train, X_test, y_test = load_ts_dataset(name)
    except Exception as e:
        print(f"  [SKIP] Could not load {name}: {e}")
        return {"dataset": name, "status": "load_error", "error": str(e)}

    dataset_info(name, X_train, y_train, X_test, y_test)

    # ── Build classifier ──────────────────────────────────────────────────────
    clf = TSPulseClassifier(
        n_epochs=n_epochs,
        batch_size=batch_size,
        freeze_backbone=freeze_backbone,
        early_stopping=False,
        random_state=42,
        verbose=True,
        tsfm_path="../granite-tsfm",
    )

    # ── Train ─────────────────────────────────────────────────────────────────
    print(f"\n  Training ({n_epochs} epochs, freeze_backbone={freeze_backbone})...")
    t0 = time.time()
    try:
        clf.fit(X_train, y_train)
    except Exception as e:
        print(f"  [ERROR] Training failed: {e}")
        return {"dataset": name, "status": "train_error", "error": str(e)}
    train_time = time.time() - t0
    print(f"  Training time: {train_time:.1f}s")

    # ── Predict ───────────────────────────────────────────────────────────────
    try:
        y_pred = clf.predict(X_test)
        y_proba = clf.predict_proba(X_test)
    except Exception as e:
        print(f"  [ERROR] Prediction failed: {e}")
        return {"dataset": name, "status": "predict_error", "error": str(e)}

    # ── Metrics ───────────────────────────────────────────────────────────────
    acc      = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
    n_classes = len(np.unique(y_train))

    print(f"\n  Results:")
    print(f"    Accuracy : {acc:.4f}  ({acc * 100:.2f}%)")
    print(f"    Macro F1 : {macro_f1:.4f}  ({macro_f1 * 100:.2f}%)")
    print(f"\n  Classification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    return {
        "dataset":    name,
        "status":     "ok",
        "n_train":    len(X_train),
        "n_test":     len(X_test),
        "n_dims":     X_train.shape[1],
        "series_len": X_train.shape[2],
        "n_classes":  n_classes,
        "accuracy":   round(acc, 4),
        "macro_f1":   round(macro_f1, 4),
        "train_time_s": round(train_time, 1),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    # Datasets to run: CLI args override DEFAULT_DATASETS
    if len(sys.argv) > 1:
        datasets = sys.argv[1:]
    else:
        datasets = DEFAULT_DATASETS

    # Validate all datasets exist before starting
    missing = [d for d in datasets if not (DATA_ROOT / d).exists()]
    if missing:
        print(f"[ERROR] These datasets were not found in {DATA_ROOT}: {missing}")
        available = sorted(p.name for p in DATA_ROOT.iterdir() if p.is_dir())
        print(f"Available datasets: {available}")
        sys.exit(1)

    print("=" * 70)
    print("  TSPulse sktime Integration — Local Multivariate Dataset Test")
    print("=" * 70)
    print(f"  Datasets : {datasets}")
    print(f"  Data root: {DATA_ROOT.resolve()}")

    out_csv = "results_local_datasets.csv"
    results = []
    total_t0 = time.time()

    for name in datasets:
        result = run_dataset(name)
        results.append(result)

        # ── Write / append after every dataset so partial runs are not lost ──
        df_so_far = pd.DataFrame(results)
        df_so_far.to_csv(out_csv, index=False)
        print(f"  [CSV] Saved {len(results)}/{len(datasets)} results → {out_csv}")

    total_time = time.time() - total_t0

    # ── Summary table ─────────────────────────────────────────────────────────
    print(f"\n\n{'=' * 70}")
    print("  SUMMARY")
    print(f"{'=' * 70}")
    header = f"{'Dataset':<28} {'Status':<12} {'Acc':>7} {'F1':>7} {'Time(s)':>8}"
    print(header)
    print("-" * 70)

    for r in results:
        if r["status"] == "ok":
            row = (f"{r['dataset']:<28} {'OK':<12} "
                   f"{r['accuracy']:>7.4f} {r['macro_f1']:>7.4f} "
                   f"{r['train_time_s']:>8.1f}")
        else:
            row = f"{r['dataset']:<28} {r['status']:<12}"
        print(row)

    ok_results = [r for r in results if r["status"] == "ok"]
    if ok_results:
        avg_acc = np.mean([r["accuracy"] for r in ok_results])
        avg_f1  = np.mean([r["macro_f1"] for r in ok_results])
        print("-" * 70)
        print(f"{'Average':<28} {'':<12} {avg_acc:>7.4f} {avg_f1:>7.4f}")

    print(f"\n  Total wall time: {total_time:.1f}s")
    print(f"  Results saved to: {out_csv}")
    print("=" * 70)


if __name__ == "__main__":
    main()