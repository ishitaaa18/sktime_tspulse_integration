"""
Train TSPulse on Real Dataset with Early Stopping
"""

import sys
import os
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, 'sktime/classification/deep_learning')
sys.path.insert(0, '../granite-tsfm')

print("=" * 80)
print("TSPulse Real Dataset Training - 100 Epochs with Early Stopping")
print("=" * 80)

from sktime_ext.classification.deep_learning.tspulse import TSPulseClassifier


# -------------------------------------------------------------------------
# [1/7] LOAD DATASET
# -------------------------------------------------------------------------
print("\n[1/7] Loading real dataset...")

dataset_loaded = False

try:
    from sktime.datasets import load_italy_power_demand
    X_train, y_train = load_italy_power_demand(split="train", return_X_y=True)
    X_test, y_test = load_italy_power_demand(split="test", return_X_y=True)
    dataset_name = "ItalyPowerDemand"
    dataset_loaded = True
    print(f"   ✓ Loaded {dataset_name} dataset")
except Exception as e:
    print(f"   ! Could not load ItalyPowerDemand: {e}")
    sys.exit(1)


# -------------------------------------------------------------------------
# [2/7] DATASET INFO
# -------------------------------------------------------------------------
print(f"\n[2/7] Dataset Information:")
print(f"   Dataset:    {dataset_name}")
print(f"   Train:      {len(X_train)} samples")
print(f"   Test:       {len(X_test)} samples")
print(f"   Classes:    {np.unique(y_train)}")          # original labels, e.g. ['1' '2']
print(f"   Class dist: {dict(zip(*np.unique(y_train, return_counts=True)))}")

# ✅ NO manual encoding here.
# TSPulseClassifier._fit() calls LabelEncoder internally.
# predict() / predict_proba() return predictions in the ORIGINAL label space.


# -------------------------------------------------------------------------
# [3/7] CREATE CLASSIFIER
# -------------------------------------------------------------------------
print(f"\n[3/7] Creating TSPulse classifier...")

clf = TSPulseClassifier(
    n_epochs=100,
    batch_size=32,
    freeze_backbone=True,
    early_stopping=True,
    early_stopping_patience=10,
    random_state=42,
    verbose=True,
    tsfm_path='../granite-tsfm',
)

print("   ✓ Classifier created")


# -------------------------------------------------------------------------
# [4/7] TRAIN MODEL
# -------------------------------------------------------------------------
print(f"\n[4/7] Training model...")
print(f"   This may take 10-20 minutes depending on your hardware...\n")

# Pass original labels directly — '1' and '2' (or whatever the dataset uses)
clf.fit(X_train, y_train)

print(f"\n   ✓ Training complete!")

# Confirm what the classifier learned internally
print(f"   Internal label mapping: "
      f"{dict(enumerate(clf.label_encoder_.classes_))}")


# -------------------------------------------------------------------------
# [5/7] TRAINING HISTORY
# -------------------------------------------------------------------------
print(f"\n[5/7] Training History:")
print("=" * 80)

if hasattr(clf, 'trainer_') and clf.trainer_.state.log_history:
    train_losses, eval_losses = [], []

    for entry in clf.trainer_.state.log_history:
        if 'loss' in entry:
            train_losses.append(entry['loss'])
            print(f"Epoch {entry.get('epoch', '?'):6.1f}: Train Loss = {entry['loss']:.4f}")
        if 'eval_loss' in entry:
            eval_losses.append(entry['eval_loss'])
            print(f"Epoch {entry.get('epoch', '?'):6.1f}: Val   Loss = {entry['eval_loss']:.4f}")

    if eval_losses:
        print(f"\nBest Validation Loss: {min(eval_losses):.4f}")
else:
    print("   ! No training history available")


# -------------------------------------------------------------------------
# [6/7] PREDICTIONS
# -------------------------------------------------------------------------
print(f"\n[6/7] Making predictions...")

# y_pred values are in the ORIGINAL label space (e.g. '1', '2')
y_pred = clf.predict(X_test)
y_proba = clf.predict_proba(X_test)

print(f"   ✓ Predictions shape:    {y_pred.shape}")
print(f"   ✓ Probabilities shape:  {y_proba.shape}")

unique, counts = np.unique(y_pred, return_counts=True)
print(f"\n   Prediction Distribution:")
for cls, count in zip(unique, 1counts):
    print(f"      Class {cls}: {count}/{len(y_pred)} ({count / len(y_pred) * 100:.1f}%)")


# -------------------------------------------------------------------------
# [7/7] EVALUATION
# -------------------------------------------------------------------------
print(f"\n[7/7] Evaluation Results:")
print("=" * 80)

# Compare directly using original labels — no remapping needed
acc = accuracy_score(y_test, y_pred)
macro_f1 = f1_score(y_test, y_pred, average='macro')

print(f"\nOverall Metrics:")
print(f"  Accuracy: {acc:.4f} ({acc * 100:.2f}%)")
print(f"  Macro F1: {macro_f1:.4f} ({macro_f1 * 100:.2f}%)")

print(f"\nClassification Report:")
print("-" * 80)
print(classification_report(y_test, y_pred))

print(f"Confusion Matrix:")
print("-" * 80)
print(confusion_matrix(y_test, y_pred))

# Per-class accuracy using original labels directly
print(f"\nPer-Class Accuracy:")
print("-" * 80)
for cls in np.unique(y_test):
    cls_mask = y_test == cls
    cls_acc = accuracy_score(y_test[cls_mask], y_pred[cls_mask])
    print(f"   Class {cls}: {cls_acc:.4f} ({cls_acc * 100:.2f}%)")


# -------------------------------------------------------------------------
# MODEL INFO
# -------------------------------------------------------------------------
print(f"\n{'=' * 80}")
print("MODEL INFORMATION")
print("=" * 80)
print(f"Model class:  {type(clf.model_).__name__}")
print(f"Module:       {type(clf.model_).__module__}")
print(f"Label classes learned: {clf.label_encoder_.classes_}")

print("=" * 80)
print(f"\n✓ Training and evaluation complete!")
print(f"Dataset:        {dataset_name}")
print(f"Final Accuracy: {acc * 100:.2f}%")
print("=" * 80)
