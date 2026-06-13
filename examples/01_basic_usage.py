"""
Basic Usage Example for TSPulse Classifier.

This script demonstrates how to use the TSPulseClassifier
for time series classification.
"""

import numpy as np
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Fix #27: TPulseClassifier → TSPulseClassifier; correct module path
from sktime_ext.classification.deep_learning.tspulse import TSPulseClassifier


def generate_synthetic_data(n_samples=100, n_timesteps=50, n_classes=3):
    """Generate synthetic time series data for demonstration."""
    X = np.random.randn(n_samples, 1, n_timesteps)
    y = np.random.randint(0, n_classes, n_samples).astype(str)  # string labels like sktime datasets

    for i in range(n_samples):
        if y[i] == '0':
            X[i, 0, :] += np.linspace(0, 2, n_timesteps)
        elif y[i] == '1':
            X[i, 0, :] += np.sin(np.linspace(0, 4 * np.pi, n_timesteps))
        else:
            X[i, 0, :] += np.linspace(2, 0, n_timesteps)

    return X, y


def main():
    """Main function demonstrating TSPulse usage."""
    print("=" * 70)
    print("TSPulse Classifier - Basic Usage Example")
    print("=" * 70)

    # Generate synthetic data
    print("\n1. Generating synthetic data...")
    X_train, y_train = generate_synthetic_data(n_samples=200, n_timesteps=50, n_classes=3)
    X_test, y_test = generate_synthetic_data(n_samples=50, n_timesteps=50, n_classes=3)

    print(f"   Train shape: {X_train.shape}")
    print(f"   Test shape:  {X_test.shape}")
    print(f"   Classes:     {np.unique(y_train)}")

    # Example 1: Default / minimal configuration
    print("\n2. Training with minimal configuration...")
    clf = TSPulseClassifier(
        n_epochs=10,       # reduced for quick demo
        batch_size=32,
        random_state=42,
        verbose=True,
    )

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = np.mean(y_pred == y_test)

    print(f"\n   Test Accuracy: {accuracy:.4f}")

    y_proba = clf.predict_proba(X_test)
    print(f"   Prediction probabilities shape: {y_proba.shape}")

    # Example 2: Frozen backbone, auto LR
    print("\n3. Training with frozen backbone and auto learning rate...")
    clf_frozen = TSPulseClassifier(
        n_epochs=10,
        batch_size=32,
        freeze_backbone=True,
        learning_rate=None,   # triggers optimal_lr_finder
        random_state=42,
        verbose=True,
    )

    clf_frozen.fit(X_train, y_train)
    y_pred_frozen = clf_frozen.predict(X_test)
    accuracy_frozen = np.mean(y_pred_frozen == y_test)

    print(f"\n   Test Accuracy (frozen backbone): {accuracy_frozen:.4f}")

    # Example 3: Custom learning rate + early stopping
    print("\n4. Training with custom learning rate and early stopping...")
    clf_custom = TSPulseClassifier(
        n_epochs=15,
        batch_size=16,
        learning_rate=5e-5,
        freeze_backbone=True,
        early_stopping=True,
        early_stopping_patience=5,
        random_state=42,
        verbose=True,
    )

    clf_custom.fit(X_train, y_train)
    y_pred_custom = clf_custom.predict(X_test)
    accuracy_custom = np.mean(y_pred_custom == y_test)

    print(f"\n   Test Accuracy (custom): {accuracy_custom:.4f}")

    # Summary
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"Minimal config accuracy:         {accuracy:.4f}")
    print(f"Frozen backbone accuracy:        {accuracy_frozen:.4f}")
    print(f"Custom config accuracy:          {accuracy_custom:.4f}")
    print("=" * 70)

    # Training history — only keys that were actually logged
    if hasattr(clf, 'training_history_') and clf.training_history_:
        print("\n5. Training History (last 5 entries):")
        for record in clf.training_history_[-5:]:
            parts = [f"Epoch {record.get('epoch', '?')}"]
            if 'train_loss' in record:
                parts.append(f"Train Loss = {record['train_loss']:.4f}")
            if 'val_loss' in record:
                parts.append(f"Val Loss = {record['val_loss']:.4f}")
            print("   " + " | ".join(parts))


if __name__ == "__main__":
    main()
