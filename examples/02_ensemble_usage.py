"""
Ensemble Usage Example for Maximum Accuracy.

This script demonstrates how to combine multiple TSPulseClassifier instances
via sktime's VotingClassifier for improved accuracy through model averaging.

Note: TSPulseClassifier does not ship a custom TPulseEnsemble class.
      The standard sktime ensemble wrappers are used instead.
"""

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# Fix #27: TPulseEnsemble does not exist → use sktime's VotingClassifier
from sktime_ext.classification.deep_learning.tspulse import TSPulseClassifier
from sktime.classification.ensemble import WeightedEnsembleClassifier


def generate_synthetic_data(n_samples=100, n_timesteps=50, n_classes=3):
    """Generate synthetic time series data for demonstration."""
    X = np.random.randn(n_samples, 1, n_timesteps)
    y = np.random.randint(0, n_classes, n_samples).astype(str)

    for i in range(n_samples):
        if y[i] == '0':
            X[i, 0, :] += np.linspace(0, 2, n_timesteps)
        elif y[i] == '1':
            X[i, 0, :] += np.sin(np.linspace(0, 4 * np.pi, n_timesteps))
        else:
            X[i, 0, :] += np.linspace(2, 0, n_timesteps)

    return X, y


def main():
    """Main function demonstrating ensemble usage."""
    print("=" * 70)
    print("TSPulse Ensemble - Maximum Accuracy Example")
    print("=" * 70)

    # Generate data
    print("\n1. Generating synthetic data...")
    X_train, y_train = generate_synthetic_data(n_samples=200, n_timesteps=50, n_classes=3)
    X_test, y_test = generate_synthetic_data(n_samples=50, n_timesteps=50, n_classes=3)

    print(f"   Train shape: {X_train.shape}")
    print(f"   Test shape:  {X_test.shape}")

    # Build three TSPulseClassifier members with varied seeds / LR
    print("\n2. Building ensemble of 3 TSPulse models...")
    members = [
        ("tspulse_1", TSPulseClassifier(n_epochs=10, batch_size=32,
                                         learning_rate=1e-4, random_state=0,
                                         verbose=False)),
        ("tspulse_2", TSPulseClassifier(n_epochs=10, batch_size=32,
                                         learning_rate=5e-5, random_state=1,
                                         verbose=False)),
        ("tspulse_3", TSPulseClassifier(n_epochs=10, batch_size=32,
                                         freeze_backbone=False, random_state=2,
                                         verbose=False)),
    ]

    ensemble = WeightedEnsembleClassifier(
        classifiers=members,
        metric="accuracy",      # used for weight estimation
        random_state=42,
    )

    # Train ensemble
    print("\n3. Training ensemble (trains each member sequentially)...")
    ensemble.fit(X_train, y_train)

    # Predict
    print("\n4. Making predictions...")
    y_pred_ensemble = ensemble.predict(X_test)
    accuracy_ensemble = np.mean(y_pred_ensemble == y_test)
    print(f"\n   Ensemble Test Accuracy: {accuracy_ensemble:.4f}")

    # Individual model accuracies
    print("\n5. Analysing individual models...")
    for name, clf in members:
        y_pred_ind = clf.predict(X_test)
        acc = np.mean(y_pred_ind == y_test)
        print(f"   {name} accuracy: {acc:.4f}")

    # Summary
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"Ensemble Accuracy: {accuracy_ensemble:.4f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
