"""
Basic Usage Example for TPulse Classifier.

This script demonstrates how to use the TPulse classifier
for time series classification.
"""

import numpy as np
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sktime.classification.deep_learning import TPulseClassifier


def generate_synthetic_data(n_samples=100, n_timesteps=50, n_classes=3):
    """Generate synthetic time series data for demonstration."""
    X = np.random.randn(n_samples, 1, n_timesteps)
    
    # Add class-specific patterns
    y = np.random.randint(0, n_classes, n_samples)
    
    for i in range(n_samples):
        if y[i] == 0:
            # Class 0: increasing trend
            X[i, 0, :] += np.linspace(0, 2, n_timesteps)
        elif y[i] == 1:
            # Class 1: sine wave
            X[i, 0, :] += np.sin(np.linspace(0, 4*np.pi, n_timesteps))
        else:
            # Class 2: decreasing trend
            X[i, 0, :] += np.linspace(2, 0, n_timesteps)
    
    return X, y


def main():
    """Main function demonstrating TPulse usage."""
    print("=" * 70)
    print("TPulse Classifier - Basic Usage Example")
    print("=" * 70)
    
    # Generate synthetic data
    print("\n1. Generating synthetic data...")
    X_train, y_train = generate_synthetic_data(n_samples=200, n_timesteps=50, n_classes=3)
    X_test, y_test = generate_synthetic_data(n_samples=50, n_timesteps=50, n_classes=3)
    
    print(f"   Train shape: {X_train.shape}")
    print(f"   Test shape: {X_test.shape}")
    print(f"   Classes: {np.unique(y_train)}")
    
    # Example 1: Default configuration
    print("\n2. Training with default configuration...")
    clf = TPulseClassifier(
        num_epochs=10,  # Reduced for quick demo
        batch_size=32,
        random_state=42,
        verbose=1
    )
    
    clf.fit(X_train, y_train)
    
    # Predict
    y_pred = clf.predict(X_test)
    accuracy = np.mean(y_pred == y_test)
    
    print(f"\n   Test Accuracy: {accuracy:.4f}")
    
    # Get probabilities
    y_proba = clf.predict_proba(X_test)
    print(f"   Prediction probabilities shape: {y_proba.shape}")
    
    # Example 2: Fast configuration
    print("\n3. Training with 'fast' preset...")
    clf_fast = TPulseClassifier(
        config='fast',
        random_state=42,
        verbose=1
    )
    
    clf_fast.fit(X_train, y_train)
    y_pred_fast = clf_fast.predict(X_test)
    accuracy_fast = np.mean(y_pred_fast == y_test)
    
    print(f"\n   Test Accuracy (fast): {accuracy_fast:.4f}")
    
    # Example 3: Custom configuration
    print("\n4. Training with custom configuration...")
    clf_custom = TPulseClassifier(
        num_epochs=15,
        batch_size=16,
        learning_rate=5e-5,
        dropout=0.4,
        use_mixup=True,
        mixup_alpha=0.3,
        early_stopping=True,
        patience=5,
        random_state=42,
        verbose=1
    )
    
    clf_custom.fit(X_train, y_train)
    y_pred_custom = clf_custom.predict(X_test)
    accuracy_custom = np.mean(y_pred_custom == y_test)
    
    print(f"\n   Test Accuracy (custom): {accuracy_custom:.4f}")
    
    # Summary
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"Default config accuracy:  {accuracy:.4f}")
    print(f"Fast config accuracy:     {accuracy_fast:.4f}")
    print(f"Custom config accuracy:   {accuracy_custom:.4f}")
    print("=" * 70)
    
    # Training history
    if hasattr(clf, 'training_history_'):
        print("\n5. Training History (last 5 epochs):")
        for record in clf.training_history_[-5:]:
            print(f"   Epoch {record['epoch']}: "
                  f"Val Acc = {record['val_acc']:.4f}, "
                  f"Train Loss = {record['train_loss']:.4f}")


if __name__ == "__main__":
    main()
