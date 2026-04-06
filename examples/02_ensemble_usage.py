"""
Ensemble Usage Example for Maximum Accuracy.

This script demonstrates how to use TPulseEnsemble
for improved accuracy through model averaging.
"""

import numpy as np
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sktime.classification.deep_learning import TPulseEnsemble


def generate_synthetic_data(n_samples=100, n_timesteps=50, n_classes=3):
    """Generate synthetic time series data for demonstration."""
    X = np.random.randn(n_samples, 1, n_timesteps)
    
    # Add class-specific patterns
    y = np.random.randint(0, n_classes, n_samples)
    
    for i in range(n_samples):
        if y[i] == 0:
            X[i, 0, :] += np.linspace(0, 2, n_timesteps)
        elif y[i] == 1:
            X[i, 0, :] += np.sin(np.linspace(0, 4*np.pi, n_timesteps))
        else:
            X[i, 0, :] += np.linspace(2, 0, n_timesteps)
    
    return X, y


def main():
    """Main function demonstrating ensemble usage."""
    print("=" * 70)
    print("TPulse Ensemble - Maximum Accuracy Example")
    print("=" * 70)
    
    # Generate data
    print("\n1. Generating synthetic data...")
    X_train, y_train = generate_synthetic_data(n_samples=200, n_timesteps=50, n_classes=3)
    X_test, y_test = generate_synthetic_data(n_samples=50, n_timesteps=50, n_classes=3)
    
    print(f"   Train shape: {X_train.shape}")
    print(f"   Test shape: {X_test.shape}")
    
    # Create ensemble
    print("\n2. Creating ensemble with 3 models...")
    ensemble = TPulseEnsemble(
        n_models=3,
        base_config='fast',  # Use fast config for quick demo
        vary_hyperparameters=True,
        vary_dropout=True,
        vary_mixup=True,
        voting='soft',
        n_jobs=1,  # Set to -1 for parallel training
        verbose=1,
        random_state=42
    )
    
    # Train ensemble
    print("\n3. Training ensemble...")
    ensemble.fit(X_train, y_train)
    
    # Predict
    print("\n4. Making predictions...")
    y_pred_ensemble = ensemble.predict(X_test)
    accuracy_ensemble = np.mean(y_pred_ensemble == y_test)
    
    print(f"\n   Ensemble Test Accuracy: {accuracy_ensemble:.4f}")
    
    # Get individual model predictions
    print("\n5. Analyzing individual models...")
    individual_preds = ensemble.get_model_predictions(X_test)
    
    for i, pred in individual_preds.items():
        acc = np.mean(pred == y_test)
        print(f"   Model {i+1} accuracy: {acc:.4f}")
    
    # Compute diversity metrics
    print("\n6. Computing ensemble diversity...")
    diversity = ensemble.get_ensemble_diversity(X_test, y_test)
    
    print(f"   Pairwise disagreement: {diversity['pairwise_disagreement']:.4f}")
    print(f"   Avg individual accuracy: {diversity['avg_individual_accuracy']:.4f}")
    print(f"   Std individual accuracy: {diversity['std_individual_accuracy']:.4f}")
    
    # Summary
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"Ensemble Accuracy:        {accuracy_ensemble:.4f}")
    print(f"Best Single Model:        {diversity['max_individual_accuracy']:.4f}")
    print(f"Worst Single Model:       {diversity['min_individual_accuracy']:.4f}")
    print(f"Ensemble Improvement:     {accuracy_ensemble - diversity['avg_individual_accuracy']:.4f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
