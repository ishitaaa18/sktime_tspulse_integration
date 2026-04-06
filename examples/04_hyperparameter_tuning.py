"""
Hyperparameter Tuning Example using Optuna.

This script demonstrates automated hyperparameter optimization
for TPulse classifier to achieve maximum accuracy.
"""

import numpy as np
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sktime.classification.deep_learning import TPulseClassifier

try:
    import optuna
except ImportError:
    print("Optuna not installed. Install with: pip install optuna")
    sys.exit(1)


def generate_synthetic_data(n_samples=100, n_timesteps=50, n_classes=3):
    """Generate synthetic time series data."""
    X = np.random.randn(n_samples, 1, n_timesteps)
    y = np.random.randint(0, n_classes, n_samples)
    
    for i in range(n_samples):
        if y[i] == 0:
            X[i, 0, :] += np.linspace(0, 2, n_timesteps)
        elif y[i] == 1:
            X[i, 0, :] += np.sin(np.linspace(0, 4*np.pi, n_timesteps))
        else:
            X[i, 0, :] += np.linspace(2, 0, n_timesteps)
    
    return X, y


def objective(trial, X_train, y_train):
    """
    Optuna objective function for hyperparameter optimization.
    
    This function defines the hyperparameter search space and
    returns the validation accuracy to maximize.
    """
    # Define hyperparameter search space
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-3)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
    dropout = trial.suggest_uniform('dropout', 0.1, 0.5)
    weight_decay = trial.suggest_loguniform('weight_decay', 1e-4, 1e-1)
    mixup_alpha = trial.suggest_uniform('mixup_alpha', 0.1, 0.4)
    unfreeze_after_epochs = trial.suggest_int('unfreeze_after_epochs', 3, 7)
    
    # Create classifier with suggested hyperparameters
    clf = TPulseClassifier(
        num_epochs=15,  # Reduced for faster tuning
        batch_size=batch_size,
        learning_rate=learning_rate,
        dropout=dropout,
        weight_decay=weight_decay,
        mixup_alpha=mixup_alpha,
        unfreeze_after_epochs=unfreeze_after_epochs,
        early_stopping=True,
        patience=5,
        validation_split=0.2,
        random_state=42,
        verbose=0  # Silent for cleaner output
    )
    
    # Train and get validation accuracy
    clf.fit(X_train, y_train)
    
    # Return best validation accuracy
    if hasattr(clf, 'best_val_accuracy_'):
        return clf.best_val_accuracy_
    else:
        # Fallback: use last validation accuracy
        if clf.training_history_:
            return clf.training_history_[-1]['val_acc']
        return 0.0


def main():
    """Main hyperparameter tuning function."""
    print("=" * 80)
    print("TPulse Classifier - Hyperparameter Tuning with Optuna")
    print("=" * 80)
    
    # Generate data
    print("\n1. Generating synthetic data...")
    X_train, y_train = generate_synthetic_data(n_samples=300, n_timesteps=50, n_classes=3)
    X_test, y_test = generate_synthetic_data(n_samples=100, n_timesteps=50, n_classes=3)
    
    print(f"   Train shape: {X_train.shape}")
    print(f"   Test shape: {X_test.shape}")
    
    # Create Optuna study
    print("\n2. Creating Optuna study...")
    study = optuna.create_study(
        direction='maximize',
        study_name='tpulse_tuning',
        sampler=optuna.samplers.TPESampler(seed=42)
    )
    
    # Run optimization
    print("\n3. Running hyperparameter optimization...")
    print(f"   This will run {20} trials. Each trial trains a model.")
    print(f"   This may take several minutes...\n")
    
    study.optimize(
        lambda trial: objective(trial, X_train, y_train),
        n_trials=20,
        show_progress_bar=True,
        n_jobs=1  # Set to -1 for parallel trials (if you have multiple GPUs)
    )
    
    # Print results
    print("\n" + "=" * 80)
    print("OPTIMIZATION RESULTS")
    print("=" * 80)
    
    print(f"\nBest validation accuracy: {study.best_value:.4f}")
    print(f"\nBest hyperparameters:")
    for key, value in study.best_params.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.6f}")
        else:
            print(f"  {key}: {value}")
    
    # Train final model with best hyperparameters
    print("\n4. Training final model with best hyperparameters...")
    best_params = study.best_params
    
    clf_best = TPulseClassifier(
        num_epochs=30,  # Use more epochs for final model
        batch_size=best_params['batch_size'],
        learning_rate=best_params['learning_rate'],
        dropout=best_params['dropout'],
        weight_decay=best_params['weight_decay'],
        mixup_alpha=best_params['mixup_alpha'],
        unfreeze_after_epochs=best_params['unfreeze_after_epochs'],
        early_stopping=True,
        patience=10,
        validation_split=0.2,
        random_state=42,
        verbose=1
    )
    
    clf_best.fit(X_train, y_train)
    
    # Evaluate on test set
    y_pred = clf_best.predict(X_test)
    test_accuracy = np.mean(y_pred == y_test)
    
    print(f"\n5. Test set accuracy: {test_accuracy:.4f}")
    
    # Compare with default configuration
    print("\n6. Comparing with default configuration...")
    clf_default = TPulseClassifier(
        num_epochs=30,
        random_state=42,
        verbose=0
    )
    clf_default.fit(X_train, y_train)
    y_pred_default = clf_default.predict(X_test)
    accuracy_default = np.mean(y_pred_default == y_test)
    
    print(f"   Default config accuracy: {accuracy_default:.4f}")
    print(f"   Tuned config accuracy:   {test_accuracy:.4f}")
    print(f"   Improvement:             {test_accuracy - accuracy_default:.4f}")
    
    # Plot optimization history
    try:
        import matplotlib.pyplot as plt
        
        print("\n7. Saving optimization history plot...")
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # Optimization history
        trials_df = study.trials_dataframe()
        axes[0].plot(trials_df['number'], trials_df['value'], 'b-', alpha=0.3)
        axes[0].plot(trials_df['number'], trials_df['value'].cummax(), 'r-', linewidth=2)
        axes[0].set_xlabel('Trial')
        axes[0].set_ylabel('Validation Accuracy')
        axes[0].set_title('Optimization History')
        axes[0].grid(True, alpha=0.3)
        axes[0].legend(['Trial accuracy', 'Best accuracy'])
        
        # Parameter importances
        try:
            importance = optuna.importance.get_param_importances(study)
            params = list(importance.keys())
            values = list(importance.values())
            
            axes[1].barh(params, values)
            axes[1].set_xlabel('Importance')
            axes[1].set_title('Hyperparameter Importance')
            axes[1].grid(True, alpha=0.3)
        except Exception:
            axes[1].text(0.5, 0.5, 'Importance calculation failed',
                        ha='center', va='center', transform=axes[1].transAxes)
        
        plt.tight_layout()
        output_path = Path(__file__).parent / 'tuning_results.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"   Plot saved to: {output_path}")
        
    except ImportError:
        print("\n   Matplotlib not installed. Skipping plot generation.")
    
    print("\n" + "=" * 80)
    print("Hyperparameter tuning complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
