"""
Hyperparameter Tuning Example using Optuna.

This script demonstrates automated hyperparameter optimisation
for TSPulseClassifier to achieve maximum accuracy.
"""

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


from sktime_ext.classification.deep_learning.tspulse import TSPulseClassifier

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
except ImportError:
    print("Optuna not installed. Install with: pip install optuna")
    sys.exit(1)


def generate_synthetic_data(n_samples=100, n_timesteps=50, n_classes=3):
    """Generate synthetic time series data."""
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


def objective(trial, X_train, y_train, X_val, y_val):
    """
    Optuna objective function for hyperparameter optimisation.

    Defines the search space over TSPulseClassifier's actual parameters
    and returns validation accuracy to maximise.
    """
    
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)
    batch_size    = trial.suggest_categorical('batch_size', [16, 32, 64])
    freeze_backbone = trial.suggest_categorical('freeze_backbone', [True, False])
    mask_ratio    = trial.suggest_float('mask_ratio', 0.1, 0.5)

    clf = TSPulseClassifier(
        n_epochs=10,            # reduced for faster tuning
        batch_size=batch_size,
        learning_rate=learning_rate,
        freeze_backbone=freeze_backbone,
        mask_ratio=mask_ratio,
        random_state=42,
        verbose=False,
    )

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_val)
    return float(np.mean(y_pred == y_val))


def main():
    """Main hyperparameter tuning function."""
    print("=" * 80)
    print("TSPulse Classifier - Hyperparameter Tuning with Optuna")
    print("=" * 80)

    # Generate data
    print("\n1. Generating synthetic data...")
    X_train, y_train = generate_synthetic_data(n_samples=300, n_timesteps=50, n_classes=3)
    X_val,   y_val   = generate_synthetic_data(n_samples=100, n_timesteps=50, n_classes=3)
    X_test,  y_test  = generate_synthetic_data(n_samples=100, n_timesteps=50, n_classes=3)

    print(f"   Train shape: {X_train.shape}")
    print(f"   Val shape:   {X_val.shape}")
    print(f"   Test shape:  {X_test.shape}")

    # Create Optuna study
    print("\n2. Creating Optuna study...")
    study = optuna.create_study(
        direction='maximize',
        study_name='tspulse_tuning',
        sampler=optuna.samplers.TPESampler(seed=42),
    )

    N_TRIALS = 20
    print(f"\n3. Running {N_TRIALS} trials...")

    study.optimize(
        lambda trial: objective(trial, X_train, y_train, X_val, y_val),
        n_trials=N_TRIALS,
        show_progress_bar=True,
        n_jobs=1,
    )

    # Results
    print("\n" + "=" * 80)
    print("OPTIMISATION RESULTS")
    print("=" * 80)
    print(f"\nBest validation accuracy: {study.best_value:.4f}")
    print("\nBest hyperparameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value:.6f}" if isinstance(value, float) else f"  {key}: {value}")

    # Train final model with best params
    print("\n4. Training final model with best hyperparameters...")
    best = study.best_params

    clf_best = TSPulseClassifier(
        n_epochs=30,
        batch_size=best['batch_size'],
        learning_rate=best['learning_rate'],
        freeze_backbone=best['freeze_backbone'],
        mask_ratio=best['mask_ratio'],
        random_state=42,
        verbose=True,
    )
    clf_best.fit(X_train, y_train)
    y_pred_best = clf_best.predict(X_test)
    acc_best = float(np.mean(y_pred_best == y_test))

    # Compare with default
    print("\n5. Comparing with default configuration...")
    clf_default = TSPulseClassifier(n_epochs=30, random_state=42, verbose=False)
    clf_default.fit(X_train, y_train)
    y_pred_default = clf_default.predict(X_test)
    acc_default = float(np.mean(y_pred_default == y_test))

    print(f"   Default config accuracy: {acc_default:.4f}")
    print(f"   Tuned config accuracy:   {acc_best:.4f}")
    print(f"   Improvement:             {acc_best - acc_default:+.4f}")

    # Optional plot
    try:
        import matplotlib.pyplot as plt

        print("\n6. Saving optimisation history plot...")
        trials_df = study.trials_dataframe()

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        axes[0].plot(trials_df['number'], trials_df['value'], 'b-', alpha=0.3)
        axes[0].plot(trials_df['number'], trials_df['value'].cummax(), 'r-', linewidth=2)
        axes[0].set_xlabel('Trial')
        axes[0].set_ylabel('Validation Accuracy')
        axes[0].set_title('Optimisation History')
        axes[0].grid(True, alpha=0.3)
        axes[0].legend(['Trial accuracy', 'Best accuracy'])

        try:
            importance = optuna.importance.get_param_importances(study)
            axes[1].barh(list(importance.keys()), list(importance.values()))
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
