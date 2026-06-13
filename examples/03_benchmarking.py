"""
Benchmarking Script for TSPulse Classifier.

This script benchmarks TSPulseClassifier on synthetic UCR-style datasets
and reports accuracy and timing.
"""

import numpy as np
import pandas as pd
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


from sktime_ext.classification.deep_learning.tspulse import TSPulseClassifier


def generate_synthetic_dataset(n_train, n_test, n_timesteps, n_classes):
    """Generate a synthetic dataset mimicking UCR structure."""
    X_train = np.random.randn(n_train, 1, n_timesteps)
    y_train = np.random.randint(0, n_classes, n_train).astype(str)

    X_test = np.random.randn(n_test, 1, n_timesteps)
    y_test = np.random.randint(0, n_classes, n_test).astype(str)

    for i in range(n_train):
        if y_train[i] == '0':
            X_train[i] += np.sin(np.linspace(0, 4 * np.pi, n_timesteps))
        elif y_train[i] == '1':
            X_train[i] += np.linspace(0, 2, n_timesteps)
        else:
            X_train[i] += np.cos(np.linspace(0, 4 * np.pi, n_timesteps))

    return X_train, y_train, X_test, y_test


def benchmark_tspulse(X_train, y_train, X_test, y_test, n_epochs=20):
    """Train and evaluate a TSPulseClassifier, returning timing and accuracy."""
    
    clf = TSPulseClassifier(
        n_epochs=n_epochs,
        batch_size=32,
        random_state=42,
        verbose=False,
    )

    start = time.time()
    clf.fit(X_train, y_train)
    train_time = time.time() - start

    start = time.time()
    y_pred = clf.predict(X_test)
    predict_time = time.time() - start

    accuracy = np.mean(y_pred == y_test)

    return {
        'accuracy': accuracy,
        'train_time': train_time,
        'predict_time': predict_time,
    }


def main():
    """Main benchmarking function."""
    print("=" * 80)
    print("TSPulse Classifier Benchmarking")
    print("=" * 80)

    datasets = {
        'GunPoint':  {'n_train': 150, 'n_test': 150, 'n_timesteps': 150, 'n_classes': 2},
        'ECG200':    {'n_train': 100, 'n_test': 100, 'n_timesteps': 96,  'n_classes': 2},
        'ArrowHead': {'n_train': 175, 'n_test': 175, 'n_timesteps': 251, 'n_classes': 3},
    }

    # Two epoch budgets: quick vs. full
    epoch_configs = {'quick': 5, 'full': 20}

    results = []

    for dataset_name, params in datasets.items():
        print(f"\n{'=' * 80}")
        print(f"Dataset: {dataset_name}")
        print(f"{'=' * 80}")

        X_train, y_train, X_test, y_test = generate_synthetic_dataset(
            n_train=params['n_train'],
            n_test=params['n_test'],
            n_timesteps=params['n_timesteps'],
            n_classes=params['n_classes'],
        )
        print(f"  Train: {X_train.shape}, Test: {X_test.shape}")

        for config_name, n_epochs in epoch_configs.items():
            print(f"\n  Config: {config_name} ({n_epochs} epochs)")

            try:
                result = benchmark_tspulse(
                    X_train, y_train, X_test, y_test, n_epochs=n_epochs
                )
                print(f"    Accuracy:     {result['accuracy']:.4f}")
                print(f"    Train time:   {result['train_time']:.2f}s")
                print(f"    Predict time: {result['predict_time']:.4f}s")

                results.append({
                    'Dataset':          dataset_name,
                    'Config':           config_name,
                    'Epochs':           n_epochs,
                    'Accuracy':         result['accuracy'],
                    'Train Time (s)':   result['train_time'],
                    'Predict Time (s)': result['predict_time'],
                    'Samples':          params['n_train'],
                    'Timesteps':        params['n_timesteps'],
                    'Classes':          params['n_classes'],
                })

            except Exception as e:
                print(f"    Error: {e}")
                results.append({
                    'Dataset':          dataset_name,
                    'Config':           config_name,
                    'Epochs':           n_epochs,
                    'Accuracy':         np.nan,
                    'Train Time (s)':   np.nan,
                    'Predict Time (s)': np.nan,
                    'Samples':          params['n_train'],
                    'Timesteps':        params['n_timesteps'],
                    'Classes':          params['n_classes'],
                })

    df_results = pd.DataFrame(results)

    print("\n" + "=" * 80)
    print("BENCHMARK RESULTS SUMMARY")
    print("=" * 80)
    print(df_results.to_string(index=False))

    output_file = Path(__file__).parent / 'benchmark_results.csv'
    df_results.to_csv(output_file, index=False)
    print(f"\nResults saved to: {output_file}")

    print("\n" + "=" * 80)
    print("AGGREGATE STATISTICS")
    print("=" * 80)
    for config_name in epoch_configs:
        subset = df_results[df_results['Config'] == config_name]
        print(f"\nConfig: {config_name}")
        print(f"  Mean Accuracy:    {subset['Accuracy'].mean():.4f}")
        print(f"  Std Accuracy:     {subset['Accuracy'].std():.4f}")
        print(f"  Mean Train Time:  {subset['Train Time (s)'].mean():.2f}s")
        print(f"  Mean Predict Time:{subset['Predict Time (s)'].mean():.4f}s")


if __name__ == "__main__":
    main()
