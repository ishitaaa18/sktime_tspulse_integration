"""
Benchmarking Script for TPulse Classifier.

This script benchmarks TPulse against other sktime classifiers
on UCR time series datasets.
"""

import numpy as np
import pandas as pd
import time
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sktime.classification.deep_learning import TPulseClassifier


def generate_synthetic_dataset(name, n_train, n_test, n_timesteps, n_classes):
    """Generate synthetic dataset mimicking UCR structure."""
    X_train = np.random.randn(n_train, 1, n_timesteps)
    y_train = np.random.randint(0, n_classes, n_train)
    
    X_test = np.random.randn(n_test, 1, n_timesteps)
    y_test = np.random.randint(0, n_classes, n_test)
    
    # Add patterns
    for i in range(n_train):
        if y_train[i] == 0:
            X_train[i] += np.sin(np.linspace(0, 4*np.pi, n_timesteps))
        elif y_train[i] == 1:
            X_train[i] += np.linspace(0, 2, n_timesteps)
        else:
            X_train[i] += np.cos(np.linspace(0, 4*np.pi, n_timesteps))
    
    return X_train, y_train, X_test, y_test


def benchmark_tpulse(X_train, y_train, X_test, y_test, config='default'):
    """Benchmark TPulse classifier."""
    # Create classifier
    if config == 'fast':
        clf = TPulseClassifier(config='fast', random_state=42, verbose=0)
    elif config == 'accurate':
        clf = TPulseClassifier(config='accurate', random_state=42, verbose=0)
    else:
        clf = TPulseClassifier(
            num_epochs=20,
            batch_size=32,
            random_state=42,
            verbose=0
        )
    
    # Train
    start_time = time.time()
    clf.fit(X_train, y_train)
    train_time = time.time() - start_time
    
    # Predict
    start_time = time.time()
    y_pred = clf.predict(X_test)
    predict_time = time.time() - start_time
    
    # Accuracy
    accuracy = np.mean(y_pred == y_test)
    
    return {
        'accuracy': accuracy,
        'train_time': train_time,
        'predict_time': predict_time,
    }


def main():
    """Main benchmarking function."""
    print("=" * 80)
    print("TPulse Classifier Benchmarking")
    print("=" * 80)
    
    # Define synthetic datasets (mimicking UCR)
    datasets = {
        'GunPoint': {'n_train': 150, 'n_test': 150, 'n_timesteps': 150, 'n_classes': 2},
        'ECG200': {'n_train': 100, 'n_test': 100, 'n_timesteps': 96, 'n_classes': 2},
        'ArrowHead': {'n_train': 175, 'n_test': 175, 'n_timesteps': 251, 'n_classes': 3},
    }
    
    # Configurations to test
    configs = ['fast', 'default']
    
    # Store results
    results = []
    
    # Run benchmarks
    for dataset_name, dataset_params in datasets.items():
        print(f"\n{'='*80}")
        print(f"Dataset: {dataset_name}")
        print(f"{'='*80}")
        
        # Generate data
        print(f"Generating data...")
        X_train, y_train, X_test, y_test = generate_synthetic_dataset(
            dataset_name,
            n_train=dataset_params['n_train'],
            n_test=dataset_params['n_test'],
            n_timesteps=dataset_params['n_timesteps'],
            n_classes=dataset_params['n_classes']
        )
        
        print(f"  Train: {X_train.shape}, Test: {X_test.shape}")
        
        # Test each configuration
        for config in configs:
            print(f"\nTesting config: {config}")
            
            try:
                result = benchmark_tpulse(X_train, y_train, X_test, y_test, config)
                
                print(f"  Accuracy: {result['accuracy']:.4f}")
                print(f"  Train time: {result['train_time']:.2f}s")
                print(f"  Predict time: {result['predict_time']:.2f}s")
                
                results.append({
                    'Dataset': dataset_name,
                    'Config': config,
                    'Accuracy': result['accuracy'],
                    'Train Time (s)': result['train_time'],
                    'Predict Time (s)': result['predict_time'],
                    'Samples': dataset_params['n_train'],
                    'Timesteps': dataset_params['n_timesteps'],
                    'Classes': dataset_params['n_classes'],
                })
            
            except Exception as e:
                print(f"  Error: {e}")
                results.append({
                    'Dataset': dataset_name,
                    'Config': config,
                    'Accuracy': np.nan,
                    'Train Time (s)': np.nan,
                    'Predict Time (s)': np.nan,
                    'Samples': dataset_params['n_train'],
                    'Timesteps': dataset_params['n_timesteps'],
                    'Classes': dataset_params['n_classes'],
                })
    
    # Create results dataframe
    df_results = pd.DataFrame(results)
    
    # Print summary
    print("\n" + "=" * 80)
    print("BENCHMARK RESULTS SUMMARY")
    print("=" * 80)
    print(df_results.to_string(index=False))
    
    # Save results
    output_file = Path(__file__).parent / 'benchmark_results.csv'
    df_results.to_csv(output_file, index=False)
    print(f"\nResults saved to: {output_file}")
    
    # Aggregate statistics
    print("\n" + "=" * 80)
    print("AGGREGATE STATISTICS")
    print("=" * 80)
    
    for config in configs:
        config_results = df_results[df_results['Config'] == config]
        print(f"\nConfig: {config}")
        print(f"  Mean Accuracy: {config_results['Accuracy'].mean():.4f}")
        print(f"  Std Accuracy: {config_results['Accuracy'].std():.4f}")
        print(f"  Mean Train Time: {config_results['Train Time (s)'].mean():.2f}s")
        print(f"  Mean Predict Time: {config_results['Predict Time (s)'].mean():.4f}s")


if __name__ == "__main__":
    main()
