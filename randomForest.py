import numpy as np
import pandas as pd
import time
import multiprocessing
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def generate_dataset(n_samples, n_features):
    """
    Generate a synthetic classification dataset.
    
    Args:
        n_samples (int): Number of samples in the dataset
        n_features (int): Number of features
    
    Returns:
        tuple: X (features), y (labels)
    """
    X, y = make_classification(
        n_samples=n_samples, 
        n_features=n_features, 
        n_informative=int(n_features * 0.7), 
        n_redundant=int(n_features * 0.2), 
        random_state=42
    )
    return X, y

def benchmark_random_forest(X, y, max_threads=None, n_estimators=100):
    """
    Benchmark RandomForest with varying thread counts.
    
    Args:
        X (numpy.ndarray): Feature matrix
        y (numpy.ndarray): Target labels
        max_threads (int, optional): Maximum number of threads to test. 
                                     Defaults to number of CPU cores.
        n_estimators (int): Number of trees in the forest
    
    Returns:
        pandas.DataFrame: Performance benchmarking results
    """
    if max_threads is None:
        max_threads = multiprocessing.cpu_count()
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Benchmark results storage
    results = []
    
    for n_jobs in range(1, max_threads + 1):
        start_time = time.time()
        
        # Create and train RandomForest
        rf = RandomForestClassifier(
            n_estimators=n_estimators, 
            n_jobs=n_jobs,  # Parallel processing
            random_state=42
        )
        rf.fit(X_train, y_train)
        
        # Predict and calculate accuracy
        y_pred = rf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        end_time = time.time()
        
        # Calculate metrics
        total_time = end_time - start_time
        
        results.append({
            'Threads': n_jobs,
            'Training Time (s)': total_time,
            'Accuracy (%)': accuracy * 100
        })
    
    return pd.DataFrame(results)

def main():
    """
    Main function to demonstrate parallel RandomForest performance.
    """
    # Dataset sizes to test
    dataset_sizes = [1000, 10000, 100000]
    feature_count = 20
    
    print("Parallel Random Forest Performance Benchmarking")
    print("=" * 50)
    
    for size in dataset_sizes:
        print(f"\nDataset Size: {size} samples")
        
        # Generate dataset
        X, y = generate_dataset(size, feature_count)
        
        # Benchmark performance
        results = benchmark_random_forest(X, y)
        
        # Display results
        print("\nPerformance Metrics:")
        print(results)
        
        # Calculate and print speedup
        baseline_time = results.loc[0, 'Training Time (s)']
        results['Speedup'] = baseline_time / results['Training Time (s)']
        
        print("\nSpeedup Comparison:")
        print(results[['Threads', 'Speedup']])
        print("-" * 50)

if __name__ == '__main__':
    main()