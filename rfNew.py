# import numpy as np
# import pandas as pd
# import time
# import multiprocessing
# from concurrent.futures import ProcessPoolExecutor
# from sklearn.datasets import make_classification
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score
# from sklearn.preprocessing import LabelEncoder

# class ParallelRandomForest:
#     @staticmethod
#     def generate_dataset(n_samples, n_features):
#         """Generate synthetic classification dataset."""
#         X, y = make_classification(
#             n_samples=n_samples, 
#             n_features=n_features, 
#             n_informative=int(n_features * 0.7), 
#             n_redundant=int(n_features * 0.2), 
#             random_state=42
#         )
#         return X, y

#     @staticmethod
#     def train_and_evaluate_forest(args):
#         """
#         Train and evaluate RandomForest with distributed dataset splitting.
        
#         Args:
#             args (tuple): Contains training configuration
        
#         Returns:
#             dict: Performance metrics
#         """
#         X_train, X_test, y_train, y_test, n_estimators, random_state = args
        
#         # Ensure labels are encoded
#         le = LabelEncoder()
#         y_train_encoded = le.fit_transform(y_train)
#         y_test_encoded = le.transform(y_test)
        
#         # Split training data further for sub-ensemble training
#         split_indices = np.array_split(np.arange(len(X_train)), multiprocessing.cpu_count())
        
#         # Custom parallel training of sub-ensembles
#         sub_forests = []
#         for indices in split_indices:
#             sub_rf = RandomForestClassifier(
#                 n_estimators=max(1, n_estimators // multiprocessing.cpu_count()),
#                 random_state=random_state
#             )
#             sub_rf.fit(X_train[indices], y_train_encoded[indices])
#             sub_forests.append(sub_rf)
        
#         # Aggregate predictions with robust handling
#         def safe_predict_proba(forest, X):
#             try:
#                 return forest.predict_proba(X)
#             except Exception:
#                 # Fallback to predict method if predict_proba fails
#                 return np.eye(len(np.unique(y_train_encoded)))[forest.predict(X)]
        
#         # Collect predictions from all sub-forests
#         predictions = [safe_predict_proba(forest, X_test) for forest in sub_forests]
        
#         # Ensure consistent prediction shape
#         predictions = [
#             pred if pred.ndim == 2 else pred.reshape(-1, 1) 
#             for pred in predictions
#         ]
        
#         # Aggregate predictions with voting
#         final_predictions = np.mean(predictions, axis=0)
#         final_class_predictions = np.argmax(final_predictions, axis=1)
        
#         # Decode back to original labels
#         final_decoded_predictions = le.inverse_transform(final_class_predictions)
        
#         return {
#             'accuracy': accuracy_score(y_test, final_decoded_predictions),
#             'training_time': time.time()  # Placeholder for timing
#         }
    
#     def advanced_parallel_train(self, X, y, n_estimators=100, max_workers=None):
#         """
#         Advanced parallelized training with multi-level parallelism.
        
#         Args:
#             X (np.ndarray): Feature matrix
#             y (np.ndarray): Target labels
#             n_estimators (int): Number of trees
#             max_workers (int, optional): Max parallel workers
        
#         Returns:
#             pd.DataFrame: Performance results
#         """
#         if max_workers is None:
#             max_workers = multiprocessing.cpu_count()
        
#         # Split data
#         X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
#         # Prepare arguments for parallel execution
#         parallel_args = [(
#             X_train, X_test, y_train, y_test, 
#             n_estimators, np.random.randint(1000)
#         ) for _ in range(max_workers)]
        
#         start_time = time.time()
        
#         # Use ProcessPoolExecutor for true parallel execution
#         with ProcessPoolExecutor(max_workers=max_workers) as executor:
#             results = list(executor.map(self.train_and_evaluate_forest, parallel_args))
        
#         total_time = time.time() - start_time
        
#         # Aggregate results
#         avg_accuracy = np.mean([r['accuracy'] for r in results])
        
#         return pd.DataFrame([{
#             'Workers': max_workers,
#             'Total Training Time': total_time,
#             'Average Accuracy': avg_accuracy * 100
#         }])

# def main():
#     # Test different dataset configurations
#     dataset_configs = [
#         (1000, 20),   # Small dataset
#         (10000, 50),  # Medium dataset
#         (100000, 100) # Large dataset
#     ]
    
#     parallel_rf = ParallelRandomForest()
    
#     # Store results for comparison
#     all_results = []
    
#     for size, features in dataset_configs:
#         print(f"\nDataset Size: {size} samples, Features: {features}")
        
#         # Generate dataset
#         X, y = parallel_rf.generate_dataset(size, features)
        
#         # Advanced parallel training
#         results = parallel_rf.advanced_parallel_train(X, y)
        
#         print("\nAdvanced Parallel Performance:")
#         print(results)
        
#         all_results.append(results)
    
#     # Compile overall results
#     final_results = pd.concat(all_results)
#     print("\n\nOverall Performance Summary:")
#     print(final_results)

# if __name__ == '__main__':
#     main()

import numpy as np
import pandas as pd
import time
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

class ParallelRandomForestBenchmark:
    @staticmethod
    def generate_dataset(n_samples, n_features):
        """Generate synthetic classification dataset."""
        X, y = make_classification(
            n_samples=n_samples, 
            n_features=n_features, 
            n_informative=int(n_features * 0.7), 
            n_redundant=int(n_features * 0.2), 
            random_state=42
        )
        return X, y

    @staticmethod
    def benchmark_training(X, y, max_threads=None, n_estimators=100):
        """
        Benchmark RandomForest training with varying thread counts.
        
        Args:
            X (np.ndarray): Feature matrix
            y (np.ndarray): Target labels
            max_threads (int, optional): Maximum number of threads to test
            n_estimators (int): Number of trees in the forest
        
        Returns:
            pd.DataFrame: Performance benchmarking results
        """
        if max_threads is None:
            max_threads = multiprocessing.cpu_count()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Label encoding
        le = LabelEncoder()
        y_train_encoded = le.fit_transform(y_train)
        y_test_encoded = le.transform(y_test)
        
        # Benchmark results storage
        results = []
        
        # Test different thread configurations
        for n_jobs in range(1, max_threads + 1):
            start_time = time.time()
            
            # Create and train RandomForest
            rf = RandomForestClassifier(
                n_estimators=n_estimators, 
                n_jobs=n_jobs,  # Parallel processing
                random_state=42
            )
            rf.fit(X_train, y_train_encoded)
            
            # Predict and calculate accuracy
            y_pred = rf.predict(X_test)
            accuracy = accuracy_score(y_test_encoded, y_pred)
            
            end_time = time.time()
            
            # Calculate metrics
            total_time = end_time - start_time
            
            results.append({
                'Threads': n_jobs,
                'Training Time (s)': total_time,
                'Accuracy (%)': accuracy * 100
            })
        
        # Convert to DataFrame and calculate speedup
        results_df = pd.DataFrame(results)
        baseline_time = results_df.loc[0, 'Training Time (s)']
        results_df['Speedup'] = baseline_time / results_df['Training Time (s)']
        
        return results_df

    @staticmethod
    def run_comprehensive_benchmark():
        """
        Run comprehensive benchmarking across different dataset sizes and thread counts.
        
        Returns:
            dict: Benchmark results for different dataset configurations
        """
        # Dataset configurations to test
        dataset_configs = [
            {'samples': 1000, 'features': 20, 'estimators': 50},
            {'samples': 10000, 'features': 50, 'estimators': 100},
            {'samples': 100000, 'features': 100, 'estimators': 200}
        ]
        
        # Store comprehensive results
        comprehensive_results = {}
        
        for config in dataset_configs:
            print(f"\n--- Dataset Configuration ---")
            print(f"Samples: {config['samples']}, Features: {config['features']}")
            
            # Generate dataset
            X, y = ParallelRandomForestBenchmark.generate_dataset(
                config['samples'], config['features']
            )
            
            # Benchmark training
            benchmark_results = ParallelRandomForestBenchmark.benchmark_training(
                X, y, 
                n_estimators=config['estimators']
            )
            
            # Display results
            print("\nPerformance Metrics:")
            print(benchmark_results)
            
            # Store results
            comprehensive_results[f"{config['samples']} samples"] = benchmark_results
        
        return comprehensive_results

def main():
    # Run comprehensive benchmark
    results = ParallelRandomForestBenchmark.run_comprehensive_benchmark()
    
    # Optional: Visualization preparation
    print("\n--- Comprehensive Benchmark Summary ---")
    for dataset, df in results.items():
        print(f"\nDataset: {dataset}")
        print("Best Speedup:", df['Speedup'].max())
        print("Optimal Threads:", df.loc[df['Speedup'].idxmax(), 'Threads'])

if __name__ == '__main__':
    main()