
import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from concurrent.futures import ThreadPoolExecutor
import time


def train_single_tree(X, y, tree_index, n_estimators, max_depth=None):
    tree = RandomForestClassifier(
        n_estimators=1,
        max_depth=max_depth,
        bootstrap=True,
        random_state=tree_index
    )
    tree.fit(X, y)
    return tree.estimators_[0]


def parallel_random_forest(X, y, n_estimators=10, max_depth=None, n_threads=1):
    trees = []
    
    with ThreadPoolExecutor(max_workers=n_threads) as executor:
        futures = [
            executor.submit(train_single_tree, X, y, i, n_estimators, max_depth)
            for i in range(n_estimators)
        ]
        
        for future in futures:
            trees.append(future.result())
    
    return trees


def predict_with_forest(trees, X):
    predictions = np.array([tree.predict(X) for tree in trees])
    final_predictions = np.apply_along_axis(
        lambda x: np.argmax(np.bincount(x.astype(int))), axis=0, arr=predictions
    )
    return final_predictions


def benchmark():
    dataset_sizes = [(1000, 10), (1000, 100), (1000, 1000), (5000, 10), (5000, 100),(5000, 1000), (10000, 10), (10000, 100), (10000,1000)]
    n_estimators = 50
    max_depth = 10
    thread_counts =  [1, 2, 4, 8, 12, 16 ,24, 32 ,40 ,56 ,64]
    
    for n_samples, n_features in dataset_sizes:
        print(f"\nDataset size: {n_samples} samples, {n_features} features")
        
        # Create the dataset
        X, y = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=n_features - 5,
            n_redundant=5,
            random_state=42
        )
        
        for n_threads in thread_counts:
            print(f"\nRunning with {n_threads} threads:")
            
            # Parallel random forest
            start_parallel = time.time()
            parallel_trees = parallel_random_forest(X, y, n_estimators=n_estimators, max_depth=max_depth, n_threads=n_threads)
            end_parallel = time.time()
            
            # Sklearn random forest
            start_sklearn = time.time()
            rf_model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, n_jobs=1)
            rf_model.fit(X, y)
            end_sklearn = time.time()
            
            # Prediction and accuracy
            y_pred_parallel = predict_with_forest(parallel_trees, X)
            y_pred_sklearn = rf_model.predict(X)
            
            accuracy_parallel = accuracy_score(y, y_pred_parallel)
            accuracy_sklearn = accuracy_score(y, y_pred_sklearn)
            
            print(f"Parallel Random Forest Time: {end_parallel - start_parallel:.2f} seconds")
            print(f"Sklearn Random Forest Time: {end_sklearn - start_sklearn:.2f} seconds")
            print(f"Speedup: {(end_sklearn - start_sklearn) / (end_parallel - start_parallel):.2f}x")
            print(f"Accuracy (Parallel): {accuracy_parallel:.4f}")
            print(f"Accuracy (Sklearn): {accuracy_sklearn:.4f}")


if __name__ == "__main__":
    benchmark()
