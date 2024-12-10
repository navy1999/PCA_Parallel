import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from concurrent.futures import ThreadPoolExecutor
from scipy.spatial.distance import cdist
import time
import matplotlib.pyplot as plt

# k-NN function for prediction
def knn_predict(X_train, y_train, X_test, k):
    distances = cdist(X_test, X_train, metric='euclidean')
    neighbors = np.argsort(distances, axis=1)[:, :k]
    predictions = [np.bincount(y_train[neighbor]).argmax() for neighbor in neighbors]
    return np.array(predictions)

# Parallelized k-NN function
def parallel_knn(X_train, y_train, X_test, k, num_threads):
    def process_chunk(chunk):
        return knn_predict(X_train, y_train, chunk, k)

    chunk_size = len(X_test) // num_threads
    chunks = [X_test[i:i + chunk_size] for i in range(0, len(X_test), chunk_size)]

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        results = list(executor.map(process_chunk, chunks))

    return np.concatenate(results)

if __name__ == "__main__":
    # Generate a larger synthetic dataset
    X, y = make_classification(n_samples=10000, n_features=100, n_classes=2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Measure speedup
    k = 5
    num_threads_list = [1,2,3,4,5,6,7,8]
    speedups = []

    # Single-threaded baseline
    start_time = time.perf_counter()
    predictions = knn_predict(X_train, y_train, X_test, k)
    end_time = time.perf_counter()
    base_time = end_time - start_time
    print(f"Base time (single-threaded): {base_time:.4f} seconds")

    for num_threads in num_threads_list:
        start_time = time.perf_counter()
        predictions = parallel_knn(X_train, y_train, X_test, k, num_threads)
        end_time = time.perf_counter()
        thread_time = end_time - start_time
        speedup = base_time / thread_time
        speedups.append(speedup)
        print(f"Time with {num_threads} threads: {thread_time:.4f} seconds, Speedup: {speedup:.2f}x")

    # Plot speedup vs number of threads
    plt.figure(figsize=(10, 6))
    plt.plot(num_threads_list, speedups, marker='o')
    plt.title('Speedup vs Number of Threads')
    plt.xlabel('Number of Threads')
    plt.ylabel('Speedup')
    plt.grid()
    plt.show()
