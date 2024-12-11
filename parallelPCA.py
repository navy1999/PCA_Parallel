import numpy as np
from concurrent.futures import ThreadPoolExecutor
import time
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Function to generate synthetic datasets with varying dimensions and sample sizes
def generate_dataset(n_samples, n_features):
    return np.random.rand(n_samples, n_features)

# Compute covariance matrix
def compute_covariance(X):
    return np.dot(X.T, X) / (X.shape[0] - 1)

# Eigen decomposition
def eigen_decomposition(cov_matrix):
    eig_vals, eig_vecs = np.linalg.eigh(cov_matrix)
    sorted_indices = np.argsort(eig_vals)[::-1]
    return eig_vals[sorted_indices], eig_vecs[:, sorted_indices]

# Time PCA with a given number of threads
def time_pca(num_threads, X_std, num_runs=5):
    total_time = 0
    for _ in range(num_runs):
        start_time = time.perf_counter()
        cov_matrix = compute_covariance(X_std)
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            eig_vals, eig_vecs = executor.submit(eigen_decomposition, cov_matrix).result()
        end_time = time.perf_counter()
        total_time += (end_time - start_time)
    return total_time / num_runs

if __name__ == "__main__":
    # Dataset configurations: (number of samples, number of features)
    dataset_configs = [
        (1000, 50),
        (1000, 500),
        (1000, 5000),
        (5000, 50),
        (5000, 500),
        (5000, 5000),
        (10000, 50),
        (10000, 500),
        (10000, 5000)
    ]

    thread_counts = [1, 2, 4, 8, 12, 16 ,24, 32 ,40 ,56 ,64]
    results = {}

    for n_samples, n_features in dataset_configs:
        print(f"Processing dataset with {n_samples} samples and {n_features} features...")
        X = generate_dataset(n_samples, n_features)

        # Standardize the dataset
        scaler = StandardScaler()
        X_std = scaler.fit_transform(X)

        # Measure execution times for different thread counts
        thread_execution_times = [
            (num_threads, time_pca(num_threads, X_std))
            for num_threads in thread_counts
        ]

        single_thread_time = thread_execution_times[0][1]
        speedup = [
            (num_threads, single_thread_time / exec_time if exec_time > 0 else 0)
            for num_threads, exec_time in thread_execution_times
        ]

        results[(n_samples, n_features)] = {
            "execution_times": thread_execution_times,
            "speedups": speedup,
        }

    # Plot results in one image with legends
    plt.figure(figsize=(14, 12))

    # Plot Execution Time
    plt.subplot(2, 1, 1)
    for ((n_samples, n_features), data) in results.items():
        threads, times = zip(*data["execution_times"])
        plt.plot(threads, times, marker="o", linestyle="-", label=f"{n_samples}x{n_features}")

    plt.title("Execution Time vs Number of Threads")
    plt.xlabel("Number of Threads")
    plt.ylabel("Execution Time (seconds)")
    plt.grid(True)
    plt.legend(title="Dataset Size")
    
    # Plot Speedup
    plt.subplot(2, 1, 2)
    for ((n_samples, n_features), data) in results.items():
        threads, _ = zip(*data["execution_times"])
        _, speedups = zip(*data["speedups"])
        plt.plot(threads, speedups, marker="o", linestyle="-", label=f"{n_samples}x{n_features}")

    plt.title("Speedup vs Number of Threads")
    plt.xlabel("Number of Threads")
    plt.ylabel("Speedup")
    plt.grid(True)
    plt.legend(title="Dataset Size")

    plt.tight_layout()
    plt.savefig("pca_performance_comparison.png")
    plt.show()
