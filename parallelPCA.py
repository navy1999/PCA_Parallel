import numpy as np
from concurrent.futures import ThreadPoolExecutor
import time
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def generate_synthetic_data():
    logging.info("Generating synthetic dataset...")
    X, _ = make_classification(n_samples=10000, n_features=1000, random_state=42)
    logging.info("Synthetic dataset generated.")
    return X

def compute_covariance(X):
    return np.dot(X.T, X) / (X.shape[0] - 1)

def eigen_decomposition(cov_matrix):
    eig_vals, eig_vecs = np.linalg.eigh(cov_matrix)
    sorted_indices = np.argsort(eig_vals)[::-1]
    return eig_vals[sorted_indices], eig_vecs[:, sorted_indices]

def time_pca(num_threads, X_std, num_runs=5):
    total_time = 0
    for i in range(num_runs):
        start_time = time.perf_counter_ns()
        cov_matrix = compute_covariance(X_std)
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            eig_vals, eig_vecs = executor.submit(eigen_decomposition, cov_matrix).result()
        end_time = time.perf_counter_ns()
        run_time = end_time - start_time
        total_time += run_time
        logging.info(f"Time taken for {num_threads} thread(s) in run {i+1}: {run_time / 1e6:.3f} ms")
    return total_time / num_runs  # Return average time in nanoseconds

def plot_execution_times(threads, times, title, filename):
    plt.figure(figsize=(10, 6))
    plt.plot(threads, times, marker='o', linestyle='-', color='b')
    plt.title(title)
    plt.xlabel('Number of Threads')
    plt.ylabel('Execution Time (nanoseconds)')
    plt.xticks(threads)
    plt.grid(True)
    plt.savefig(filename)
    plt.close()

def calculate_speedup(single_thread_time, execution_times):
    return [(threads, single_thread_time / exec_time) for threads, exec_time in execution_times]

def plot_speedup(threads, speedup_values, title, filename):
    plt.figure(figsize=(10, 6))
    plt.plot(threads, speedup_values, marker='o', linestyle='-', color='g')
    plt.title(title)
    plt.xlabel('Number of Threads')
    plt.ylabel('Speedup')
    plt.xticks(threads)
    plt.grid(True)
    plt.savefig(filename)
    plt.close()

if __name__ == "__main__":
    try:
        # Generate and preprocess data
        X = generate_synthetic_data()
        scaler = StandardScaler()
        X_std = scaler.fit_transform(X)

        # Test with a wide range of thread counts
        thread_counts_extended = list(range(1, 65, 4))
        thread_execution_times_extended = [(threads, time_pca(threads, X_std)) for threads in thread_counts_extended]

        # Plot extended execution times
        threads_ext, times_ext = zip(*thread_execution_times_extended)
        plot_execution_times(
            threads_ext,
            times_ext,
            "Extended Execution Time vs Number of Threads",
            "Extended_Execution_Time_vs_Number_of_Threads.png"
        )

        # Calculate speedup
        single_thread_time = thread_execution_times_extended[0][1]
        speedup = calculate_speedup(single_thread_time, thread_execution_times_extended)

        # Plot speedup
        threads_ext, speedup_values = zip(*speedup)
        plot_speedup(
            threads_ext,
            speedup_values,
            "Speedup vs Number of Threads",
            "Speedup_vs_Number_of_Threads.png"
        )

        # Log results for extended thread counts and speedup
        print("Extended Execution Times (nanoseconds) and Speedup:")
        for (thread, time), (_, speedup_val) in zip(thread_execution_times_extended, speedup):
            print(f"{thread:2d} thread(s): {time:.2f} ns, Speedup: {speedup_val:.2f}x")

        logging.info("Analysis complete.")

    except Exception as e:
        logging.error(f"An error occurred: {e}")
