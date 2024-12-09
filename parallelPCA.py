import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import time
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def readAndProcessData():
    try:
        df = pd.read_csv('p4dataset2020.txt', header=None, sep='\s+')
        gender = df[1]
        population = df[2]
        logging.info(f"Unique populations: {np.unique(population)}")
        
        df.drop(df.columns[[0, 1, 2]], axis=1, inplace=True)
        modes = np.array(df.mode().values[0,:])
        return df, modes, population, gender
    except Exception as e:
        logging.error(f"Error in readAndProcessData: {e}")
        raise

def convertDfToMatrix(df, modes):
    raw_np = df.to_numpy()
    return np.where(raw_np!=modes, 1, 0)

def compute_covariance(X):
    return np.dot(X.T, X) / (X.shape[0] - 1)

def eigen_decomposition(cov_matrix):
    eig_vals, eig_vecs = np.linalg.eigh(cov_matrix)
    sorted_indices = np.argsort(eig_vals)[::-1]
    return eig_vals[sorted_indices], eig_vecs[:, sorted_indices]

def time_pca(num_threads, X_std, num_runs=5):
    total_time = 0
    for _ in range(num_runs):
        start_time = time.perf_counter_ns()
        cov_matrix = compute_covariance(X_std)
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            eig_vals, eig_vecs = executor.submit(eigen_decomposition, cov_matrix).result()
        end_time = time.perf_counter_ns()
        total_time += (end_time - start_time)
    return total_time / num_runs

def plot_results(thread_execution_times, speedup):
    threads, times = zip(*thread_execution_times)
    _, speedups = zip(*speedup)

    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    plt.plot(threads, times, marker='o', linestyle='-', color='r')
    plt.title('Execution Time vs Number of Threads for PCA')
    plt.xlabel('Number of Threads')
    plt.ylabel('Execution Time (nanoseconds)')
    plt.xticks(threads)
    plt.yscale('log')
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(threads, speedups, marker='o', linestyle='-', color='b')
    plt.title('Speedup vs Number of Threads for PCA')
    plt.xlabel('Number of Threads')
    plt.ylabel('Speedup')
    plt.xticks(threads)
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('pca_performance.png')
    plt.close()

if __name__ == "__main__":
    try:
        logging.info("Starting data processing...")
        df, modes, population, gender = readAndProcessData()
        logging.info("Data processed. Converting to matrix...")
        X = convertDfToMatrix(df, modes)
        logging.info("Matrix conversion complete. Standardizing...")
        scaler = StandardScaler()
        X_std = scaler.fit_transform(X)
        logging.info("Standardization complete. Starting PCA timing...")

        single_thread_time = time_pca(1, X_std)
        thread_execution_times = [(1, single_thread_time)] + [(num_threads, time_pca(num_threads, X_std)) for num_threads in range(2, 65, 8)]

        speedup = [(num_threads, single_thread_time / exec_time) for num_threads, exec_time in thread_execution_times]

        plot_results(thread_execution_times, speedup)

        print("Execution times (nanoseconds):")
        for thread, time in thread_execution_times:
            print(f"{thread:2d} thread(s): {time:.2f} ns")

        logging.info("PCA analysis complete.")
    except Exception as e:
        logging.error(f"An error occurred: {e}")
