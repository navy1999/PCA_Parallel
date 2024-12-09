
# pca_parallel.py

import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import time
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

def readAndProcessData():
    df = pd.read_csv('p4dataset2020.txt', header=None, delim_whitespace=True)
    gender = df[1]
    population = df[2]
    print(np.unique(population))
    
    df.drop(df.columns[[0, 1, 2]], axis=1, inplace=True)
    modes = np.array(df.mode().values[0,:])
    return df, modes, population, gender

def convertDfToMatrix(df, modes):
    raw_np = df.to_numpy()
    binarized = np.where(raw_np!=modes, 1, 0)
    return binarized

def compute_covariance(X):
    return np.dot(X.T, X) / (X.shape[0] - 1)

def eigen_decomposition(cov_matrix):
    eig_vals, eig_vecs = np.linalg.eigh(cov_matrix)
    sorted_indices = np.argsort(eig_vals)[::-1]
    return eig_vals[sorted_indices], eig_vecs[:, sorted_indices]

def time_pca(num_threads, X_std, num_runs=1):
    total_time = 0
    for i in range(num_runs):
        start_time = time.perf_counter()
        cov_matrix = compute_covariance(X_std)
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            eig_vals, eig_vecs = executor.submit(eigen_decomposition, cov_matrix).result()
        end_time = time.perf_counter()
        total_time += (end_time - start_time)
        print("Time taken for thread " + num_threads +" in "+ i + "th run : " + total_time)
    return total_time / num_runs

if __name__ == "__main__":
    df, modes, population, gender = readAndProcessData()
    X = convertDfToMatrix(df, modes)

    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)

    thread_execution_times = [(num_threads, time_pca(num_threads, X_std)) for num_threads in range(8, 65,8)]

    single_thread_time = thread_execution_times[0][1]
    speedup = [(num_threads, single_thread_time / exec_time if exec_time > 0 else 0) for num_threads, exec_time in thread_execution_times]

    threads, times = zip(*thread_execution_times)
    _, speedups = zip(*speedup)

    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    plt.plot(threads, times, marker='o', linestyle='-', color='r')
    plt.title('Execution Time vs Number of Threads for PCA')
    plt.xlabel('Number of Threads')
    plt.ylabel('Execution Time (nanoseconds)')
    plt.xticks(range(0, 65, 8))
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(threads, speedups, marker='o', linestyle='-', color='b')
    plt.title('Speedup vs Number of Threads for PCA')
    plt.xlabel('Number of Threads')
    plt.ylabel('Speedup')
    plt.xticks(range(0, 65, 8))
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('pca_performance.png')
    plt.close()

    print("Execution times (nanoseconds):")
    for thread, time in thread_execution_times:
        print(f"{thread} thread(s): {time:.2f} ns")
