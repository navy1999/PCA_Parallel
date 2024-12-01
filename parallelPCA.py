import numpy as np
import dask.array as da
from dask_ml.decomposition import PCA as DaskPCA
from sklearn.preprocessing import StandardScaler
import time
import matplotlib.pyplot as plt
from dask.distributed import Client, LocalCluster
from contextlib import contextmanager
import psutil

class OptimizedParallelPCA:
    def __init__(self, n_components=2):
        self.n_components = n_components
        
    @staticmethod
    def get_memory_limit():
        # Use 75% of available system memory
        return int(psutil.virtual_memory().available * 0.75)
    
    @contextmanager
    def cluster_context(self, n_workers):
        memory_per_worker = self.get_memory_limit() // n_workers
        
        cluster = LocalCluster(
            n_workers=n_workers,
            threads_per_worker=1,
            memory_limit=f"{memory_per_worker}B",
            processes=True,
            dashboard_address=None,
            silence_logs=True
        )
        client = Client(cluster)
        try:
            yield client
        finally:
            client.close()
            cluster.close()

    def fit_transform(self, X, n_workers=2):
        with self.cluster_context(n_workers) as client:
            # Optimize chunk size based on memory
            chunk_size = max(len(X) // (n_workers * 2), 32)
            X_dask = da.from_array(X, chunks=(chunk_size, -1))
            
            # Configure PCA
            pca = DaskPCA(
                n_components=self.n_components,
                iterated_power=3,
                random_state=42
            )
            
            # Compute result synchronously
            result = pca.fit_transform(X_dask)
            return result.compute()

def generate_sample_data(n_samples=10000, n_features=100):
    """Generate large sample dataset"""
    return np.random.randn(n_samples, n_features)

def benchmark_pca(X, max_workers=8):
    worker_counts = list(range(1, max_workers + 1))
    times = []
    
    for n_workers in worker_counts:
        # Run multiple times and take minimum
        times_per_worker = []
        for _ in range(3):
            pca = OptimizedParallelPCA(n_components=2)
            try:
                start_time = time.time()
                _ = pca.fit_transform(X, n_workers=n_workers)
                execution_time = time.time() - start_time
                times_per_worker.append(execution_time)
                time.sleep(1)  # Cool-down period
            except Exception as e:
                print(f"Error with {n_workers} workers:", str(e))
        
        # Take best time
        times.append(min(times_per_worker) if times_per_worker else float('inf'))
    
    return worker_counts, times

if __name__ == "__main__":
    # Generate sample data
    X = generate_sample_data()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    print(f"Dataset shape: {X_scaled.shape}")
    
    # Run benchmark
    worker_counts, parallel_times = benchmark_pca(X_scaled)
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(worker_counts, parallel_times, 'o-', linewidth=2)
    plt.xlabel('Number of Workers')
    plt.ylabel('Time (seconds)')
    plt.title('Parallelized PCA Performance vs Number of Workers')
    plt.grid(True)
    plt.savefig('pca_performance.png')
    plt.close()
    
    # Print optimal configuration
    optimal_workers = worker_counts[np.argmin(parallel_times)]
    print(f"Optimal number of workers: {optimal_workers}")
    print(f"Best execution time: {min(parallel_times):.2f} seconds")
