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
    def get_optimal_chunk_size(X, n_workers):
        # Optimize chunk size based on data dimensions and memory
        n_samples = X.shape[0]
        return max(n_samples // (n_workers * 4), 64)  # Increased minimum chunk size
    
    @contextmanager
    def cluster_context(self, n_workers):
        # Calculate memory limit based on available system memory
        memory_limit = int(psutil.virtual_memory().available * 0.8 / n_workers)
        
        cluster = LocalCluster(
            n_workers=n_workers,
            threads_per_worker=1,  # Single thread per worker to avoid GIL issues
            memory_limit=f"{memory_limit}B",
            processes=True,
            dashboard_address=None,
            silence_logs=False,  # Enable logs for debugging
            lifetime='1h'  # Set cluster lifetime
        )
        client = Client(cluster)
        try:
            yield client
        finally:
            client.close()
            cluster.close()

    def fit_transform(self, X, n_workers=2):
        with self.cluster_context(n_workers) as client:
            # Optimize chunking
            chunk_size = self.get_optimal_chunk_size(X, n_workers)
            X_dask = da.from_array(X, chunks=(chunk_size, -1))
            
            # Pre-scatter data to workers
            X_dask = client.persist(X_dask)
            
            # Configure PCA with reduced iterations
            pca = DaskPCA(
                n_components=self.n_components,
                iterated_power=2,  # Reduced iterations
                random_state=42
            )
            
            # Compute result synchronously
            result = pca.fit_transform(X_dask)
            return result.compute(scheduler='threads')  # Use threaded scheduler for final computation

def benchmark_pca(X, max_workers=64):
    worker_counts = list(range(1, max_workers + 1))
    times = []
    
    for n_workers in worker_counts:
        # Run multiple times and take minimum
        times_per_worker = []
        for _ in range(2):  # Reduced number of runs
            pca = OptimizedParallelPCA(n_components=2)
            try:
                start_time = time.time()
                _ = pca.fit_transform(X, n_workers=n_workers)
                execution_time = time.time() - start_time
                times_per_worker.append(execution_time)
            except Exception as e:
                print(f"Error with {n_workers} workers:", str(e))
            time.sleep(2)  # Increased cool-down period
        
        # Take best time
        times.append(min(times_per_worker) if times_per_worker else float('inf'))
        print(f"Completed benchmark for {n_workers} workers")
    
    return worker_counts, times

if __name__ == '__main__':
    # Generate larger sample data
    X = np.random.randn(100000, 1000)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    print(f"Dataset shape: {X_scaled.shape}")
    
    # Run benchmark
    worker_counts, parallel_times = benchmark_pca(X_scaled)
    
    # Plot results
    plt.figure(figsize=(12, 8))
    plt.plot(worker_counts, parallel_times, 'o-', linewidth=2, markersize=4)
    plt.xlabel('Number of Workers')
    plt.ylabel('Time (seconds)')
    plt.title('Parallelized PCA Performance vs Number of Workers (64-core cluster)')
    plt.grid(True, alpha=0.3)
    plt.savefig('pca_performance.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Print optimal configuration
    optimal_workers = worker_counts[np.argmin(parallel_times)]
    print(f"Optimal number of workers: {optimal_workers}")
    print(f"Best execution time: {min(parallel_times):.2f} seconds")
