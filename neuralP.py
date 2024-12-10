import numpy as np
import concurrent.futures
import matplotlib.pyplot as plt
import time

class ParallelNeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, num_threads=1):
        """
        Initialize a parallel neural network with spatial partitioning
        
        Args:
            input_size (int): Number of input features
            hidden_size (int): Number of neurons in hidden layer
            output_size (int): Number of output neurons
            num_threads (int): Number of threads for parallel computation
        """
        self.num_threads = num_threads
        
        # Spatial partitioning of network parameters
        partition_size = hidden_size // num_threads
        
        # Initialize weights with spatial partitioning
        self.weights1_partitions = [
            np.random.randn(input_size, partition_size) 
            for _ in range(num_threads)
        ]
        
        # 1x1 convolution-like weight reduction for hidden layer
        self.weights2 = np.random.randn(partition_size * num_threads, output_size)
        
        # Bias terms
        self.bias1 = np.zeros((1, hidden_size))
        self.bias2 = np.zeros((1, output_size))
    
    def sigmoid(self, x):
        """Sigmoid activation function"""
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, x):
        """Derivative of sigmoid"""
        return x * (1 - x)
    
    def process_partition(self, weights, X):
        """
        Process a partition of the neural network
        
        Args:
            weights (np.ndarray): Weights for this partition
            X (np.ndarray): Input data
        
        Returns:
            np.ndarray: Processed partition
        """
        return self.sigmoid(np.dot(X, weights))
    
    def parallel_forward_propagation(self, X):
        """
        Parallel forward propagation using ThreadPoolExecutor
        
        Args:
            X (np.ndarray): Input data
        
        Returns:
            tuple: Hidden layer and output layer activations
        """
        # Use ThreadPoolExecutor for parallel computation
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            # Submit tasks for each weight partition
            futures = [
                executor.submit(self.process_partition, w, X) 
                for w in self.weights1_partitions
            ]
            
            # Wait for and collect results
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        # Combine results from partitions
        hidden_layer = np.hstack(results)
        output_layer = self.sigmoid(np.dot(hidden_layer, self.weights2) + self.bias2)
        
        return hidden_layer, output_layer
    
    def train(self, X, y, epochs=1000, learning_rate=0.01):
        """
        Training method with parallel forward and sequential backpropagation
        
        Args:
            X (np.ndarray): Input training data
            y (np.ndarray): Target values
            epochs (int): Number of training iterations
            learning_rate (float): Learning rate for gradient descent
        """
        for _ in range(epochs):
            # Parallel forward propagation
            hidden_layer, output_layer = self.parallel_forward_propagation(X)
            
            # Backpropagation (kept sequential to demonstrate trade-offs)
            output_error = y - output_layer
            output_delta = output_error * self.sigmoid_derivative(output_layer)
            
            # Update weights
            hidden_error = np.dot(output_delta, self.weights2.T)
            hidden_delta = hidden_error * self.sigmoid_derivative(hidden_layer)
            
            # Update weights using gradient descent
            self.weights2 += learning_rate * np.dot(hidden_layer.T, output_delta)
            
            # Update partitioned weights
            for i in range(self.num_threads):
                partition_start = i * (hidden_layer.shape[1] // self.num_threads)
                partition_end = (i+1) * (hidden_layer.shape[1] // self.num_threads)
                
                self.weights1_partitions[i] += learning_rate * np.dot(
                    X.T, 
                    hidden_delta[:, partition_start:partition_end]
                )

def benchmark_parallel_network():
    """
    Benchmark parallel neural network performance
    
    This function tests the speedup with increasing:
    1. Dataset size
    2. Number of features
    3. Number of threads
    """
    # Reduce dataset sizes for faster testing
    dataset_sizes = [1000, 5000, 10000]
    feature_sizes = [10, 50, 100]
    thread_counts = [1, 2, 4, 6, 8, 10, 12]
    
    results = {
        'dataset_size': [],
        'features': [],
        'threads': [],
        'sequential_time': [],
        'parallel_time': [],
        'speedup': []
    }
    
    for data_size in dataset_sizes:
        for features in feature_sizes:
            X = np.random.rand(data_size, features)
            y = np.random.rand(data_size, 1)
            
            for num_threads in thread_counts:
                # Sequential timing
                start = time.time()
                net_sequential = ParallelNeuralNetwork(
                    input_size=features, 
                    hidden_size=64, 
                    output_size=1, 
                    num_threads=1
                )
                net_sequential.train(X, y, epochs=10)
                sequential_time = time.time() - start
                
                # Parallel timing
                start = time.time()
                net_parallel = ParallelNeuralNetwork(
                    input_size=features, 
                    hidden_size=64, 
                    output_size=1, 
                    num_threads=num_threads
                )
                net_parallel.train(X, y, epochs=10)
                parallel_time = time.time() - start
                
                speedup = sequential_time / parallel_time if parallel_time > 0 else 0
                
                results['dataset_size'].append(data_size)
                results['features'].append(features)
                results['threads'].append(num_threads)
                results['sequential_time'].append(sequential_time)
                results['parallel_time'].append(parallel_time)
                results['speedup'].append(speedup)
                
                # Print the speedup values for each configuration
                print(f"Dataset size: {data_size}, Features: {features}, Threads: {num_threads} => Speedup: {speedup:.2f}")
    
    # Plotting results
    plt.figure(figsize=(15, 5))
    
    # Speedup by threads
    plt.subplot(131)
    threads_speedup = [
        results['speedup'][i] 
        for i in range(len(results['speedup'])) 
        if results['dataset_size'][i] == dataset_sizes[-1] and 
           results['features'][i] == feature_sizes[-1]
    ]
    plt.plot(thread_counts, threads_speedup, marker='o')
    plt.title('Speedup vs Threads')
    plt.xlabel('Number of Threads')
    plt.ylabel('Speedup')
    
    # Speedup by dataset size
    plt.subplot(132)
    dataset_speedup = [
        results['speedup'][i] 
        for i in range(len(results['speedup'])) 
        if results['threads'][i] == thread_counts[-1] and 
           results['features'][i] == feature_sizes[-1]
    ]
    plt.plot(dataset_sizes, dataset_speedup, marker='o')
    plt.title('Speedup vs Dataset Size')
    plt.xlabel('Dataset Size')
    plt.ylabel('Speedup')
    
    # Speedup by features
    plt.subplot(133)
    features_speedup = [
        results['speedup'][i] 
        for i in range(len(results['speedup'])) 
        if results['threads'][i] == thread_counts[-1] and 
           results['dataset_size'][i] == dataset_sizes[-1]
    ]
    plt.plot(feature_sizes, features_speedup, marker='o')
    plt.title('Speedup vs Features')
    plt.xlabel('Number of Features')
    plt.ylabel('Speedup')
    
    plt.tight_layout()
    
    # Save the figure instead of showing it
    plt.savefig('benchmark_results.png', dpi=300)
    print("Benchmark results saved to 'benchmark_results.png'.")
    
    return results

def main():
    benchmark_results = benchmark_parallel_network()
    print("Benchmark completed. Check the saved plot for performance analysis.")

if __name__ == '__main__':
    main()