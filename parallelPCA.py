import numpy as np
from sklearn.datasets import make_classification, make_blobs, make_multilabel_classification
from sklearn.decomposition import PCA
import time
import matplotlib.pyplot as plt
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Generate datasets with different types and sizes
def generate_datasets():
    logging.info("Generating datasets with different data types and sizes...")

    # Binary classification dataset (small)
    binary_data_small, _ = make_classification(n_samples=500, n_features=10, n_classes=2, random_state=42)

    # Continuous data (medium)
    continuous_data_medium, _ = make_blobs(n_samples=1000, n_features=50, centers=3, random_state=42)

    # Multilabel classification dataset (large)
    categorical_data_large, _ = make_multilabel_classification(n_samples=5000, n_features=100, n_classes=5, random_state=42)

    logging.info("Datasets generated successfully.")
    return binary_data_small, continuous_data_medium, categorical_data_large

# Run PCA and measure execution time
def run_pca_and_measure_time(data):
    logging.info(f"Running PCA on dataset with shape {data.shape}...")
    start_time = time.perf_counter()
    pca = PCA()
    pca.fit(data)
    end_time = time.perf_counter()
    execution_time = end_time - start_time
    logging.info(f"PCA completed in {execution_time:.4f} seconds.")
    return execution_time

# Plot execution times for different datasets
def plot_execution_times(dataset_labels, times):
    plt.figure(figsize=(10, 6))
    plt.bar(dataset_labels, times, color=['blue', 'orange', 'green'])
    plt.title("PCA Execution Time for Different Dataset Types and Sizes")
    plt.xlabel("Dataset Type and Size")
    plt.ylabel("Execution Time (seconds)")
    plt.grid(axis='y')
    plt.savefig("pca_execution_times.png")
    plt.show()

if __name__ == "__main__":
    try:
        # Generate datasets
        binary_data_small, continuous_data_medium, categorical_data_large = generate_datasets()

        # Measure PCA execution times
        small_pca_time = run_pca_and_measure_time(binary_data_small)
        medium_pca_time = run_pca_and_measure_time(continuous_data_medium)
        large_pca_time = run_pca_and_measure_time(categorical_data_large)

        # Dataset labels and times for plotting
        dataset_labels = ["Binary (500x10)", "Continuous (1000x50)", "Categorical (5000x100)"]
        times = [small_pca_time, medium_pca_time, large_pca_time]

        # Plot the results
        plot_execution_times(dataset_labels, times)

        # Log the results
        logging.info("Execution Times:")
        for label, time in zip(dataset_labels, times):
            logging.info(f"{label}: {time:.4f} seconds")

    except Exception as e:
        logging.error(f"An error occurred: {e}")
