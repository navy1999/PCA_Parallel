import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import time
import matplotlib.pyplot as plt
import numpy as np

# Define the neural network with 1x1 convolutions and reduced fully-connected layers
class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNN, self).__init__()
        
        # Using 1x1 convolutions to reduce communication
        self.conv1 = nn.Conv2d(1, 32, kernel_size=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=1)
        
        # Compute the flattened size after convolutional layers (height and width remain the same as input_size)
        self.flattened_size = 64 * input_size * input_size  # After 2 1x1 convolutions, the size stays the same

        # Fully connected layer, reduced size
        self.fc1 = nn.Linear(self.flattened_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Function to parallelize the model across multiple devices (GPUs)
def parallelize_model(model, device_ids):
    # Distribute the model across GPUs
    if len(device_ids) > 1:
        model = nn.DataParallel(model, device_ids=device_ids)
    return model

# Function to generate a synthetic dataset
def generate_dataset(num_samples, num_features):
    X = torch.randn(num_samples, 1, num_features, num_features)
    y = torch.randint(0, 10, (num_samples,))
    return X, y

# Benchmark function to test speedup with varying dataset sizes and threads
def benchmark_parallelization(num_samples_list, num_features_list, num_threads_list):
    speedups = []
    for num_samples in num_samples_list:
        for num_features in num_features_list:
            # Generate synthetic dataset
            X, y = generate_dataset(num_samples, num_features)
            dataset = TensorDataset(X, y)
            dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
            
            for num_threads in num_threads_list:
                # Set the number of threads for parallelism
                torch.set_num_threads(num_threads)
                
                # Create and parallelize the model
                model = SimpleNN(num_features, 128, 10)
                device_ids = list(range(num_threads)) if num_threads > 1 else [0]
                model = parallelize_model(model, device_ids)
                
                # Move model to device
                model = model.cuda() if torch.cuda.is_available() else model
                
                # Train the model
                optimizer = optim.Adam(model.parameters(), lr=0.001)
                criterion = nn.CrossEntropyLoss()
                
                start_time = time.time()
                model.train()
                for epoch in range(2):  # 2 epochs for benchmark
                    for inputs, labels in dataloader:
                        inputs, labels = inputs.cuda() if torch.cuda.is_available() else inputs, labels.cuda() if torch.cuda.is_available() else labels
                        optimizer.zero_grad()
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                        loss.backward()
                        optimizer.step()
                
                elapsed_time = time.time() - start_time
                speedups.append(elapsed_time)
                print(f"Num samples: {num_samples}, Features: {num_features}, Threads: {num_threads}, Time: {elapsed_time:.4f}s")
    
    return speedups


# Define the list of parameters for the benchmark
num_samples_list = [1000, 5000, 10000]
num_features_list = [10, 100, 1000]
num_threads_list = [1, 2, 4, 8, 12, 16 ,24, 32 ,40 ,56 ,64]

# Run the benchmark and plot the speedup
speedups = benchmark_parallelization(num_samples_list, num_features_list, num_threads_list)