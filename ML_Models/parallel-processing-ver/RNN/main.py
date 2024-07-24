"""
Main structure that calls functions and executes the training and validation loops.

Modules:
    training_config: Contains global parameters and model instantiation.
    model: Defines the model structure.
    get_data: Defines the data loader structure.

Note:
    Added a workaround to suppress a specific error message related to the environment variable 'KMP_DUPLICATE_LIB_OK'.

Imports:
    os: Standard library for operating system interactions.
    argparse: Standard library for parsing command-line arguments.
    torch: Main package for PyTorch.
    torch.nn: Neural network module from PyTorch.
    torch.distributed: Distributed training support in PyTorch.
    torch.utils.data: Data loading utilities from PyTorch.
    torch.utils.tensorboard: TensorBoard support for PyTorch.
    time: Standard library for time-related functions.

Author:
    [Kanak Parmar]

Revision Authors:
    [Owen A. Johnson] 

Last updated:
    [2024-07-23]
"""

# --- Preamble ---
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # Suppresses a specific error message related to the environment variable 'KMP_DUPLICATE_LIB_OK'
import argparse
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter
import time
import matplotlib.pyplot as plt

def initialize_distributed_backend(backend='nccl'):  # must be called before any other function in this file
    """
    Initializes the distributed backend.

    Args:
        backend (str): The backend to use ('nccl', 'gloo', 'mpi').

    Returns:
        int: The local rank of the process.
    """
    dist.init_process_group(backend=backend)
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    return local_rank

def parse_args():
    """
    Parses command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description='Distributed Training Script')
    parser.add_argument('--verbose', action='store_true', help='Increase output verbosity')
    return parser.parse_args()

def write_python_file(filename, target_dir):
    """
    Reads a Python file and writes its content to a new file in the target directory.

    Args:
        filename (str): The path of the file to read.
        target_dir (str): The directory to save the new file.
    """
    with open(filename) as f:
        data = f.read()
    
    with open(os.path.join(target_dir, "training_config.txt"), mode="w") as f:
        f.write(data)

args = parse_args()
local_rank = initialize_distributed_backend()

if args.verbose and local_rank == 0:  # Print information only on the main process
    print(f"Local Rank: {local_rank}")
    print(f"Using Backend: {dist.get_backend()}")
    print(f"World Size: {dist.get_world_size()}")
    print(f"Rank: {dist.get_rank()}")

from training_config import (
    model_train, optimizer, device,
    batch_size_defined, num_epochs, data_record_interval,
    model_save_dir, config_file_path,
    shuffle_train, shuffle_test,
    root_data_dir, train_EID_file, test_EID_file, data_exclude_list
)
from get_data import CustomDataset, collate_custom

labels_str = ['carbonate', 'chloride', 'oxidized organic carbon', 'oxychlorine', 'sulfate', 'sulfide', 'nitrate', 'iron_oxide', 'phyllosilicate', 'silicate']
# Move model to the correct device before wrapping it with DistributedDataParallel
model_train = model_train.to(local_rank)

if torch.cuda.device_count() > 1:
    model_train = nn.parallel.DistributedDataParallel(model_train, device_ids=[local_rank], output_device=local_rank)

# Define run via time/day of initiation
timestr = time.strftime("%Y%m%d-%H%M%S")
save_path = os.path.join(model_save_dir, timestr)
model_save_path = os.path.join(save_path, 'model.pt')

# Ensure the parent directory exists
os.makedirs(save_path, exist_ok=True)

# Instantiate the tensorboard logger
writer = SummaryWriter(save_path)

# Save everything in the training_config.py as a txt file in the model folder
write_python_file(config_file_path, save_path)

# --- Data Load --------------------------------------------------------------------------------------------
# Load training data
train_dataset = CustomDataset(root_data_dir, train_EID_file, data_exclude_list)
train_sampler = DistributedSampler(train_dataset)
train_data = DataLoader(train_dataset, batch_size=batch_size_defined, shuffle=False, sampler=train_sampler, collate_fn=collate_custom)

# Load test data
test_dataset = CustomDataset(root_data_dir, test_EID_file, data_exclude_list)
test_sampler = DistributedSampler(test_dataset)
test_data = DataLoader(test_dataset, batch_size=batch_size_defined, shuffle=False, sampler=test_sampler, collate_fn=collate_custom)

# ---------------------------------------------------------------------------------------------------------

best_accuracy = 0  # Keep track of the best validation accuracy
torch.manual_seed(42)  # Set seed for reproducibility
batch_global_train = 0  # Keep track of the global batch index for training
batch_global_val = 0  # Keep track of the global batch index for validation

# Use BCEWithLogitsLoss for multi-label classification
criterion = nn.BCEWithLogitsLoss()

# Initialize variables to store per-label accuracies
train_label_accuracies = []
val_label_accuracies = []

for epoch in range(num_epochs):
    """
    Training loop
    """
    if args.verbose and local_rank == 0:  # Print information only on the main process
        print(f'Training model on Epoch {epoch}')
    train_sampler.set_epoch(epoch)  # Ensure each process gets a different subset of data each epoch
    running_loss_train = 0
    accuracy_batch_train = []
    label_accuracies_train = {label: [] for label in range(10)}  # Assuming 10 labels
    for i, data_train in enumerate(train_data):
        # Debug: Print rank and batch index
        if args.verbose and local_rank == 0:  # Print information only on the main process
            print(f"Process {local_rank}, Epoch {epoch}, Batch {i}")

        # Extract input data and corresponding labels
        inputs_train, label_train = data_train
        inputs_train, label_train = inputs_train.to(device), label_train.to(device)
        
        # Convert label_train to float
        label_train = label_train.float()
        
        # Zero your gradients for every batch
        optimizer.zero_grad()

        # Get the prediction on batch
        pred_train = model_train(inputs_train)

        # Compute loss and update
        loss_train = criterion(pred_train, label_train)
        running_loss_train += loss_train.item()
        
        # Do the gradient backprop and adjust weights
        loss_train.backward()
        optimizer.step()
        
        # Compute accuracy AFTER optimizer update
        if i % data_record_interval == 0:
            # Compute batch accuracy for each label
            pred_train_sigmoid = torch.sigmoid(pred_train)
            for label in range(10):
                batch_accuracy_label = ((pred_train_sigmoid[:, label] > 0.5) == label_train[:, label]).float().mean()
                label_accuracies_train[label].append(batch_accuracy_label.item())

            if args.verbose and local_rank == 0:  # Print information only on the main process
                print(f'Process {local_rank}, Epoch: {epoch}, Batch {i} of {len(train_data)}, Train Loss: {loss_train.item()}')
                for label in range(10):
                    print(f'Label {label} Train Accuracy: {label_accuracies_train[label][-1]:.4f}')
            
            # Write to tensorboard, print epoch loss to console
            batch_global_train += data_record_interval
            writer.add_scalar('Batch Train Loss', loss_train, batch_global_train)

        # Delete pred and losses to reduce memory consumption
        del loss_train, pred_train, inputs_train, label_train

    # Calculate mean accuracy for each label
    mean_label_accuracies_train = {label: sum(accs) / len(accs) for label, accs in label_accuracies_train.items()}
    train_label_accuracies.append(mean_label_accuracies_train)

    """
    Validation loop
    """
    if args.verbose and local_rank == 0:  # Print information only on the main process
        print(f'Validating model on Epoch {epoch}')
    running_loss_val = 0
    accuracy_batch_val = []
    label_accuracies_val = {label: [] for label in range(10)}  # Assuming 10 labels
    with torch.no_grad():
        for j, data_val in enumerate(test_data):
            # Debug: Print rank and batch index
            if args.verbose and local_rank == 0:  # Print information only on the main process
                print(f"Process {local_rank}, Validation, Epoch {epoch}, Batch {j}")

            # Extract input data and corresponding labels
            inputs_val, label_val = data_val
            inputs_val, label_val = inputs_val.to(device), label_val.to(device)

            # Convert label_val to float
            label_val = label_val.float()

            # Get the prediction on batch
            pred_val = model_train(inputs_val)

            # Compute loss
            loss_val = criterion(pred_val, label_val)
            running_loss_val += loss_val.item()
            
            if j % data_record_interval == 0:
                # Compute batch accuracy for each label
                pred_val_sigmoid = torch.sigmoid(pred_val)
                for label in range(10):
                    batch_accuracy_label = ((pred_val_sigmoid[:, label] > 0.5) == label_val[:, label]).float().mean()
                    label_accuracies_val[label].append(batch_accuracy_label.item())

                if args.verbose and local_rank == 0:  # Print information only on the main process
                    print(f'Process {local_rank}, Epoch: {epoch}, Batch {j} of {len(test_data)}, Val Loss: {loss_val.item()}')
                    for label in range(10):
                        print(f'{labels_str[label]} Val Accuracy: {label_accuracies_val[label][-1]:.4f}')

                # Write to tensorboard, print epoch loss to console
                batch_global_val += data_record_interval
                writer.add_scalar('Batch Val Loss', loss_val, batch_global_val)

            # Delete pred and losses to reduce memory consumption
            del loss_val, pred_val, inputs_val, label_val

        # Calculate mean accuracy for each label
        mean_label_accuracies_val = {label: sum(accs) / len(accs) for label, accs in label_accuracies_val.items()}
        val_label_accuracies.append(mean_label_accuracies_val)

        # Calculate the overall average accuracy across all labels
        avg_accuracy_val = sum(mean_label_accuracies_val.values()) / len(mean_label_accuracies_val)

        # If new best validation accuracy, save model
        if avg_accuracy_val > best_accuracy and local_rank == 0:  # Save model only on the main process
            torch.save(model_train.state_dict(), model_save_path)
            best_accuracy = avg_accuracy_val

# Close tensorboard logger instance
writer.close()

# Save the label accuracies for plotting later
torch.save({
    'train_label_accuracies': train_label_accuracies,
    'val_label_accuracies': val_label_accuracies
}, os.path.join(save_path, 'label_accuracies.pt'))

# Plot the accuracies for each label
train_label_accuracies = torch.load(os.path.join(save_path, 'label_accuracies.pt'))['train_label_accuracies']
val_label_accuracies = torch.load(os.path.join(save_path, 'label_accuracies.pt'))['val_label_accuracies']

for label in range(10):
    plt.plot([epoch[label] for epoch in train_label_accuracies], label=f'Train Label {label}')
    plt.plot([epoch[label] for epoch in val_label_accuracies], label=f'Val Label {label}')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title(f'Accuracy for Label {label}')
    plt.legend()
    plt.savefig(str(label) + '_accuracy.png')