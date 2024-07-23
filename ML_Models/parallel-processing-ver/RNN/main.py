"""
Main structure that calls functions and exevcutes the training and validation loops.

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
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE" # Suppresses a specific error message related to the environment variable 'KMP_DUPLICATE_LIB_OK'
import argparse
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter
import time

def initialize_distributed_backend(backend='nccl'): # must be called before any other function in this file
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
    with open(filename) as f:
        data = f.read()
        f.close()

    with open(os.path.join(target_dir,"traninig_config.txt"), mode="w") as f:
        f.write(data)
        f.close()


args = parse_args()

local_rank = initialize_distributed_backend()

if args.verbose:
    print(f"Local Rank: {local_rank}")
    print(f"Using Backend: {dist.get_backend()}")
    print(f"World Size: {dist.get_world_size()}")
    print(f"Rank: {dist.get_rank()}")

from training_config import (
    model_train, optimizer, criterion, device,
    batch_size_defined, num_epochs, data_record_interval,
    model_save_dir, config_file_path,
    shuffle_train, shuffle_test,
    root_data_dir, train_EID_file, test_EID_file, data_exclude_list
)
from get_data import CustomDataset, collate_custom


if torch.cuda.device_count() > 1:
    model_train = nn.parallel.DistributedDataParallel(model_train, device_ids=[local_rank]) # Ensure the model is wrapped with DistributedDataParallel after moving it to the correct device

#----------------------------------------------------------------------------------------------------------
# Define run via time/day of initiation
timestr = time.strftime("%Y%m%d-%H%M%S")
save_path = os.path.join(model_save_dir, timestr)
model_save_path = os.path.join(save_path, 'model.pt')

# Ensure the parent directory exists
os.makedirs(save_path, exist_ok=True)

if not os.path.exists(save_path):
    print('Making save folder...')
    os.mkdir(save_path)

# Instantiate the tensorboard logger
writer = SummaryWriter(save_path)

# Save everything in the training_config.py as a txt file in the model folder
write_python_file(config_file_path, save_path)

#--- Data Load --------------------------------------------------------------------------------------------
# Load training data
train_dataset = CustomDataset(root_data_dir, train_EID_file, data_exclude_list)
train_sampler = DistributedSampler(train_dataset)
train_data = DataLoader(train_dataset, batch_size=batch_size_defined, shuffle=False, sampler=train_sampler, collate_fn=collate_custom)

# Load test data
test_dataset = CustomDataset(root_data_dir, test_EID_file, data_exclude_list)
test_sampler = DistributedSampler(test_dataset)
test_data = DataLoader(test_dataset, batch_size=batch_size_defined, shuffle=False, sampler=test_sampler, collate_fn=collate_custom)

#---------------------------------------------------------------------------------------------------------

best_accuracy = 0 # Keep track of the best validation accuracy
torch.manual_seed(42) # Set seed for reproducibility
batch_global_train = 0; batch_global_val = 0 # Keep track of the global batch index

for epoch in range(num_epochs):
    """
    Training loop
    """
    if args.verbose:
        print(f'Training model on Epoch {epoch}')
    train_sampler.set_epoch(epoch)  # Ensure each process gets a different subset of data each epoch
    running_loss_train = 0
    accuracy_batch_train = []
    for i, data_train in enumerate(train_data):
        # Debug: Print rank and batch index
        if args.verbose:
            print(f"Process {local_rank}, Epoch {epoch}, Batch {i}")

        # Extract input data and corresponding labels
        inputs_train, label_train = data_train
        inputs_train, label_train = inputs_train.to(device), label_train.to(device)
        
        # Zero your gradients for every batch
        optimizer.zero_grad()

        # Get the prediction on batch
        pred_train = model_train(inputs_train)

        # Compute loss and update
        loss_train = criterion(pred_train.float(), label_train.float())
        running_loss_train += loss_train.item()
        
        # Do the gradient backprop and adjust weights
        loss_train.backward()
        optimizer.step()
        
        # Compute accuracy AFTER optimizer update
        if i % data_record_interval == 0:
            # Compute batch accuracy
            batch_accuracy_train = (torch.softmax(pred_train, dim=0).argmax(dim=0) == label_train).sum().float() / float(label_train.size(0))
            
            if args.verbose:
                print(f'Process {local_rank}, Epoch: {epoch}, Batch {i} of {len(train_data)}, Train Loss: {loss_train.item()}, Avg Batch Accuracy: {batch_accuracy_train}')
            
            # Write to tensorboard, print epoch loss to console
            batch_global_train += data_record_interval
            writer.add_scalar('Batch Train Loss', loss_train, batch_global_train)
            writer.add_scalar('Batch Train Accuracy', batch_accuracy_train, batch_global_train)

        # Delete pred and losses to reduce memory consumption
        del loss_train, pred_train, inputs_train, label_train

    """
    Validation loop
    """
    if args.verbose:
        print(f'Validating model on Epoch {epoch}')
    running_loss_val = 0
    accuracy_batch_val = []
    with torch.no_grad():
        for j, data_val in enumerate(test_data):
            # Debug: Print rank and batch index
            if args.verbose:
                print(f"Process {local_rank}, Validation, Epoch {epoch}, Batch {j}")

            # Extract input data and corresponding labels
            inputs_val, label_val = data_val
            inputs_val, label_val = inputs_val.to(device), label_val.to(device)

            # Get the prediction on batch
            pred_val = model_train(inputs_val)

            # Compute loss and update
            loss_val = criterion(pred_val.float(), label_val.float())
            running_loss_val += loss_val.item()
            
            if j % data_record_interval == 0:
               # Compute batch accuracy
                batch_accuracy_val = (torch.softmax(pred_val, dim=0).argmax(dim=0) == label_val).sum().float() / float(label_val.size(0))

                if args.verbose:
                    print(f'Process {local_rank}, Epoch: {epoch}, Batch {j} of {len(test_data)}, Train Loss: {loss_val.item()}, Batch Accuracy: {batch_accuracy_val}')

                # Write to tensorboard, print epoch loss to console
                batch_global_val += data_record_interval
                writer.add_scalar('Batch Val Loss', loss_val, batch_global_val)
                writer.add_scalar('Batch Val Accuracy', batch_accuracy_val, batch_global_val)

            # Delete pred and losses to reduce memory consumption
            del loss_val, pred_val, inputs_val, label_val

        # If new best validation accuracy, save model
        if batch_accuracy_val > best_accuracy:
            torch.save(model_train.state_dict(), model_save_path)
            best_accuracy = batch_accuracy_val

# Close tensorboard logger instance
writer.close()