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
    dist.init_process_group(backend=backend)
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    return local_rank

def parse_args():
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
    model_train, optimizer, device,
    batch_size_defined, num_epochs, data_record_interval,
    model_save_dir, config_file_path,
    shuffle_train, shuffle_test,
    root_data_dir, train_EID_file, test_EID_file, data_exclude_list
)
from get_data import CustomDataset, collate_custom

if torch.cuda.device_count() > 1:
    model_train = nn.parallel.DistributedDataParallel(model_train, device_ids=[local_rank])

timestr = time.strftime("%Y%m%d-%H%M%S")
save_path = os.path.join(model_save_dir, timestr)
model_save_path = os.path.join(save_path, 'model.pt')

os.makedirs(save_path, exist_ok=True)

if not os.path.exists(save_path):
    print('Making save folder...')
    os.mkdir(save_path)

writer = SummaryWriter(save_path)
write_python_file(config_file_path, save_path)

train_dataset = CustomDataset(root_data_dir, train_EID_file, data_exclude_list)
train_sampler = DistributedSampler(train_dataset)
train_data = DataLoader(train_dataset, batch_size=batch_size_defined, shuffle=False, sampler=train_sampler, collate_fn=collate_custom)

test_dataset = CustomDataset(root_data_dir, test_EID_file, data_exclude_list)
test_sampler = DistributedSampler(test_dataset)
test_data = DataLoader(test_dataset, batch_size=batch_size_defined, shuffle=False, sampler=test_sampler, collate_fn=collate_custom)

best_accuracy = 0
torch.manual_seed(42)
batch_global_train = 0
batch_global_val = 0

# Use BCEWithLogitsLoss for multi-label classification
criterion = nn.BCEWithLogitsLoss()

for epoch in range(num_epochs):
    if args.verbose:
        print(f'Training model on Epoch {epoch}')
    train_sampler.set_epoch(epoch)
    running_loss_train = 0
    accuracy_batch_train = []
    for i, data_train in enumerate(train_data):
        if args.verbose:
            print(f"Process {local_rank}, Epoch {epoch}, Batch {i}")

        inputs_train, label_train = data_train
        inputs_train, label_train = inputs_train.to(device), label_train.to(device)
        
        optimizer.zero_grad()
        pred_train = model_train(inputs_train)
        loss_train = criterion(pred_train, label_train)
        running_loss_train += loss_train.item()
        loss_train.backward()
        optimizer.step()
        
        if i % data_record_interval == 0:
            # Compute batch accuracy for each label
            pred_train_sigmoid = torch.sigmoid(pred_train)
            batch_accuracy_train = ((pred_train_sigmoid > 0.5) == label_train).float().mean()

            if args.verbose:
                print(f'Process {local_rank}, Epoch: {epoch}, Batch {i} of {len(train_data)}, Train Loss: {loss_train.item()}, Avg Batch Accuracy: {batch_accuracy_train}')
            
            batch_global_train += data_record_interval
            writer.add_scalar('Batch Train Loss', loss_train, batch_global_train)
            writer.add_scalar('Batch Train Accuracy', batch_accuracy_train, batch_global_train)

        del loss_train, pred_train, inputs_train, label_train

    if args.verbose:
        print(f'Validating model on Epoch {epoch}')
    running_loss_val = 0
    accuracy_batch_val = []
    with torch.no_grad():
        for j, data_val in enumerate(test_data):
            if args.verbose:
                print(f"Process {local_rank}, Validation, Epoch {epoch}, Batch {j}")

            inputs_val, label_val = data_val
            inputs_val, label_val = inputs_val.to(device), label_val.to(device)

            pred_val = model_train(inputs_val)
            loss_val = criterion(pred_val, label_val)
            running_loss_val += loss_val.item()
            
            if j % data_record_interval == 0:
                pred_val_sigmoid = torch.sigmoid(pred_val)
                batch_accuracy_val = ((pred_val_sigmoid > 0.5) == label_val).float().mean()

                if args.verbose:
                    print(f'Process {local_rank}, Epoch: {epoch}, Batch {j} of {len(test_data)}, Val Loss: {loss_val.item()}, Batch Accuracy: {batch_accuracy_val}')

                batch_global_val += data_record_interval
                writer.add_scalar('Batch Val Loss', loss_val, batch_global_val)
                writer.add_scalar('Batch Val Accuracy', batch_accuracy_val, batch_global_val)

            del loss_val, pred_val, inputs_val, label_val

        if batch_accuracy_val > best_accuracy:
            torch.save(model_train.state_dict(), model_save_path)
            best_accuracy = batch_accuracy_val

writer.close()