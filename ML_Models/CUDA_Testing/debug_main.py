"""
Main script that trains the defined model
1. Global parameters and model instantiation happen in 'training_config.py'
2. Model structure is defined in 'model.py'
3. Data loader structure is defined in 'get_data.py'
"""
import os
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import time
import gc
import numpy as np
from torch.cuda.amp import GradScaler, autocast
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# Initialize process group for distributed training
dist.init_process_group(backend='nccl')

# Import the rest of the modules
from training_config import *
from get_data import *
from save_py_as_txt import *

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

#----------------------------------------------------------------------------------------------------------
# Define run via time/day of initiation
timestr = time.strftime("%Y%m%d-%H%M%S")
save_path = os.path.join(model_save_dir, timestr)
model_save_path = os.path.join(save_path, 'model.pt')

if not os.path.exists(save_path):
    print('Making save folder...')
    os.mkdir(save_path)

# Instantiate the tensorboard logger
writer = SummaryWriter()

# Save everything in the training_config.py as a txt file in the model folder
write_python_file(config_file_path, save_path)

#----------------------------------------------------------------------------------------------------------
# Load training data
train_sampler = torch.utils.data.distributed.DistributedSampler(
    CustomDataset(root_data_dir, train_EID_file))
train_data = DataLoader(CustomDataset(root_data_dir, train_EID_file), batch_size=batch_size_defined, 
                        shuffle=False, sampler=train_sampler, collate_fn=collate_custom, num_workers=4, pin_memory=True)

test_sampler = torch.utils.data.distributed.DistributedSampler(
    CustomDataset(root_data_dir, test_EID_file))
test_data = DataLoader(CustomDataset(root_data_dir, test_EID_file), batch_size=batch_size_defined, 
                       shuffle=False, sampler=test_sampler, collate_fn=collate_custom, num_workers=4, pin_memory=True)

#----------------------------------------------------------------------------------------------------------
torch.cuda.empty_cache()

# Move model to the appropriate device and wrap with DistributedDataParallel
device = torch.device('cuda', dist.get_rank())
model_train = model_train.to(device)
model_train = DDP(model_train, device_ids=[dist.get_rank()])

best_accuracy = 0
accumulation_steps = 4  # Gradient accumulation steps
scaler = GradScaler()

for epoch in range(num_epochs):
    """
    Train loop
    """
    print(f'Training model on Epoch {epoch}')
    model_train.train()
    running_loss_train = 0
    accuracy_batch_train = []

    optimizer.zero_grad()
    train_sampler.set_epoch(epoch)

    for i, data in enumerate(train_data):
        inputs, label = data
        inputs, label = inputs.to(device), label.to(device)
        
        with autocast():
            pred = model_train(inputs)
            loss = criterion(pred.float(), label.float())
        
        scaler.scale(loss).backward()
        
        if (i + 1) % accumulation_steps == 0:  # Update weights every accumulation_steps
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        running_loss_train += loss.item()
        accuracy_batch = (torch.softmax(pred, dim=0).argmax(dim=0) == label).sum().float() / float(label.size(0))
        accuracy_batch_train.append(accuracy_batch.item())

    accuracy_epoch_train = np.mean(accuracy_batch_train)

    # Write to tensorboard, print epoch loss to console
    if dist.get_rank() == 0:  # Only log from the main process
        writer.add_scalar('Epoch Train Loss', running_loss_train / len(train_data), epoch)
        writer.add_scalar('Epoch Train Avg Accuracy', accuracy_epoch_train, epoch)
    print(f'Epoch: {epoch}, Epoch Train Loss: {running_loss_train / len(train_data)}, Avg Accuracy: {accuracy_epoch_train}')

    """
    Validation loop
    """
    print(f'Validating model on Epoch {epoch}')
    model_train.eval()
    running_loss_val = 0
    accuracy_batch_val = []

    with torch.no_grad():
        for j, data in enumerate(test_data):
            inputs, label = data
            inputs, label = inputs.to(device), label.to(device)

            with autocast():
                pred = model_train(inputs)
                loss = criterion(pred.float(), label.float())
            
            running_loss_val += loss.item()
            accuracy_batch = (torch.softmax(pred, dim=0).argmax(dim=0) == label).sum().float() / float(label.size(0))
            accuracy_batch_val.append(accuracy_batch.item())

    accuracy_epoch_val = np.mean(accuracy_batch_val)

    # Write to tensorboard, print epoch loss to console
    if dist.get_rank() == 0:  # Only log from the main process
        writer.add_scalar('Epoch Val Loss', running_loss_val / len(test_data), epoch)
        writer.add_scalar('Epoch Val Avg Accuracy', accuracy_epoch_val, epoch)
    print(f'Epoch: {epoch}, Epoch Val Loss: {running_loss_val / len(test_data)}, Avg Accuracy: {accuracy_epoch_val}')

    # If new best val accuracy, save model
    if accuracy_epoch_val > best_accuracy and dist.get_rank() == 0:  # Only save from the main process
        torch.save(model_train.state_dict(), model_save_path)
        best_accuracy = accuracy_epoch_val

    # Clear cache and garbage collect
    torch.cuda.empty_cache()
    gc.collect()

# Close tensorboard logger instance
if dist.get_rank() == 0:
    writer.close()

# Clean up the process group
dist.destroy_process_group()
