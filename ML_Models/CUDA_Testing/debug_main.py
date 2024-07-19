import os
import time
import torch
import torch.cuda.amp as amp
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from training_config import *
from get_data import *
from save_py_as_txt import *

#----------------------------------------------------------------------------------------------------------
# define run via time/day of initiation
timestr = time.strftime("%Y%m%d-%H%M%S")
save_path = os.path.join(model_save_dir, timestr)
model_save_path = os.path.join(save_path, 'model.pt')
os.mkdir(save_path)

# instantiate the tensorboard logger
writer = SummaryWriter()

# save everything in the training_config.py as a txt file in the model folder
write_python_file('training_config.py', save_path)

#----------------------------------------------------------------------------------------------------------
# Ensure the model is on GPU
model_train = model_train.cuda()

# LOAD TRAINING DATA
train_data = DataLoader(CustomDataset(root_data_dir, train_EID_file), batch_size=batch_size_defined,
                        shuffle=shuffle_train, collate_fn=collate_custom, num_workers=4, pin_memory=True)

# LOAD TEST DATA
test_data = DataLoader(CustomDataset(root_data_dir, test_EID_file), batch_size=batch_size_defined,
                        shuffle=shuffle_test, collate_fn=collate_custom, num_workers=4, pin_memory=True)


#----------------------------------------------------------------------------------------------------------
scaler = amp.GradScaler()
best_accuracy = 0
accumulation_steps = 4  # Number of batches to accumulate gradients over

for epoch in range(num_epochs):
    """
    train loop
    """
    print(f'Training model on Epoch {epoch}')
    model_train.train()
    running_loss_train = 0
    accuracy_batch_train = []
    optimizer.zero_grad()

    for i, data in enumerate(train_data):
        # Track memory usage
        print(f"Batch {i}: Allocated memory: {torch.cuda.memory_allocated() / 1024**2} MB, Reserved memory: {torch.cuda.memory_reserved() / 1024**2} MB")
        
        # extract input data and corresponding labels
        input, label = data
        input, label = input.cuda(), label.cuda()

        with amp.autocast():
            # get the prediction on batch
            pred = model_train(input)
            loss = criterion(pred.float(), label.float())
        
        # do the gradient backprop and adjust weights
        scaler.scale(loss).backward()

        if (i + 1) % accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            torch.cuda.empty_cache()  # Clear cache periodically

        running_loss_train += loss.item()

        # compute accuracy AFTER optimizer update
        accuracy_batch = (torch.softmax(pred, dim=0).argmax(dim=0) == label).sum().float() / float(label.size(0))
        accuracy_batch_train.append(accuracy_batch.item())

    accuracy_epoch_train = np.mean(accuracy_batch_train)

    # write to tensorboard, print epoch loss to console
    writer.add_scalar('Epoch Train Loss', running_loss_train/len(train_data), epoch)
    writer.add_scalar('Epoch Train Avg Accuracy', accuracy_epoch_train, epoch)
    print(f'Epoch: {epoch}, Epoch Train Loss: {running_loss_train/len(train_data)}, Avg Accuracy: {accuracy_epoch_train}')


    """
    validation loop
    """
    print(f'Validating model on Epoch {epoch}')
    model_train.eval()
    running_loss_val = 0
    accuracy_batch_val = []

    with torch.no_grad():
        for j, data in enumerate(test_data):
            # Track memory usage
            print(f"Batch {j}: Allocated memory: {torch.cuda.memory_allocated() / 1024**2} MB, Reserved memory: {torch.cuda.memory_reserved() / 1024**2} MB")
            
            # extract input data and corresponding labels
            input, label = data
            input, label = input.cuda(), label.cuda()

            with amp.autocast():
                # get the prediction on batch
                pred = model_train(input)
                loss = criterion(pred.float(), label.float())

            running_loss_val += loss.item()

            # compute accuracy
            accuracy_batch = (torch.softmax(pred, dim=0).argmax(dim=0) == label).sum().float() / float(label.size(0))
            accuracy_batch_val.append(accuracy_batch.item())

    accuracy_epoch_val = np.mean(accuracy_batch_val)

    # write to tensorboard, print epoch loss to console
    writer.add_scalar('Epoch Val Loss', running_loss_val / len(test_data), epoch)
    writer.add_scalar('Epoch Val Avg Accuracy', accuracy_epoch_val, epoch)
    print(f'Epoch: {epoch}, Epoch Val Loss: {running_loss_val / len(test_data)}, Avg Accuracy: {accuracy_epoch_val}')

    # if new best val accuracy, save model
    if accuracy_epoch_val > best_accuracy:
        torch.save(model_train.state_dict(), model_save_path)
        best_accuracy = accuracy_epoch_val

# close tensorboard logger instance
writer.close()