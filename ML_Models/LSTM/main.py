"""
main script that trains the defined model
1. global parameters and model instantiation happen in 'training_config.py'
2. model structure is defined in 'model.py'
3. data loader structure is defined in 'get_data.py'
"""
# add this thing to fix some error message I don't know how to supress
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# package imports
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import time

# supporting file imports
from training_config import *
from get_data import *
from save_py_as_txt import *

#----------------------------------------------------------------------------------------------------------
# define run via time/day of initiation
timestr = time.strftime("%Y%m%d-%H%M%S")
save_path = os.path.join(model_save_dir,timestr)
model_save_path = os.path.join(save_path, 'model.pt')
os.mkdir(save_path)

# instantiate the tensorboard logger
writer = SummaryWriter()

# save everything in the training_config.py as a txt file in the model folder
write_python_file('training_config.py',save_path)

#----------------------------------------------------------------------------------------------------------
# LOAD TRAINING DATA
train_data = DataLoader(CustomDataset(root_data_dir, train_EID_file), batch_size=batch_size_defined,
                        shuffle=shuffle_train, collate_fn=collate_custom)

# LOAD TEST DATA
test_data = DataLoader(CustomDataset(root_data_dir, test_EID_file), batch_size=batch_size_defined,
                        shuffle=shuffle_test, collate_fn=collate_custom)


#----------------------------------------------------------------------------------------------------------
best_accuracy = 0
for epoch in range(num_epochs):
    """
    train loop
    """
    print(f'Training model on Epoch {epoch}')
    model_train.train()
    running_loss_train = 0
    accuracy_batch_train = []
    for i, data in enumerate(train_data):
        # extract input data and corresponding labels
        input, label = data

        # Zero your gradients for every batch
        optimizer.zero_grad()

        # get the prediction on batch
        pred = model_train(input)
        print(pred, label)

        # compute loss and update
        loss = criterion(pred.float(), label.float())
        running_loss_train+=loss.item()

        # do the gradient backprop and adjust weights
        loss.backward()
        # Adjust learning weights
        optimizer.step()

        # compute accuracy AFTER optimizer update
        accuracy_batch = (torch.softmax(pred, dim=0).argmax(dim=0) == label).sum().float() / float( label.size(0) )

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
    for j, data in enumerate(test_data):
        # extract input data and corresponding labels
        input, label = data
        # get the prediction on batch
        pred = model_train(input)

        # compute loss and update
        loss = criterion(pred.float(), label.float())
        running_loss_val += loss.item()

        # compute accuracy
        accuracy_batch = (torch.softmax(pred, dim=0).argmax(dim=0) == label).sum().float() / float( label.size(0) )
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