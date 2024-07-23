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
writer = SummaryWriter(save_path)

# save everything in the training_config.py as a txt file in the model folder
write_python_file('training_config.py',save_path)

#----------------------------------------------------------------------------------------------------------
# LOAD TRAINING DATA
train_data = DataLoader(CustomDataset(root_data_dir, train_EID_file, data_exclude_list), batch_size=batch_size_defined,
                        shuffle=shuffle_train, collate_fn=collate_custom)

# LOAD TEST DATA
test_data = DataLoader(CustomDataset(root_data_dir, test_EID_file, data_exclude_list), batch_size=batch_size_defined,
                        shuffle=shuffle_test, collate_fn=collate_custom)


#----------------------------------------------------------------------------------------------------------
best_accuracy = 0

torch.manual_seed(42)

for epoch in range(num_epochs):
    """
    train loop
    """
    print(f'Training model on Epoch {epoch}')
    running_loss_train = 0
    accuracy_batch_train = []
    for i, data_train in enumerate(train_data):
        # extract input data and corresponding labels
        inputs_train, label_train = data_train
        inputs_train, label_train = inputs_train.to(device), label_train.to(device)
        
        # Zero your gradients for every batch
        optimizer.zero_grad()

        # get the prediction on batch
        pred_train = model_train(inputs_train, device)

        # compute loss and update
        loss_train = criterion(pred_train.float(), label_train.float())
        running_loss_train += loss_train.item()
        
        # do the gradient backprop and adjust weights
        loss_train.backward()
        # Adjust learning weights
        optimizer.step()
        
        # compute accuracy AFTER optimizer update
        # accuracy_batch = (torch.softmax(pred_train, dim=0).argmax(dim=0) == label_train).sum().float() / float( label_train.size(0) )
        
                    
        if i % data_record_interval == 0:
            # compute batch accuracy
            batch_accuracy_train = (torch.softmax(pred_train, dim=0).argmax(dim=0) == label_train).sum().float() / float( label_train.size(0) )
            
            print(f'Epoch: {epoch}, Batch {i} of {len(train_data)}, Train Loss: {loss_train.item()}, Avg Batch Accuracy: {batch_accuracy_train}')
            
            # write to tensorboard, print epoch loss to console
            writer.add_scalar('Batch Train Loss', loss_train, i)
            writer.add_scalar('Batch Train Accuracy', batch_accuracy_train, i)

       
        # delete pred and losses to reduce memory consumption>
        del loss_train, pred_train, inputs_train, label_train



    """
    validation loop
    """
    print(f'Validating model on Epoch {epoch}')
    # model_train.eval()
    running_loss_val = 0
    accuracy_batch_val = []
    with torch.no_grad():
        for j, data_val in enumerate(test_data):
            # extract input data and corresponding labels
            inputs_val, label_val = data_val
            inputs_val, label_val = inputs_val.to(device), label_val.to(device)

            # get the prediction on batch
            pred_val = model_train(inputs_val, device)

            # compute loss and update
            loss_val = criterion(pred_val.float(), label_val.float())
            running_loss_val += loss_val.item()
            
            if j % data_record_interval == 0:
               # compute batch accuracy
                batch_accuracy_val = (torch.softmax(pred_val, dim=0).argmax(dim=0) == label_val).sum().float() / float( label_val.size(0) )

                print(f'Epoch: {epoch}, Batch {j} of {len(test_data)}, Train Loss: {loss_val.item()}, Batch Accuracy: {batch_accuracy_val}')

                # write to tensorboard, print epoch loss to console
                writer.add_scalar('Batch Train Loss', loss_val, j)
                writer.add_scalar('Batch Train Accuracy', batch_accuracy_val, j)
            
            # delete pred and losses to reduce memory consumption>
            del loss_val, pred_val, inputs_val, label_val

        # if new best val accuracy, save model
        if batch_accuracy_val > best_accuracy:
            torch.save(model_train.state_dict(), model_save_path)
            best_accuracy = batch_accuracy_val
        

# close tensorboard logger instance
writer.close()