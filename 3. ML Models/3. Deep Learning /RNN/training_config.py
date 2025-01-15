"""
this script houses all the global parameters, divided by category as necessary
"""
import torch
import torch.nn as nn
import os

from model import RNN

#------------------------------------------------------------------------------------
# data_dir path for all raw data
root_data_dir = "/home/jupyter/pugal_data_tree/samurai_data_mini"
# root_data_dir = "/home/jupyter/pugal_data_tree/samurai_data_base"

# train/test EID files
train_EID_file = '/home/jupyter/fdl-2024-mars/ML_Models/data_split_EID/train_set_EID.hdf'   
test_EID_file = '/home/jupyter/fdl-2024-mars/ML_Models/data_split_EID/test_set_EID.hdf'

data_exclude_list = ['S0401']


#------------------------------------------------------------------------------------
model_save_dir = '/home/jupyter/fdl-2024-mars/ML_Models/RNN/trained_models'
model_eval_fig_dir = '/home/jupyter/fdl-2024-mars/ML_Models/RNN/model_eval'

config_file_path = '/home/jupyter/fdl-2024-mars/ML_Models/RNN/training_config.py'

# dataloader variables
batch_size_defined = 30 # MUST BE BETWEEN 2 AND 32, anything over 32 risks chance to crash code via memory errors
# batch_size_defined = 7


shuffle_train = True #DO NOT CHANGE 
shuffle_test = False #DO NOT CHANGE

#------------------------------------------------------------------------------------
# model params
max_val_possible = 201 #DO NOT CHANGE FOR THIS DATASET

# embedding_dim = 5
embedding_dim = 10

# hidden_dim = 5
hidden_dim = 25

output_dim = 10 #DO NOT CHANGE (we only have 10 classes in the data)

# num_layers_rnn = 1
num_layers_rnn = 2

# call the model and instantiate
model_train = RNN(max_val_possible, embedding_dim, hidden_dim, output_dim, num_layers_rnn)

device = 'cuda' if torch.cuda.is_available() else 'cpu' #DO NOT CHANGE
print(f'Using device: {device}')

if torch.cuda.device_count() > 1:
    model_train = nn.parallel.DistributedDataParallel(model_train)

model_train.to(device)

#------------------------------------------------------------------------------------
"""
list of available optimizer algorithm in pytorch: https://pytorch.org/docs/stable/optim.html#algorithms
"""
# training hyperparams
num_epochs = 10
# num_epochs = 20

data_record_interval = 2

learning_rate = 1e-4

weight_decay = 1e-5

# optimizer = torch.optim.Adam(model_train.parameters(), lr=learning_rate)   
optimizer = torch.optim.Adam(model_train.parameters(), lr=learning_rate, weight_decay=weight_decay)

# Including the exponential learning rate scheduler:
# (Ref: https://pytorch.org/docs/stable/optim.html)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)


"""
list of available loss functions in pytorch: https://pytorch.org/docs/stable/nn.html#loss-functions
"""
# criterion (aka loss function)
# # Kanak's approach:
# criterion = nn.CrossEntropyLoss(reduction='sum')

# Pugazh's change:
# Initiate a tensor of ones for position weight:
pos_weight = torch.ones([output_dim])

# Load the tensor to CUDA to prevent run-time error:
pos_weight = pos_weight.to(device)

# Define the loss function:
# criterion = torch.nn.BCEWithLogitsLoss(reduction = 'none', pos_weight=pos_weight)  # Error: grad can be implicitly created only for scalar outputs
# criterion = torch.nn.BCEWithLogitsLoss(reduction = 'mean', pos_weight=pos_weight)

# # Kanak's suggestion:
# criterion = torch.nn.BCEWithLogitsLoss(reduction = 'sum', pos_weight=pos_weight)

# Another Kanak's suggestion:
criterion = torch.nn.BCEWithLogitsLoss(reduction = 'mean', pos_weight=pos_weight)

