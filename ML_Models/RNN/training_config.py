"""
this script houses all the global parameters, divided by category as necessary
"""
import torch
import torch.nn as nn
import os

from model import RNN

#------------------------------------------------------------------------------------
# data_dir path for all raw data
root_data_dir = "/home/jupyter/data_tree_main/samurai_data_mini"

# train/test EID files
train_EID_file = '/home/jupyter/data_split_EID/train_set_EID.hdf'
test_EID_file = '/home/jupyter/data_split_EID/test_set_EID.hdf'

data_exclude_list = ['S0401']


#------------------------------------------------------------------------------------
model_save_dir = '/home/jupyter/RNN/trained_models'
model_eval_fig_dir = '/home/jupyter/RNN/model_eval'

config_file_path = '/home/jupyter/RNN/training_config.py'

# dataloader variables
batch_size_defined = 25 # MUST BE BETWEEN 2 AND 32, anything over 32 risks chance to crash code via memory errors
shuffle_train = True #DO NOT CHANGE 
shuffle_test = False #DO NOT CHANGE

#------------------------------------------------------------------------------------
# model params
max_val_possible = 201 #DO NOT CHANGE FOR THIS DATASET
embedding_dim = 3
hidden_dim = 25
output_dim = 10 #DO NOT CHANGE (we only have 10 classes in the data)
num_layers_rnn = 1

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
data_record_interval = 2
learning_rate = 1e-4
weight_decay = 1e-5
optimizer = torch.optim.Adam(model_train.parameters(), lr=learning_rate)


"""
list of available loss functions in pytorch: https://pytorch.org/docs/stable/nn.html#loss-functions
"""
# criterion (aka loss function)
criterion = nn.CrossEntropyLoss(reduction='sum')

