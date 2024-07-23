"""
this script houses all the global parameters, divided by category as necessary
"""
import torch
import torch.nn as nn
import os
from model import RNN

# - Directory Definitions --------------------------------------------------------------
code_root_dir = "/home/owenj/fdl-2024-mars/" # change to the root directory of the code repository
bucket_dir = "/home/owenj/bucket" # change to the root directory of the mounted bucket

root_data_dir = bucket_dir + '/samurai_data_base/' # Data directory, change to samurai_data_mini to use smaller dataset for debugging. 

# - EID metadate file paths -----------------------------------------------------------
train_EID_file = code_root_dir + '/ML_Models/data_split_EID/train_set_EID.hdf'
test_EID_file = code_root_dir + '/ML_Models/data_split_EID/test_set_EID.hdf'

data_exclude_list = ['S0401'] # exlusionary file due to extensive array size. 
#- Output Directories -----------------------------------------------------------------
model_save_dir = code_root_dir + '/ML_Models/parallel-processing-ver/RNN/trained_models'
model_eval_fig_dir = code_root_dir + '/ML_Models/parallel-processing-ver/RNN/model_eval'

config_file_path = code_root_dir + '/ML_Models/parallel-processing-ver/RNN/training_config.py'

# - Dataloader Variables --------------------------------------------------------------
batch_size_defined = 30 # MUST BE BETWEEN 2 AND 32, anything over 32 risks chance to crash code via memory errors
shuffle_train = True #DO NOT CHANGE 
shuffle_test = False #DO NOT CHANGE

#- Model Parameters -------------------------------------------------------------------
max_val_possible = 201 #DO NOT CHANGE FOR THIS DATASET
embedding_dim = 5
hidden_dim = 5
output_dim = 10 #DO NOT CHANGE (we only have 10 classes in the data)
num_layers_rnn = 1

# - Model Definition ------------------------------------------------------------------
model_train = RNN(max_val_possible, embedding_dim, hidden_dim, output_dim, num_layers_rnn) # call the model and instantiate

device = 'cuda' if torch.cuda.is_available() else 'cpu' #DO NOT CHANGE
print(f'Using device: {device}')

model_train.to(device)

# - Training Parameters --------------------------------------------------------------
num_epochs = 10
data_record_interval = 2
learning_rate = 1e-4
weight_decay = 1e-5
optimizer = torch.optim.Adam(model_train.parameters(), lr=learning_rate)

# - Criterion -----------------------------------------------------------------------
criterion = nn.CrossEntropyLoss(reduction='sum')
