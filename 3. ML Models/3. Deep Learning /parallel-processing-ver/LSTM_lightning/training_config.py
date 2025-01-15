"""
this script houses all the global parameters, divided by category as necessary
"""
import torch
import torch.nn as nn

from model import LSTM

#------------------------------------------------------------------------------------
# data_dir path for all raw data
root_data_dir = "/home/jupyter/data_tree_main/samurai_data_mini"

# train/test EID files
train_EID_file = '/home/jupyter/data_split_EID/train_set_EID.hdf'
test_EID_file = '/home/jupyter/data_split_EID/test_set_EID.hdf'

data_exclude_list = ['S0401']


#------------------------------------------------------------------------------------
model_save_dir = '/home/jupyter/LSTM/trained_models'
model_eval_fig_dir = '/home/jupyter/LSTM/model_eval'

config_file_path = '/home/jupyter/LSTM/training_config.py'

# dataloader variables
batch_size_defined = 10 # MUST BE BETWEEN 2 AND 32, anything over 32 risks chance to crash code via memory errors
shuffle_train = True #DO NOT CHANGE 
shuffle_test = False #DO NOT CHANGE

#------------------------------------------------------------------------------------
model_save_dir = '/home/jupyter/LSTM/trained_models'
model_eval_fig_dir = '/home/jupyter/LSTM/model_eval'

# model params
lstm_features = 2 #DO NOT CHANGE (pleas!)
lstm_hidden_units = 128
output_dim = 10 #DO NOT CHANGE (we only have 10 classes in the data)
num_layers_lstm = 1