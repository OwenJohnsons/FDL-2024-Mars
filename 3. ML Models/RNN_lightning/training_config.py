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


# dataloader variables
batch_size_defined = 35 # MUST BE BETWEEN 2 AND 32, anything over 32 risks chance to crash code via memory errors
shuffle_train = True #DO NOT CHANGE 
shuffle_test = False #DO NOT CHANGE

#------------------------------------------------------------------------------------
# model params
max_val_possible = 201 #DO NOT CHANGE FOR THIS DATASET
embedding_dim = 15
hidden_dim = 25
output_dim = 10 #DO NOT CHANGE (we only have 10 classes in the data)
num_layers_rnn = 1