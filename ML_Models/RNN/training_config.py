"""
this script houses all the global parameters, divided by category as necessary
"""
import torch
import torch.nn as nn

from model import RNN

#------------------------------------------------------------------------------------
# data_dir path for all raw data
root_data_dir = "C:/Users/Kanak Parmar/Desktop/FDL 2024/Mars Data/Codes/samurai_data_base"

# train/test EID files
train_EID_file = 'C:/Users/Kanak Parmar/Desktop/FDL 2024/Mars Data/Codes/ML_pipelines/data_split_EID/train_set_EID.hdf'
test_EID_file = 'C:/Users/Kanak Parmar/Desktop/FDL 2024/Mars Data/Codes/ML_pipelines/data_split_EID/test_set_EID.hdf'

#------------------------------------------------------------------------------------
model_save_dir = 'C:/Users/Kanak Parmar/Desktop/FDL 2024/Mars Data/Codes/ML_pipelines/RNN/trained_models'
model_eval_fig_dir = 'C:/Users/Kanak Parmar/Desktop/FDL 2024/Mars Data/Codes/ML_pipelines/RNN/model_eval'

# dataloader variables
batch_size_defined = 3 # MUST BE 2 OR MORE
shuffle_train = True #DO NOT CHANGE
shuffle_test = False #DO NOT CHANGE

#------------------------------------------------------------------------------------
# model params
max_val_possible = 201 #DO NOT CHANGE FOR THIS DATASET
embedding_dim = 25
hidden_dim = 256
output_dim = 10 #DO NOT CHANGE (we only have 10 classes in the data)
num_layers_rnn = 1

# call the model and instantiate
model_train = RNN(max_val_possible, embedding_dim, hidden_dim, output_dim, num_layers_rnn)

device = 'cuda' if torch.cuda.is_available() else 'cpu' #DO NOT CHANGE
model_train.to(device)

#------------------------------------------------------------------------------------
"""
list of available optimizer algorithm in pytorch: https://pytorch.org/docs/stable/optim.html#algorithms
"""
# training hyperparams
learning_rate = 1e-4
optimizer = torch.optim.AdamW(model_train.parameters(), lr=learning_rate)
num_epochs = 10

"""
list of available loss functions in pytorch: https://pytorch.org/docs/stable/nn.html#loss-functions
"""
# criterion (aka loss function)
criterion = nn.CrossEntropyLoss()