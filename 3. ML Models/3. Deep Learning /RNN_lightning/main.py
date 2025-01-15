"""
main script that trains the defined model
1. global parameters in 'training_config.py'
2. model structure and pl.Module pipeline is defined in 'model.py'
3. data loader structure is defined in 'get_data.py'
"""
# add this thing to fix some error message I don't know how to supress
import os
# os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# package imports
from torch.utils.data import DataLoader
import pytorch_lightning as pl

# supporting file imports
from training_config import *
from get_data import *
from model import *

#----------------------------------------------------------------------------------------------------------
# LOAD TRAINING DATA
train_data = DataLoader(CustomDataset(root_data_dir, train_EID_file, data_exclude_list), batch_size=batch_size_defined,
                        shuffle=shuffle_train, collate_fn=collate_custom, num_workers=10)


# LOAD TEST DATA
test_data = DataLoader(CustomDataset(root_data_dir, test_EID_file, data_exclude_list), batch_size=batch_size_defined,
                        shuffle=shuffle_test, collate_fn=collate_custom, num_workers=10)

#---------------------------------------------------------------------------------------------------------
# lets go
base_model = RNN(max_val_possible, embedding_dim, hidden_dim, output_dim, num_layers_rnn)


# train
trainer = pl.Trainer(accelerator='gpu',
                     # devices=1,
                     strategy='ddp',
                     max_epochs=3,
                    log_every_n_steps=5)

trainer.fit(base_model, train_data, test_data)