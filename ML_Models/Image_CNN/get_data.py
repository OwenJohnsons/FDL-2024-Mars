# load the libraries
from glob import glob
import os
import numpy as np
import pandas as pd

# --------------------------------------------------------------
# Only run it one time so that it creates the folders 
# separateing test and training using the EID split
#---------------------------------------------------------------

## TODO: make this efficient, this is basically copying the files 
# with training and test images in respective folders

# changes for the waterfall plots for the Mars spectral data
input_dir    = "/home/arushi/saxena-data-tree/waterfall_plots/"
file_names   = glob(input_dir + '*/*.png')
image_names  = [(f_name.split('/')[-2] + f_name.split('/')[-1]) for f_name in file_names]

sample_names = np.vectorize(lambda x: x[-9:-4:])
training_im  = sample_names(image_names)

train_labels  = pd.read_hdf('/home/arushi/fdl-2024-mars/ML_Models/data_split_EID/train_set_EID.hdf')
train_samples = list(train_labels['Sample ID'])

test_labels  = pd.read_hdf('/home/arushi/fdl-2024-mars/ML_Models/data_split_EID/test_set_EID.hdf')
test_samples = list(test_labels['Sample ID'])

pd_train     = pd.DataFrame(columns=['Sample ID', 'Labels'])
pd_test      = pd.DataFrame(columns=['Sample ID', 'Labels'])

num_train    = 0
num_test     = 0

os.system('mkdir "train/"')
os.system('mkdir "test/"')

for i in range (len(training_im)):
    sample_id  = training_im[i]

    # take care of mars samples that are strings #TODO: mars dataset
    if (sample_id[0] == '2'):
        sample_id = int(sample_id)

    if (sample_id in train_samples):
        os.system('cp ' + file_names[i] +  ' ' + ' train/' + image_names[i])
        idx        = np.argwhere(str(sample_id) == np.array(train_samples))[0][0]
        label_dict = train_labels.iloc[idx]['Labels']
        dict_out   =  {'Sample ID': image_names[i], 'Labels': label_dict}     
        pd_train.loc[num_train] = dict_out
        num_train += 1
        
    else:
        os.system('cp ' + file_names[i] + ' ' + ' test/' + image_names[i])
        idx        = np.argwhere(str(sample_id) == np.array(test_samples))[0][0]
        label_dict = test_labels.iloc[idx]['Labels']
        dict_out   =  {'Sample ID': image_names[i], 'Labels': label_dict}     
        pd_test.loc[num_test] = dict_out
        num_test += 1 