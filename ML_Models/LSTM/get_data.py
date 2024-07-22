from torch.utils.data import Dataset
import glob
import os
import pandas as pd
import numpy as np
from torch.nn.utils.rnn import pad_sequence
import torch


class CustomDataset(Dataset):
    def __init__(self, root_dir_main, dataset_dir, data_exclude_list):
        self.root_dir = root_dir_main
        self.folders = [f for f in os.listdir(self.root_dir) if os.path.isdir(os.path.join(self.root_dir, f))]
        self.paths_matched = []
        self.EID_matched = []

        # extract the list of EIDs
        self.data_subset = pd.read_hdf(dataset_dir)
        EID_samples = list(self.data_subset['Sample ID'])

        for label, folders in enumerate(self.folders):
            data_paths = glob.glob(os.path.join(root_dir_main, folders, '*.hdf'))
            basenames = [os.path.splitext(os.path.basename(file))[0] for file in data_paths]

            # match the basenames
            basenames_match = [name for name in basenames if name in EID_samples and name not in data_exclude_list]

            # map matched basenames back to full paths
            matched_paths = [file for file in data_paths if os.path.splitext(os.path.basename(file))[0] in basenames_match]

            self.paths_matched.extend(matched_paths)
            self.EID_matched.extend(basenames_match)

    def __len__(self):
        return len(self.paths_matched)

    def __getitem__(self, idx):
        # print(idx)
        # get data and make the raw data stack
        data_path_raw = self.paths_matched[idx]
        # print(data_path_raw)
        data = pd.read_hdf(data_path_raw)
        col1 = data['m/z']
        col2 = data.iloc[:, -1]
        data_stack = np.vstack((col1, col2)).T

        # get corresponding labels
        EID_sample = self.EID_matched[idx]
        # print(EID_sample)


        # label_idx = self.data_subset.index[self.data_subset['Sample ID'] == EID_sample]
        # isolated_row = self.data_subset[self.data_subset['Sample ID'] == EID_sample]['Labels'].values()
        isolated_row = int(np.where(self.data_subset['Sample ID'] == str(EID_sample))[0])
        # print(isolated_row)

        EID_label_dict = self.data_subset['Labels'].iloc[isolated_row]
        EID_label_array = [int(value) for key, value in sorted(EID_label_dict.items())]

        return data_stack, EID_label_array

"""
pad collate from:
https://suzyahyah.github.io/pytorch/2019/07/01/DataLoader-Pad-Pack-Sequence.html

https://www.codefull.net/2018/11/use-pytorchs-dataloader-with-variable-length-sequences-for-lstm-gru/
"""
# DO NOT CHANGE THIS, YOU WILL BREAK THE CODE
def collate_custom(data):
    xraw = []
    yraw = []
    label_stack = []
    for i in range(len(data)):
        data_transposed = data[i][0].T
        label_local = data[i][1]
        # print(test_label)
        xraw.append(torch.tensor(data_transposed[0]))
        yraw.append(torch.tensor(data_transposed[1]))
        label_stack.append(torch.tensor(label_local))

    padded_x = pad_sequence(xraw, batch_first=True, padding_value=0)
    padded_y = pad_sequence(yraw, batch_first=True, padding_value=0)

    batch_stack = []
    for a,b in zip(padded_x, padded_y):
        stack = np.vstack((a,b)).T
        batch_stack.append(stack)

    return torch.tensor(np.array(batch_stack)), torch.tensor(np.array(label_stack))