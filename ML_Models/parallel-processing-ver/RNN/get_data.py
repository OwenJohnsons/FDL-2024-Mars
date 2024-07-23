import os
import glob
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

class CustomDataset(Dataset):
    def __init__(self, root_dir_main, dataset_dir, data_exclude_list):
        """
        Initializes the CustomDataset with the specified directory paths and exclusion list.

        Args:
            root_dir_main (str): Path to the main directory containing the data folders.
            dataset_dir (str): Path to the HDF5 file containing the dataset metadata.
            data_exclude_list (list): List of Sample IDs to exclude from the dataset.
        """
        self.root_dir = root_dir_main
        self.folders = [f for f in os.listdir(self.root_dir) if os.path.isdir(os.path.join(self.root_dir, f))]
        self.paths_matched = []
        self.EID_matched = []

        # Extract the list of EIDs from the dataset
        self.data_subset = pd.read_hdf(dataset_dir)
        EID_samples = list(self.data_subset['Sample ID'])

        # Match folders and file paths based on the EID_samples and exclusion list
        for label, folders in enumerate(self.folders):
            data_paths = glob.glob(os.path.join(root_dir_main, folders, '*.hdf'))
            basenames = [os.path.splitext(os.path.basename(file))[0] for file in data_paths]

            # Filter basenames to include only those in EID_samples and not in the exclusion list
            basenames_match = [name for name in basenames if name in EID_samples and name not in data_exclude_list]

            # Map matched basenames back to full file paths
            matched_paths = [file for file in data_paths if os.path.splitext(os.path.basename(file))[0] in basenames_match]

            self.paths_matched.extend(matched_paths)
            self.EID_matched.extend(basenames_match)

    def __len__(self):
        """
        Returns the total number of matched paths in the dataset.

        Returns:
            int: Number of matched paths.
        """
        return len(self.paths_matched)

    def __getitem__(self, idx):
        """
        Retrieves the data and labels for a given index.

        Args:
            idx (int): Index of the data item to retrieve.

        Returns:
            tuple: Flattened data stack and corresponding label array.
        """
        # Retrieve the raw data path for the given index
        data_path_raw = self.paths_matched[idx]
        data = pd.read_hdf(data_path_raw)

        # Extract the m/z column and the last column from the data
        col1 = data['m/z']
        col2 = data.iloc[:, -1]
        data_stack = np.vstack((col1, col2)).T
        flat_data_stack = data_stack.flatten()

        # Get the corresponding EID sample
        EID_sample = self.EID_matched[idx]

        # Find the isolated row corresponding to the EID sample
        isolated_row = int(np.where(self.data_subset['Sample ID'] == str(EID_sample))[0])

        # Extract the label dictionary and convert it to an array
        EID_label_dict = self.data_subset['Labels'].iloc[isolated_row]
        EID_label_array = [int(value) for key, value in sorted(EID_label_dict.items())]

        return flat_data_stack, EID_label_array


def collate_custom(data):
    """
    Collates a batch of data by padding sequences and converting labels to tensors.

    Args:
        data (list of tuples): Each tuple contains a sequence and its corresponding label.

    Returns:
        tuple: Padded sequences tensor and labels tensor.
    """
    xraw = []
    label_raw = []

    # Separate the sequences and labels
    for i in range(len(data)):
        xraw.append(torch.tensor(data[i][0]))  # Convert sequence to tensor and add to list
        label_raw.append(data[i][1])          # Add label to list

    # Pad sequences with padding value 0
    padded_sequences = pad_sequence(xraw, batch_first=True, padding_value=0)

    # Convert labels to a tensor
    labels_tensor = torch.tensor(np.array(label_raw))

    return padded_sequences, labels_tensor
