import numpy as np 
import pandas as pd 
from glob import glob
import os
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

def create_file_map(database_path):
    """
    Function Purpose: Create a DataFrame to map sample IDs to their file paths

    Args:
        database_path: Path to the directory containing the HDF files

    Returns:
        file_map_df: DataFrame with columns 'path' and 'id'
    """
    hdf_files = glob(os.path.join(database_path, '**', 'S*.hdf'), recursive=True)
    
    print('Number of HDF files found:', len(hdf_files))  # Debugging statement
    
    data = []
    for path in hdf_files:

        sample_id = os.path.basename(path).replace('.hdf', '')
        if sample_id == 'S0401': 
            continue
        data.append({'path': path, 'id': sample_id})
    
    file_map_df = pd.DataFrame(data)
    return file_map_df

def get_sample_length(path):
    data_df = pd.read_hdf(path)
    return len(data_df["time"].unique())

def longest_sample(file_map_df):
    """
    Function Purpose: Find the longest sample in the dataset

    Args:
        file_map_df: DataFrame mapping sample IDs to file paths

    Returns:
        longest_sample_id: ID of the longest sample
    """

    max_length = 0; workers = os.cpu_count()

    with ProcessPoolExecutor(max_workers=int(workers)) as executor:
        results = list(tqdm(executor.map(get_sample_length, file_map_df['path']), total=len(file_map_df)))

    max_length = max(results)

    return max_length

def load_spectra(file_path, max_length):
    """
    Function Purpose: Load the spectra data for a given sample ID

    Args:
        sample_id: ID of the sample
        file_map_df: DataFrame mapping sample IDs to file paths

    Returns:
        concatenated_spectra_data: Concatenated and transposed spectra data array with columns 'm/z' and 'abundance'
    """

    data_df = pd.read_hdf(file_path)
    time_ind = data_df["time"].unique()

    amu_array = []; abundance_array = []

    for i in range(len(time_ind)):
        df_slice = data_df[data_df["time"] == time_ind[i]]
        amu = df_slice["m/z"].values
        abundance = df_slice["abundance"].values

        amu_array.append(amu); abundance_array.append(abundance)
    
    # 3D array with shape (n_samples, n_channels, n_timesteps)
    result_array = np.stack((amu_array, abundance_array), axis=1)

    if result_array.shape[0] != max_length:
        padding = np.zeros((max_length - result_array.shape[0], 2, result_array.shape[2]))
        result_array = np.concatenate((result_array, padding), axis=0)

    return result_array

def flatten_spectra(spectra_data):
    """
    Function Purpose: Flatten the spectra data array

    Args:
        spectra_data: 3D array with shape (n_samples, n_channels, n_timesteps)

    Returns:
        flattened_spectra_data: 2D array with shape (1, n_samples * n_channels * n_timesteps)
    """
    n_samples, n_channels, n_spectra = spectra_data.shape
    flattened_spectra_data = spectra_data.reshape(1, n_samples * n_channels * n_spectra)

    return flattened_spectra_data

def load_data_set(ids, file_map, max_length):
    """
    Function Purpose: Load the data set for a given set of sample IDs into a [n, (samples * channels * spectra)]

    Args:
        ids: List of sample IDs for training or testing 
        file_map: DataFrame mapping sample IDs to file paths
        max_length: Length of the longest sample in the dataset

    Returns:
        An array that contains the spectra data for the given sample IDs [n, (samples * channels * spectra)]
    """
    data_2D = []; ids_in_data = []

    for i in tqdm(range(len(ids))):
        sample_id = ids[i]
        paths = file_map[file_map['id'] == sample_id]['path'].values
        for path in paths:
            spectra_data = load_spectra(path, max_length)
            flattened_spectra = flatten_spectra(spectra_data)
            data_2D.append(flattened_spectra)
            ids_in_data.append(sample_id)
        
    return np.concatenate(data_2D, axis=0), ids_in_data

def load_labels(hdf_file, needed_ids):
    """
    Function Purpose: Load the labels and sample IDs from an HDF5 file

    Args:
        hdf_file: Path to the HDF5 file

    Returns:
        labels: Array of labels
        label_eids: Array of sample IDs
    """
    df = pd.read_hdf(hdf_file)
    
    if not isinstance(df['Labels'].iloc[0], dict):
        raise ValueError("The 'Labels' column is not in the expected dictionary format.")

    needed_labels = []
    for idx in needed_ids:
        label_array = df[df['Sample ID'] == idx]['Labels'].values
        numeric_array = [int(value) for value in label_array[0].values()]
        needed_labels.append(numeric_array)

    return np.array(needed_labels)