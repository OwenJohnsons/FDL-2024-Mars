import numpy as np 
import pandas as pd 
from glob import glob
import os
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import Pool, cpu_count

def create_file_map(database_path):
    """
    Function Purpose: Create a DataFrame to map sample IDs to their file paths

    Args:
        database_path: Path to the directory containing the HDF files

    Returns:
        file_map_df: DataFrame with columns 'path' and 'id'
    """
    hdf_files = glob(os.path.join(database_path, '**', '*.hdf'), recursive=True)
    
    print('Number of HDF files found:', len(hdf_files))  # Debugging statement
    
    data = []
    for path in hdf_files:

        sample_id = os.path.basename(path).replace('.hdf', '')
        if sample_id == 'S0401' or sample_id == '25048': 
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
    with pd.HDFStore(file_path, 'r') as store:
        data_df = store.get('data')
        
    amu_array = []
    abundance_array = []

    if file_path.split('/')[-1][0] == 'S':
        time_groups = data_df.groupby('time')
        for time, group in time_groups:
            amu_array.append(group['m/z'].values)
            abundance_array.append(group['abundance'].values)
    else:
        idxes = np.where(data_df['m/z'].values[:-1] > data_df['m/z'].values[1:])[0].tolist()
        idxes = [0] + idxes + [len(data_df)]
        for i in range(len(idxes) - 1):
            amu_array.append(data_df['m/z'].values[idxes[i]:idxes[i + 1]])
            abundance_array.append(data_df['abundance'].values[idxes[i]:idxes[i + 1]])

    amu_array = np.array(amu_array)
    abundance_array = np.array(abundance_array)
    result_array = np.stack((amu_array, abundance_array), axis=1)
    current_length = result_array.shape[0]

    if current_length < max_length:
        padding = np.zeros((max_length - current_length, 2, result_array.shape[2]))
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

def process_sample(args):
    sample_id, file_map, max_length = args
    data_2D = []; idv_sample_id_array = []
    paths = file_map[file_map['id'] == sample_id]['path'].values
    for path in paths:
        spectra_data = load_spectra(path, max_length)
        flattened_spectra = flatten_spectra(spectra_data)
        data_2D.append(flattened_spectra)
        idv_sample_id_array.append(sample_id)

    return np.concatenate(data_2D, axis=0), idv_sample_id_array

def load_data_set(ids, file_map, max_length, num_workers=None):
    """
    Function Purpose: Load the data set for a given set of sample IDs into a [n, (samples * channels * spectra)]

    Args:
        ids: List of sample IDs for training or testing 
        file_map: DataFrame mapping sample IDs to file paths
        max_length: Length of the longest sample in the dataset
        num_workers: Number of worker processes for parallel processing

    Returns:
        An array that contains the spectra data for the given sample IDs [n, (samples * channels * spectra)]
    """
    data_2D = []
    ids_in_data = []

    with ProcessPoolExecutor(max_workers=num_workers or os.cpu_count()) as executor:
        futures = {executor.submit(process_sample, (sample_id, file_map, max_length)): sample_id for sample_id in ids}
        for future in tqdm(as_completed(futures), total=len(futures)):
            try:
                result, sample_id = future.result()
                data_2D.append(result)
                ids_in_data.append(sample_id)
            except Exception as e:
                print(f"Error processing sample {futures[future]}: {e}")

    data_2D = np.concatenate(data_2D, axis=0)
    ids_in_data = list(ids_in_data)
    ids_in_data = [item for sublist in ids_in_data for item in sublist]

    return data_2D, ids_in_data

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