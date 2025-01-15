import numpy as np
import pandas as pd
from glob import glob
import os
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import time

def create_file_map(database_path):
    hdf_files = glob(os.path.join(database_path, '**', '*.hdf'), recursive=True)
    
    data = []
    for path in hdf_files:
        sample_id = os.path.basename(path).replace('.hdf', '')
        data.append({'path': path, 'id': sample_id})
    
    file_map_df = pd.DataFrame(data)
    file_map_df['path'] = file_map_df['path'].astype(str)
    file_map_df['id'] = file_map_df['id'].astype(str)
    return file_map_df

def get_sample_length(path):
    data_df = pd.read_hdf(path)
    return len(data_df["time"].unique())

def longest_sample(file_map_df):
    max_length = 0
    workers = os.cpu_count()

    with ProcessPoolExecutor(max_workers=int(workers)) as executor:
        results = list(tqdm(executor.map(get_sample_length, file_map_df['path']), total=len(file_map_df)))

    max_length = max(results)

    return max_length

def load_spectra(file_path, max_length):
    with pd.HDFStore(file_path, 'r') as store:
        data_df = store.get('data')
        
    amu_array = []
    abundance_array = []
    target_length = max_length  # Set a target length for all segments

    if 'S' not in file_path.split('/')[-1]:
        idxes = np.where(data_df['m/z'].values[:-1] > data_df['m/z'].values[1:])[0].tolist()
        idxes = [0] + idxes + [len(data_df)]
        for i in range(len(idxes) - 1):
            amu_segment = data_df['m/z'].values[idxes[i]:idxes[i + 1]]
            abundance_segment = data_df['abundance'].values[idxes[i]:idxes[i + 1]]
            
            # Handle inconsistency by padding or truncating
            if len(amu_segment) < target_length:
                amu_segment = np.pad(amu_segment, (0, target_length - len(amu_segment)), 'constant')
                abundance_segment = np.pad(abundance_segment, (0, target_length - len(abundance_segment)), 'constant')
            elif len(amu_segment) > target_length:
                amu_segment = np.interp(np.linspace(0, len(amu_segment), target_length), np.arange(len(amu_segment)), amu_segment)
                abundance_segment = np.interp(np.linspace(0, len(abundance_segment), target_length), np.arange(len(abundance_segment)), abundance_segment)
            
            amu_array.append(amu_segment)
            abundance_array.append(abundance_segment)
    
    else: 
        time_groups = data_df.groupby('time')
        for time, group in time_groups:
            amu_segment = group['m/z'].values
            abundance_segment = group['abundance'].values
            
            # Handle inconsistency by padding or truncating
            if len(amu_segment) < target_length:
                amu_segment = np.pad(amu_segment, (0, target_length - len(amu_segment)), 'constant')
                abundance_segment = np.pad(abundance_segment, (0, target_length - len(abundance_segment)), 'constant')
            elif len(amu_segment) > target_length:
                amu_segment = np.interp(np.linspace(0, len(amu_segment), target_length), np.arange(len(amu_segment)), amu_segment)
                abundance_segment = np.interp(np.linspace(0, len(abundance_segment), target_length), np.arange(len(abundance_segment)), abundance_segment)
            
            amu_array.append(amu_segment)
            abundance_array.append(abundance_segment)

    amu_array = np.array(amu_array)
    abundance_array = np.array(abundance_array)

    # Ensure shapes match before stacking
    if amu_array.shape != abundance_array.shape:
        print(f"Inconsistent shapes detected: amu_array shape {amu_array.shape}, abundance_array shape {abundance_array.shape}")
        return None

    result_array = np.stack((amu_array, abundance_array), axis=1)
    current_length = result_array.shape[0]

    if current_length < max_length:
        padding = np.zeros((max_length - current_length, 2, result_array.shape[2]))
        result_array = np.concatenate((result_array, padding), axis=0)
    elif current_length > max_length:
        downsampled_amu_array = []
        downsampled_abundance_array = []
        
        for i in range(result_array.shape[1]):
            interp_func_amu = interp1d(np.linspace(0, 1, current_length), result_array[:, 0, i])
            interp_func_abundance = interp1d(np.linspace(0, 1, current_length), result_array[:, 1, i])
            downsampled_amu_array.append(interp_func_amu(np.linspace(0, 1, max_length)))
            downsampled_abundance_array.append(interp_func_abundance(np.linspace(0, 1, max_length)))
        
        result_array = np.stack((downsampled_amu_array, downsampled_abundance_array), axis=1)
        result_array = np.array(result_array).transpose(1, 2, 0)

    return result_array

def flatten_spectra(spectra_data):
    n_samples, n_channels, n_spectra = spectra_data.shape
    flattened_spectra_data = spectra_data.reshape(1, n_samples * n_channels * n_spectra)
    return flattened_spectra_data

def process_sample(args):
    sample_id, file_map, max_length = args
    data_2D = []
    idv_sample_id_array = []
    paths = file_map[file_map['id'] == sample_id]['path'].values
    for path in paths:
        spectra_data = load_spectra(path, max_length)
        flattened_spectra = flatten_spectra(spectra_data)
        data_2D.append(flattened_spectra)
        idv_sample_id_array.append(sample_id)

    return np.concatenate(data_2D, axis=0), idv_sample_id_array

def load_data_set(ids, file_map, max_length, num_workers=None, chunk_size=100):
    data_2D = []
    ids_in_data = []

    def process_chunk(chunk):
        with ProcessPoolExecutor(max_workers=num_workers or os.cpu_count()) as executor:
            futures = {executor.submit(process_sample, (sample_id, file_map, max_length)): sample_id for sample_id in chunk}
            for future in as_completed(futures):
                try:
                    result, sample_id = future.result()
                    yield result, sample_id
                except Exception as e:
                    # print(f"Error processing sample {futures[future]}: {e}")
                    pass

    # Process data in chunks
    for i in tqdm(range(0, len(ids), chunk_size), desc="Loading Data Set", unit="chunk"):
        chunk = ids[i:i + chunk_size]
        for result, sample_id in process_chunk(chunk):
            data_2D.append(result)
            ids_in_data.append(sample_id)

    data_2D_stacked = np.vstack(data_2D)
    ids_in_data = [item for sublist in ids_in_data for item in sublist]

    return data_2D_stacked, ids_in_data

def load_labels(hdf_file, needed_ids):
    df = pd.read_hdf(hdf_file)
    
    if not isinstance(df['Labels'].iloc[0], dict):
        raise ValueError("The 'Labels' column is not in the expected dictionary format.")

    needed_labels = []
    for idx in tqdm(needed_ids, desc="Loading Labels", unit="sample"):
        idx = str(idx)
    
        label_array = df[df['Sample ID'].astype(str) == idx]['Labels'].values
        if len(label_array) > 0:  # Check if label_array is not empty
            numeric_array = [int(value) for value in label_array[0].values()]
            needed_labels.append(numeric_array)
        else:
            # print(f"Warning: No labels found for Sample ID {idx}.")
            needed_labels.append([0] * len(df['Labels'].iloc[0].values()))  # Append a default label (e.g., all zeros)

    # print(df['Sample ID'].values)
    return np.array(needed_labels)