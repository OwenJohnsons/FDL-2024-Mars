'''
Author: Owen A. Johnson
Modified by: Pugazh. S
Last Major Update: 2024-07-29
Code Purpose: Generating waterfall plots from EGAMS augmented Mars PDS data for model training. 
'''

import matplotlib.pyplot as plt
import numpy as np
from glob import glob
import scienceplots
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
import logging
import os

# Configure logging
os.makedirs('./logs', exist_ok=True)
log_file = f'./logs/EGAMS_Augment_Waterfall_generation_{time.strftime("%Y-%m-%d_%H-%M-%S")}.log'
error_log_file = f'./logs/EGAMS_Augment_Waterfall_generation_errors_{time.strftime("%Y-%m-%d_%H-%M-%S")}.log'

logging.basicConfig(level=logging.INFO, format='%(asctime)s:%(levelname)s:%(message)s',
                    handlers=[logging.FileHandler(log_file), logging.FileHandler(error_log_file)])

logger = logging.getLogger()

plt.style.use(['science', 'ieee', 'no-latex'])

def create_waterfall_plot(df_aug, output_path):    
    try:
        # Isolate each spectra:
        indexes_with_mz_10 = df_aug.index[df_aug['m/z'] == 10].tolist()
        indexes_with_mz_150 = df_aug.index[df_aug['m/z'] == 150].tolist()

        # Create an m/z array:
        mz_values = np.linspace(10, 150, 150-10+1)

        # Work around for the troublesome file:
        file_name = output_path.split("/")[-1]
        if file_name == '25048.png':
            print(f'File with issue: {file_name}')
            indexes_with_mz_150 = df_aug.index[df_aug['m/z'] == 149].tolist()
            mz_values = np.linspace(10, 149, 149-10+1)

        # Create a time array:
        time_ind = np.r_[0:len(indexes_with_mz_10)]

        # Create a 2D array for intensity values:
        intensity = np.zeros((len(time_ind), len(mz_values)))

        # Create the figure lay-out:
        fig, ax = plt.subplots(figsize=(6, 6), constrained_layout=True, dpi=200)  # Owen standard

        # Fill the intensity array
        for i in range(0, len(indexes_with_mz_10)):    
            intensity[i, :] = df_aug.iloc[indexes_with_mz_10[i]:indexes_with_mz_150[i]+1, -1].values

        # Plot the heatmap
        c = ax.pcolormesh(mz_values, time_ind, intensity, shading='auto', cmap='viridis')
        ax.set_xlim((10, 150))
        plt.axis('off')
        plt.savefig(output_path)
        plt.close(fig)

    except Exception as e:
        logger.error(f"Error creating waterfall plot for: {e}")
        return f"Error creating waterfall plot for : {e}"


def process_file(h5):
    start_time = time.time()
    try:
        df_aug = pd.read_hdf(h5)
        output_path = h5.replace('.hdf', '.png')
        output_path = output_path.replace('pugal_data_tree/samurai_data_base', 'waterfall_plots-mars')

        if os.path.exists(output_path):
            process_time = time.time() - start_time
            return f"File {output_path} already exists, processing time: {process_time:.2f} seconds"

        else:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            create_waterfall_plot(df_aug, output_path)
        
    except Exception as e:
        logger.error(f"Error processing file {h5}: {e}")
        process_time = time.time() - start_time
        return f"Error processing file {h5}: {e}, processing time: {process_time:.2f} seconds"

    process_time = time.time() - start_time
    logger.info(f"Processed file {output_path}, processing time: {process_time:.2f} seconds")
    return f"Processed file {output_path}, processing time: {process_time:.2f} seconds"

def main():
    h5_fnames = glob('/home/jupyter/pugal_data_tree/samurai_data_base/*/2*.hdf')  
    
    
    logger.info('Number of h5 files: %d', len(h5_fnames))

    start_time = time.time()

    with ProcessPoolExecutor() as executor:
        num_workers = executor._max_workers
        logger.info(f"Running with {num_workers} parallel processes")
        futures = {executor.submit(process_file, h5): h5 for h5 in h5_fnames}
        total_files = len(futures)
        completed_files = 0
        
        for future in as_completed(futures):
            result = future.result()
            logger.info(result)
            completed_files += 1
            percentage_completed = (completed_files / total_files) * 100
            process_time = result.split("processing time: ")[-1]
            print(f'Completed: {percentage_completed:.2f}%, Time for last file: {process_time}')
            if future.exception() is not None:
                logger.error(f"Exception: {future.exception()}")

    execution_time = time.time() - start_time
    logger.info(f"Total execution time: {execution_time:.2f} seconds")

if __name__ == '__main__':
    main()
