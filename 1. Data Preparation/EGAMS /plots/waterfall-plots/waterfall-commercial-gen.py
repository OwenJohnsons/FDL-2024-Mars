'''
Author: Owen A. Johnson
Last Major Update: 2024-07-17
Code Purpose: Generating waterfall plots from EGAMS augmented data for model training. 
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
        fig, ax = plt.subplots(figsize=(6, 6), constrained_layout=True, dpi=200)  # Owen standard
        time_ind = df_aug["time"].unique()
        mz_values = df_aug["m/z"].unique()

        # Create a 2D array for intensity values
        intensity = np.zeros((len(time_ind), len(mz_values)))

        # Fill the intensity array
        for i, t in enumerate(time_ind):
            spectrum = df_aug[df_aug["time"] == t]
            intensity[i, :] = spectrum.iloc[:, -1]

        # Plot the heatmap
        c = ax.pcolormesh(mz_values, time_ind, intensity, shading='auto', cmap='viridis')
        ax.set_xlim((10, 150))
        plt.axis('off')
        # ax.set_xlabel("m/z")
        # ax.set_ylabel("Time")

        # Add a colorbar
        # cbar = fig.colorbar(c, ax=ax)
        # cbar.set_label('Intensity')

        plt.savefig(output_path)
        plt.close(fig)

    except Exception as e:
        logger.error(f"Error creating waterfall plot for {output_path}: {e}")
        return f"Error creating waterfall plot for {output_path}: {e}"


def process_file(h5):
    start_time = time.time()
    try:
        df_aug = pd.read_hdf(h5)
        output_path = h5.replace('.hdf', '.png')
        output_path = output_path.replace('bucket/samurai_data_base', 'waterfall_plots')

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
    h5_fnames = glob('/home/owenj/bucket/samurai_data_base/*/S*.hdf')
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