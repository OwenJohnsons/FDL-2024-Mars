'''
Author: Owen A. Johnson and Pugazhenthi Sivasankar 
Last Major Update: 2024-07-17
Code Purpose: Generating .mp4 files from EGAMS augmented data for model training. 
'''

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, FFMpegWriter
from glob import glob
import scienceplots
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
import logging
import os

# Configure logging
log_file = f'./logs/EGAMS_Augment_MP4_generation_{time.strftime("%Y-%m-%d_%H-%M-%S")}.log'
error_log_file = f'./logs/EGAMS_Augment_MP4_generation_errors_{time.strftime("%Y-%m-%d_%H-%M-%S")}.log'

logging.basicConfig(level=logging.INFO, format='%(asctime)s:%(levelname)s:%(message)s',
                    handlers=[logging.FileHandler(log_file), logging.FileHandler(error_log_file)])

logger = logging.getLogger()

plt.style.use(['science', 'ieee', 'no-latex'])

# Suppress specific matplotlib messages
logging.getLogger('matplotlib.animation').setLevel(logging.WARNING)

# Function to update the plot
def update(i, df_aug, time_ind, ax):
    ax.clear()
    ax.plot(df_aug[df_aug["time"] == time_ind[i]]["m/z"], df_aug[df_aug["time"] == time_ind[i]].iloc[:, -1], "-k")
    ax.set_xlim((10, 150))
    ax.set_ylim((0, 1))

# Function to process a single file
def process_file(h5):
    start_time = time.time()
    try:
        df_aug = pd.read_hdf(h5)
        output_path = h5.replace('.hdf', '.mp4')

        if os.path.exists(output_path):
            process_time = time.time() - start_time
            return f"File {output_path} already exists, processing time: {process_time:.2f} seconds"
        
        time_ind = df_aug["time"].unique()

        fig, ax = plt.subplots(figsize=(4, 3), constrained_layout=True, dpi=200)  # Owen standard
        ani = FuncAnimation(fig, update, fargs=(df_aug, time_ind, ax), frames=np.arange(len(time_ind)), repeat=False)
        ani.save(output_path, writer=FFMpegWriter(fps=24, metadata={'artist':'Me'}))
        plt.close(fig)
    except Exception as e:
        logger.error(f"Error processing file {h5}: {e}")
        process_time = time.time() - start_time
        return f"Error processing file {h5}: {e}, processing time: {process_time:.2f} seconds"

    process_time = time.time() - start_time
    logger.info(f"Processed file {output_path}, processing time: {process_time:.2f} seconds")
    return f"Processed file {output_path}, processing time: {process_time:.2f} seconds"

def main():
    h5_fnames = glob('./samurai_data_base/*/S*.hdf')
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
            print(f'Completed: {percentage_completed:.2f}%')
            if future.exception() is not None:
                logger.error(f"Exception: {future.exception()}")

    execution_time = time.time() - start_time
    logger.info(f"Total execution time: {execution_time:.2f} seconds")

if __name__ == '__main__':
    main()