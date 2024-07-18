#%%
'''
Author: Owen A. Johnson
Last Major Update: 2024-07-18 
Code Purpose: Generating .mp4 files from EGAMS augmented data for model training. 
'''
import logging
import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob
from concurrent.futures import ProcessPoolExecutor, as_completed
from matplotlib.animation import FuncAnimation, FFMpegWriter
import scienceplots; plt.style.use(['science', 'ieee', 'no-latex'])

# Configure logging
log_file = 'process.log'
error_log_file = 'error.log'
logging.basicConfig(level=logging.INFO, format='%(asctime)s:%(levelname)s:%(message)s',
                    handlers=[logging.FileHandler(log_file), logging.FileHandler(error_log_file)])
logger = logging.getLogger()

plt.style.use(['science', 'ieee', 'no-latex'])

# Suppress specific matplotlib messages
logging.getLogger('matplotlib.animation').setLevel(logging.WARNING)

# Function to update the plot
def update(i, df_aug, indices, ax):
    ax.clear()
    amu_slice = df_aug['m/z'][indices[i]:indices[i+1]]
    count_slice = df_aug['abundance'][indices[i]:indices[i+1]]
    ax.plot(amu_slice, count_slice, "-k")
    ax.set_xlim((10, 150))
    ax.set_ylim((0, 1))

# Function to process and save the animation for a single file
def process_file(h5):
    start_time = time.time()
    try:
        df_aug = pd.read_hdf(h5)
        output_path = h5.replace('.hdf', '.mp4')

        if os.path.exists(output_path):
            process_time = time.time() - start_time
            return f"File {output_path} already exists, processing time: {process_time:.2f} seconds"
        
        amu = df_aug['m/z'].values 
        indices = np.where(amu[:-1] > amu[1:])[0].tolist()  # Finding the start of each spectrum, by finding where AMU decreases.
        indices = [0] + indices + [len(amu)]  # Adding the start and end indexes to the list.

        fig, ax = plt.subplots(figsize=(4, 3), constrained_layout=True, dpi=200)
        ani = FuncAnimation(fig, update, fargs=(df_aug, indices, ax), frames=np.arange(len(indices)-1), repeat=False)
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
    h5_fnames = glob('./samurai_data_base/*/2*.hdf')
    logger.info('Number of h5 files: %d', len(h5_fnames))

    start_time = time.time()

    # Use all available cores
    num_workers = os.cpu_count()
    print('Number of cores being used:', num_workers)

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        logger.info(f"Running with {num_workers} parallel processes")
        futures = {executor.submit(process_file, h5): h5 for h5 in h5_fnames}
        total_files = len(futures)
        completed_files = 0

        for future in as_completed(futures):
            try:
                result = future.result()
                logger.info(result)
                completed_files += 1
                percentage_completed = (completed_files / total_files) * 100
                process_time = result.split("processing time: ")[-1]
                print(f'Completed: {percentage_completed:.2f}%, Time for last file: {process_time}')
            except Exception as e:
                logger.error(f"Exception: {e}")
                print(f"Exception: {e}")

    execution_time = time.time() - start_time
    logger.info(f"Total execution time: {execution_time:.2f} seconds")

if __name__ == '__main__':
    main()
