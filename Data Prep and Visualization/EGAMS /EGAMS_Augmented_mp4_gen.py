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
from tqdm import tqdm
import time
import logging

# Configure logging
log_file = 'EGAMS_Augment_MP4_generation_%s.log' % time.strftime('%Y-%m-%d_%H-%M-%S')
logging.basicConfig(filename=log_file, level=logging.ERROR, format='%(asctime)s:%(levelname)s:%(message)s')

plt.style.use(['science', 'ieee', 'no-latex'])

# Function to update the plot
def update(i, df_aug, time_ind, ax):
    ax.clear()
    ax.plot(df_aug[df_aug["time"] == time_ind[i]]["m/z"], df_aug[df_aug["time"] == time_ind[i]].iloc[:, -1], "-k")
    ax.set_xlim((10, 150))
    ax.set_ylim((0, 1))

# Function to process a single file
def process_file(h5):
    try:
        df_aug = pd.read_hdf(h5)
        output_path = h5.replace('.h5', '.mp4')

        # Get the unique time indices
        time_ind = df_aug["time"].unique()

        # Set up the figure and axis
        fig, ax = plt.subplots(figsize=(4, 3), constrained_layout=True, dpi=200)  # Owen standard

        ani = FuncAnimation(fig, update, fargs=(df_aug, time_ind, ax), frames=np.arange(len(time_ind)), repeat=False)
        ani.save(output_path, writer=FFMpegWriter(fps=24))

        plt.close(fig)
    except Exception as e:
        logging.error(f"Error processing file {h5}: {e}")
        return f"Error processing file {h5}: {e}"

def main():
    # Load the data
    h5_fnames = glob('./samurai_data_base/*/*.h5')
    print('Number of h5 files:', len(h5_fnames))

    # Parallelize the processing of files with a progress bar
    start_time = time.time()

    with ProcessPoolExecutor() as executor:
        futures = {executor.submit(process_file, h5): h5 for h5 in h5_fnames}
        for future in tqdm(as_completed(futures), total=len(futures)):
            if future.exception() is not None:
                logging.error(f"Exception: {future.exception()}")

    end_time = time.time()
    execution_time = end_time - start_time

    print(f"Total execution time: {execution_time:.2f} seconds")

if __name__ == '__main__':
    main()
