'''
Author: Owen A. Johnson (ojohnson@tcd.ie)
Last Major Update: 2024-06-28
Code Purpose: 
'''
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import scienceplots
plt.style.use(['science', 'ieee'])
from tqdm import tqdm
import re
import imageio
from glob import glob
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

verbose = False

def extract_number(filename):
    match = re.search(r'(\d+)\.png', filename)
    if match:
        return int(match.group(1))
    return 0

def flatten_data(indv_df):
    flat_overall_time = np.concatenate(indv_df['flat_overall_time'].values)
    flat_overall_pyro = np.concatenate(indv_df['flat_overall_pyro'].values)
    flat_overall_counts = np.concatenate(indv_df['flat_overall_counts'].values)
    flat_overall_amu = np.concatenate(indv_df['flat_overall_amu'].values)
    return flat_overall_time, flat_overall_pyro, flat_overall_counts, flat_overall_amu

def create_plot(data, eid, sample, description, labels, unsure, peaks, spectra_total, i, flat_overall_time, flat_overall_pyro):
    time = data['time']
    amu = data['amu']
    counts = data['counts']
    pyro_temp = data['pryro_temp']

    fig = plt.figure(constrained_layout=False, figsize=(18, 3))
    gs = fig.add_gridspec(1, 3)

    ax1 = fig.add_subplot(gs[0, 0])
    amu_space = np.linspace(10, 150, len(counts))
    ax1.plot(amu_space, counts, color='blue')  # overall normalization
    ax1.set_xlabel('Atomic Mass Unit [AMU]')
    ax1.set_ylabel('Max Peak Value')
    ax1.set_xlim(10, 150)
    ax1.set_ylim(0, 1)

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(flat_overall_time, flat_overall_pyro, color='blue')
    temp_pad = np.linspace(np.min(flat_overall_time), np.max(flat_overall_time), spectra_total)
    ax2.axvline(x=temp_pad[i], color='red', linestyle='--')

    ax2.set_xlabel('Time [s]')
    ax2.set_ylabel('Pyrolysis Temperature [°C]')

    # --- Annotations ---
    ax1.annotate(r'$\underline{\textbf{EID:}}$ %s' % eid, xycoords='axes fraction', xy=(0, 1.2), fontsize=12)
    ax1.annotate(r'$\underline{\textbf{Sample:}}$ %s' % sample, xycoords='axes fraction', xy=(0, 1.1), fontsize=12)
    ax2.annotate(r'$\underline{\textbf{Note:}}$ %s' % description, xycoords='axes fraction', xy=(0, 1.1), fontsize=12)
    ax2.annotate(r'$\underline{\textbf{Labels:}}$', xycoords='axes fraction', xy=(1.1, 0.9), fontsize=12)
    for j, label in enumerate(labels):
        if label != 'None':
            ax2.annotate(label, xycoords='axes fraction', xy=(1.1, (.8 - j * 0.1)), fontsize=12)

    if len(unsure) > 0:
        ax2.annotate(r'$\underline{\textbf{Unsure Labels:}}$', xycoords='axes fraction', xy=(1.5, 0.9), fontsize=12)
        for k, unsure_label in enumerate(unsure):
            ax2.annotate(unsure_label, xycoords='axes fraction', xy=(1.5, (0.8 - k * 0.1)), fontsize=12)

    return fig, ax1, ax2

def save_plot(fig, output_dir, eid, i):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    fig.savefig(f'{output_dir}/{eid}_{i}.png', dpi=200, bbox_inches='tight')
    plt.close(fig)

def main():
    df = pd.read_hdf('data-prep/PDS_EGAMS_H5_files/EGAMS_PDS_Data_AMU;10-150_IndvNorm;Final.h5')

    unique_files = df['Filename'].unique()  # find unique files in the dataset
    logger.info(f'Number of unique files: {len(unique_files)}')

    # Extract the overall data once
    df['flat_overall_time'] = df['Data'].apply(lambda x: x['time'])
    df['flat_overall_pyro'] = df['Data'].apply(lambda x: x['pryro_temp'])
    df['flat_overall_counts'] = df['Data'].apply(lambda x: x['counts'])
    df['flat_overall_amu'] = df['Data'].apply(lambda x: x['amu'])

    for k, file in enumerate(unique_files):
        logger.info(f'Processing file {k+1} out of {len(unique_files)}...')
        indv_df = df[df['Filename'] == file]  # get the individual file data
        eid = indv_df['Sample ID'].iloc[0]  # get the EID of the file

        # Flatten lists once for all rows in indv_df
        flat_overall_time, flat_overall_pyro, flat_overall_counts, flat_overall_amu = flatten_data(indv_df)

        if verbose:
            logger.debug('''\n Shapes of flattened data \n''')
            logger.debug(flat_overall_time.shape)
            logger.debug(flat_overall_pyro.shape)
            logger.debug(flat_overall_counts.shape)
            logger.debug(flat_overall_amu.shape)

        overall_peaks_idxs = indv_df['Max Peak AMU']
        maxpeaks_time_points = [time[0] for time in indv_df['flat_overall_time']]

        data = indv_df.iloc[0]['Data']

        labels = [key for key, value in indv_df.iloc[0]['Labels'].items() if value == 1]
        unsure = [key for key, value in indv_df.iloc[0]['Unsure'].items() if value == 1]
        sample = indv_df.iloc[0]['Sample']
        description = indv_df.iloc[0]['Description']
        peaks = indv_df.iloc[0]['Peak Index']
        spectra_total = len(indv_df)

        for i in tqdm(range(indv_df.shape[0])):

            # check if plot has already been created
            if os.path.exists(f'EGAMS-Mars-Mass-Spec-Gifs/{eid}/{eid}_{i}.png'):
                continue
            else:
                data = indv_df.iloc[i]['Data']
                counts = data['counts']
                amu_space = np.linspace(10, 150, len(counts))

                # --- Data Manipulation ---
                fig, ax1, ax2 = create_plot(data, eid, sample, description, labels, unsure, peaks, spectra_total, i, flat_overall_time, flat_overall_pyro)

                output_dir = f'EGAMS-Mars-Mass-Spec-Gifs/{eid}'
                save_plot(fig, output_dir, eid, i)

                mp4_frame_dir = f'EGAMS-Mars-Mass-Spec-MP4/{eid}'
                if not os.path.exists(mp4_frame_dir):
                    os.makedirs(mp4_frame_dir)

                plt.figure()
                plt.plot(amu_space, counts, color='black')
                plt.xlim(10, 150); plt.ylim(0, 1)
                plt.savefig(f'{mp4_frame_dir}/{eid}_{i}.png', dpi=300, bbox_inches='tight')
                plt.close()

if __name__ == "__main__":
    main()