#%%
'''
Author: Owen A. Johnson 
Date: 2024/06/18
Code Purpose: To plot MARS Mass Spectrometry data with classified label for use in FDL 2024 Mars challenge. 
'''

import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import scienceplots; plt.style.use(['science', 'ieee'])
import pandas as pd 
import os 
from scipy.signal import find_peaks
from tqdm import tqdm
from glob import glob
import ast

# --- Preamble ---
verbose = True 

# --- Load the Data ---
results_df = pd.read_csv('PDS_Data_Labelled.csv')
txt_files = glob('PDS_data/SAM_PDS/*/*.txt')

# --- Plotting the Data ---
for i in tqdm(range(0, len(results_df))):

    row = results_df.iloc[i]
    filename = row['Filename']; label = row['Labels']; location = row['Location']; unsure = row['Unsure']

    # --- Match file paths and load data --- 
    for txt_file in txt_files:
        if filename in txt_file:
            data_path = txt_file

    times, amu, pyro_temp, col_temp, counts = np.loadtxt(data_path, unpack=True, skiprows=1, delimiter=',')
        
    times = times - times[0] # Setting the time to start at 0 

    # --- Converting strings to dictionaries --- 
    label = ast.literal_eval(label); unsure = ast.literal_eval(unsure)
    labels_string  = [key for key, value in label.items() if value == 1]
    unsure_string = [key for key, value in unsure.items() if value == 1]
    

    if verbose == True: 
        print('Time Shape:', times.shape)
        print('Counts Shape:', counts.shape)
        print('AMU Shape:', amu.shape)
        print('Pyrolysis Temperature Shape:', pyro_temp.shape)
        print('Column Temperature Shape:', col_temp.shape)

    normalised_counts = counts / np.max(counts)

    sorted_amu = np.argsort(amu); amu = amu[sorted_amu]; normalised_counts = normalised_counts[sorted_amu]
    peaks = find_peaks(normalised_counts, height=0.15)[0]

    if len(peaks) == 0:
        print('No peaks found for:', filename)
        
    else: 
        min_amu = amu[peaks[0]]; max_amu = amu[peaks[-1]]
        mask = (amu > min_amu) & (amu < max_amu)
        normalised_counts = normalised_counts[mask]; masked_amu = amu[mask]

    plt.figure(figsize=(10, 5))
    plt.plot(masked_amu, normalised_counts, label=label)
    plt.xlabel('Atomic Mass Unit', fontsize=16); plt.ylabel(' Normalised Counts [s$^{-1}$]', fontsize=16)
    plt.title(filename)

    # - Annotating Plots -
    plt.annotate(r'$\underline{\textbf{Location:}}$ %s' % location, xy=(1.01, 0.95), xycoords='axes fraction')
    plt.annotate(r'$\underline{\textbf{Labels:}}$', xy=(1.01, 0.9), xycoords='axes fraction')

    for i, label in enumerate(labels_string):
        y_position = 0.85 - i * 0.05  # Adjust the spacing as needed
        plt.annotate(label, xy=(1.01, y_position), xycoords='axes fraction')

    if len(unsure_string) != 0:
        label_len = len(label)
        unsure_ypos = 0.9 - (label_len * 0.05)
        plt.annotate(r'$\underline{\textbf{Unsure:}}$', xy=(1.01, unsure_ypos), xycoords='axes fraction')
        
        for i, unsure_label in enumerate(unsure_string):
            y_position = unsure_ypos - (i + 1) * 0.05
            plt.annotate(unsure_label, xy=(1.01, y_position), xycoords='axes fraction')
    
    # - Save the plot -
    plt.savefig(f'./output/EGAMS_raw_plots/ms-plots/{filename}_EGAMS_mass_spectra.png')
    # plt.show()
    plt.close()

    # --- Combined Plot --- 
    fig = plt.figure(figsize=(12, 8))
    fig.suptitle(f'Combined Plots - {filename}', fontsize=16)
    gs = gridspec.GridSpec(2, 3, height_ratios=[1, 2])

    # Time vs AMU Plot
    ax0 = plt.subplot(gs[0, 0])
    ax0.plot(times, amu, color='blue')
    ax0.set_xlabel('Time [s]')
    ax0.set_ylabel('Atomic Mass Unit')

    # Pyrolysis Temperature Plot
    ax1 = plt.subplot(gs[0, 1])
    ax1.plot(times, pyro_temp, color='red')
    ax1.set_xlabel('Time [s]')
    ax1.set_ylabel('Pyrolysis Temperature [°C]')

    # Column Temperature Plot
    ax2 = plt.subplot(gs[0, 2])
    ax2.plot(times, col_temp, color='red')
    ax2.set_xlabel('Time [s]')
    ax2.set_ylabel('Column Temperature [°C]')

    # Mass Spectra Plot spanning both columns
    ax3 = plt.subplot(gs[1, :])
    ax3.plot(masked_amu, normalised_counts)
    ax3.set_xlabel('Atomic Mass Unit')
    ax3.set_ylabel('Normalised Counts [s$^{-1}$]')
    # - print labels - 
    ax3.annotate(r'$\underline{\textbf{Location:}}$ %s' % location, xy=(1.01, 0.95), xycoords='axes fraction')
    ax3.annotate(r'$\underline{\textbf{Labels:}}$', xy=(1.01, 0.9), xycoords='axes fraction')

    for i, label in enumerate(labels_string):
        y_position = 0.85 - i * 0.05
        ax3.annotate(label, xy=(1.01, y_position), xycoords='axes fraction')

    if len(unsure_string) != 0:
        label_len = len(label)
        unsure_ypos = 0.9 - (label_len * 0.05)
        ax3.annotate(r'$\underline{\textbf{Unsure:}}$', xy=(1.01, unsure_ypos), xycoords='axes fraction')
        
        for i, unsure_label in enumerate(unsure_string):
            y_position = unsure_ypos - (i + 1) * 0.05
            ax3.annotate(unsure_label, xy=(1.01, y_position), xycoords='axes fraction')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(f'./output/EGAMS_raw_plots/combined-plots/{filename}_combined_plot.png')
    # plt.show()
    plt.close()