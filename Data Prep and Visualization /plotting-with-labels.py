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

# --- Preamble ---
verbose = True 

# --- Load the Data ---
results_df = pd.read_csv('PDS_Data_Labelled.csv')
txt_files = glob('PDS_data/SAM_PDS/*/*.txt')

# --- Plotting the Data ---
for i in tqdm(range(0, len(results_df))):

    row = results_df.iloc[i]
    data = row['Data']; filename = row['Filename']; label = row['Labels']
    counts = data['counts']; amu = data['amu']; times = data['time']; pyro_temp = data['pryro_temp']; col_temp = data['col_temp']; location = row['Location']; unsure = row['Unsure']

    # --- Check if there there are NaN values in the data --- 
    if np.isnan(counts).any() or np.isnan(amu).any() or np.isnan(times).any():
        print('NaN values present!')
        print('Number of NaN values in counts:', np.sum(np.isnan(counts)))
        print('Number of NaN values in amu:', np.sum(np.isnan(amu)))
        print('Number of NaN values in times:', np.sum(np.isnan(times)))

        # - Mask the NaN values -
        mask = ~np.isnan(counts) & ~np.isnan(amu) & ~np.isnan(times)
        counts = counts[mask]; amu = amu[mask]; times = times[mask]; pyro_temp = pyro_temp[mask]; col_temp = col_temp[mask]
        
        times = times - times[0] # Setting the time to start at 0 

        plt.figure(figsize=(3, 3))
        plt.plot(times, pyro_temp, color='red')
        plt.xlabel('Time [s]'); plt.ylabel('Pyrolysis Temperature [°C]')
        plt.title(filename)
        plt.savefig(f'./output/EGAMS_raw_plots/pyro-plots/{filename}_pyro_temp.png')
        plt.close()

        plt.figure(figsize=(3, 3))
        plt.plot(times, col_temp, color='red')
        plt.xlabel('Time [s]', fontsize=16); plt.ylabel('Column Temperature [°C]', fontsize=16)
        plt.title(filename)
        plt.savefig(f'./output/EGAMS_raw_plots/coltemp-plots/{filename}_column_temp.png')
        plt.close()

        if verbose == True: 
            print('Time Shape:', times.shape)
            print('Counts Shape:', counts.shape)
            print('AMU Shape:', amu.shape)

        normalised_counts = counts / np.max(counts)

        sorted_amu = np.argsort(amu); amu = amu[sorted_amu]; normalised_counts = normalised_counts[sorted_amu]
        peaks = find_peaks(normalised_counts, height=0.15)[0]

        if len(peaks) == 0:
            print('No peaks found for:', filename)
            
        else: 
            min_amu = amu[peaks[0]]; max_amu = amu[peaks[-1]]
            mask = (amu > min_amu) & (amu < max_amu)
            normalised_counts = normalised_counts[mask]; amu = amu[mask]

        plt.figure(figsize=(10, 5))
        plt.plot(amu, normalised_counts, label=label)
        plt.xlabel('Atomic Mass Unit', fontsize=16); plt.ylabel(' Normalised Counts [s$^{-1}$]', fontsize=16)
        plt.title(filename)

        # - Annotating Plots -
        plt.annotate(r'$\underline{\textbf{Location:}}$ %s' % location, xy=(1.01, 0.95), xycoords='axes fraction')
        plt.annotate(r'$\underline{\textbf{Labels:}}$', xy=(1.01, 0.9), xycoords='axes fraction')

        for i, label in enumerate(label):
            y_position = 0.85 - i * 0.05  # Adjust the spacing as needed
            plt.annotate(label, xy=(1.01, y_position), xycoords='axes fraction')

        if len(unsure) != 0:
            label_len = len(label)
            unsure_ypos = 0.9 - (label_len * 0.05)
            plt.annotate(r'$\underline{\textbf{Unsure:}}$', xy=(1.01, unsure_ypos), xycoords='axes fraction')
            
            for i, unsure_label in enumerate(unsure):
                y_position = unsure_ypos - (i + 1) * 0.05
                plt.annotate(unsure_label, xy=(1.01, y_position), xycoords='axes fraction')
        
        # - Save the plot -
        plt.savefig(f'./output/EGAMS_raw_plots/ms-plots/{filename}_EGAMS_mass_spectra.png')
        plt.show()
        plt.close()

        # --- Combined Plot --- 
        fig = plt.figure(figsize=(9, 6))
        fig.suptitle(f'Combined Plots - {filename}')
        gs = gridspec.GridSpec(2, 2, height_ratios=[1, 2])

        # Pyrolysis Temperature Plot
        ax1 = plt.subplot(gs[0, 0])
        ax1.plot(times, pyro_temp, color='red')
        ax1.set_xlabel('Time [s]')
        ax1.set_ylabel('Pyrolysis Temperature [°C]')

        # Column Temperature Plot
        ax2 = plt.subplot(gs[0, 1])
        ax2.plot(times, col_temp, color='red')
        ax2.set_xlabel('Time [s]')
        ax2.set_ylabel('Column Temperature [°C]')

        # Mass Spectra Plot spanning both columns
        ax3 = plt.subplot(gs[1, :])
        ax3.plot(amu, normalised_counts)
        ax3.set_xlabel('Atomic Mass Unit')
        ax3.set_ylabel('Normalised Counts [s$^{-1}$]')

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(f'./output/EGAMS_raw_plots/combined-plots/{filename}_combined_plot.png')
        plt.close()