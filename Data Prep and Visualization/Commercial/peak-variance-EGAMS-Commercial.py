''' 
Author: Owen A. Johnson 
Code Purpose: Animate Commercial EGAMS data from an imported CSV file.
Date: 
'''
#%%
from glob import glob 
import numpy as np
import matplotlib.pyplot as plt
import smplotlib
# import scienceplots; plt.style.use(['science', 'ieee'])
import pandas as pd
from scipy.signal import find_peaks
from matplotlib.animation import FuncAnimation
from tqdm import tqdm 

# --- args --- 
plotting = False
verbose = False
min_mass = 10
max_mass = 200

# --- Load Data ---
csv_lits = glob('EGAMS.csvs/*/S*.csv')
print('Number of CSV files:', len(csv_lits))
for file in tqdm(csv_lits):
    df = pd.read_csv(file); df = df.dropna()

    mass = df['m/z']
    mask = (min_mass <= mass) & (mass <= max_mass)

    filtered_df = df[mask]

    intensity = filtered_df['abundance']; time = filtered_df['time']; mass = filtered_df['m/z']
    intensity = intensity/np.max(intensity)
    time_diff = np.diff(time)

    if verbose: print('Min Mass:', min(mass), 'Max Mass:', max(mass)) 

    file = file.split('/')[-1][0:-4]

    # - Find indexes for non-zero time differences - 
    non_zero_time_diff = np.where(time_diff != 0)[0]
    start_idx = 0; counter = 1

    # --- Individual Plots --- 

    if plotting: True
    for idx in non_zero_time_diff:
        intensity_slice = intensity[start_idx:idx]; mass_slice = mass[start_idx:idx]
        intensity_slice = intensity_slice / max(intensity_slice) # Normalise the intensity values

        # convert to arrays 
        intensity_slice = np.array(intensity_slice); mass_slice = np.array(mass_slice)

        peaks = find_peaks(intensity_slice, height=0.1)[0]
        max_peak_mass = mass_slice[np.argmax(intensity_slice)]

        if plotting:
            fig, ax = plt.subplots(figsize=(6, 3), dpi=200)
            ax.set_xlim(10, 200)
            ax.set_ylim(0, 1)
            ax.set_xlabel('Mass [m/z]')
            ax.set_ylabel('Normalised Intensity')
            ax.set_title('Time Stamp: {} (s)'.format(np.array(time)[start_idx]))

            ax.plot(mass_slice, intensity_slice, color='black')
            ax.plot(mass_slice[peaks], intensity_slice[peaks], 'x', color='red')

            ax.annotate('Max Peak Mass: {:.2f}'.format(max_peak_mass), xy=(0.5, 0.9), xycoords='axes fraction')

            # plt.savefig('EGAMS-Commercial-Mass-Spec-Plots/%s-%s.png' % (file, counter))
            plt.show()

        start_idx = idx; counter += 1

    # --- Animation ---
    fig, ax = plt.subplots(figsize=(6, 3), dpi=200)
    ax.set_xlim(10, 200)
    ax.set_ylim(0, 1)
    ax.set_xlabel('Mass [m/z]')
    ax.set_ylabel('Normalised Intensity')
    line, = ax.plot([], [], color='black')
    peaks_plot, = ax.plot([], [], 'x', color='red')
    title = ax.text(0.5, 1.05, '', transform=ax.transAxes, ha='center')
    annotation = ax.annotate('', xy=(0.5, 0.9), xycoords='axes fraction')

    start_idx = 0
    counter = 0

    def init():
        line.set_data([], [])
        peaks_plot.set_data([], [])
        title.set_text('')
        annotation.set_text('')
        return line, peaks_plot, title, annotation

    def update(frame):
        global start_idx, counter
        
        idx = non_zero_time_diff[frame]
        intensity_slice = intensity[start_idx:idx]
        mass_slice = mass[start_idx:idx]
        # intensity_slice = intensity_slice / max(intensity_slice)  # Normalise the intensity values

        # convert to arrays 
        intensity_slice = np.array(intensity_slice)
        mass_slice = np.array(mass_slice)

        peaks = find_peaks(intensity_slice, height=0.1)[0]
        max_peak_mass = mass_slice[np.argmax(intensity_slice)]

        line.set_data(mass_slice, intensity_slice)
        peaks_plot.set_data(mass_slice[peaks], intensity_slice[peaks])
        title.set_text('Time Stamp: {} (s)'.format(np.array(time)[start_idx]))
        annotation.set_text('Max Peak Mass: {:.2f}'.format(max_peak_mass))

        start_idx = idx
        counter += 1

        return line, peaks_plot, title, annotation

    ani = FuncAnimation(fig, update, frames=len(non_zero_time_diff), init_func=init, blit=True)
    ani.save('EGAMS-Commercial-Mass-Spec-Gifs/%s.gif' % file, fps=24)
    # break
 
# %%
