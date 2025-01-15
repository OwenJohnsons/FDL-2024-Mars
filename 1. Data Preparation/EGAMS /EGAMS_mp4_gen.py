#%% 
''' 
Code Purpose: Generating .mp4 files from EGAMS data for model training. 
Author: Owen A. Johnson
Date of Last Major Update: 01/07/2024
'''
import os
import pandas as pd 
import matplotlib.pyplot as plt
import scienceplots; plt.style.use(['science', 'ieee'])
from matplotlib.animation import FuncAnimation, FFMpegWriter

# --- Load the data ---
data = pd.read_hdf('EGAMS_PDS_Data.h5')
unique_files = data['Filename'].unique()

# --- Functions --- 
def update(frame, file_data, ax):
    ax.clear()
    ax.plot(file_data.iloc[frame]['Data']['amu'], file_data.iloc[frame]['Data']['norm_counts'])
    ax.set_xlim(10, 200)

for file in unique_files:
    file_data = data[data['Filename'] == file]
    eid = file_data.iloc[0]['Sample ID']
    # check if mp4 file exists
    if os.path.exists(f"EGAMS_PDS_MP4/{eid}_animation.mp4"):
        continue
    else:
        fig, ax = plt.subplots()
        ani = FuncAnimation(fig, update, frames=range(file_data.shape[0]), fargs=(file_data, ax), repeat=False)
        
        mp4_filename = f"EGAMS_PDS_MP4/{eid}_animation.mp4"
        ani.save(mp4_filename, writer=FFMpegWriter(fps=24))  
        plt.close(fig)

print("EGAMS PDS MP4 animations created successfully.")