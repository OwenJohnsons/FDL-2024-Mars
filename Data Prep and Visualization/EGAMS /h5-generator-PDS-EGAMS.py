#%%
'''
Author: Owen A. Johnson (ojohnson@tcd.ie)
Last Major Update: 2024-06-28
Code Purpose: To generate a HDF5 file from the PDS data intaking raw .txt files from the NASA Planetary Data System (PDS) for the Curosity Rover and then cross referencing the data with the labels provided by the EGAMS workshop carried out by Victoria Da Poian (GSFC) and Eric Lyness (GSFC).
'''

import pandas as pd
import numpy as np
from glob import glob
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import sys 
from scipy import stats

# --- Preamble --- 
verbose = False

nobelgas_EID = [25117, 25202, 25208, 25219, 25362, 25363, 25495]
combustion_EID = [25173, 25174, 25175]
blanks_EID = [25032, 25033, 25059, 25083, 25133, 25145, 25235, 2536, 25392, 25393, 25188, 25223, 25689]

# --- Load the Labels --- 
labels_df = pd.read_csv('EGAMS_workshop_labels.csv')
EGA_df = labels_df[labels_df['Mars Locations'].str.contains('EGA')] # Just EGA labels 
Chemin_df = labels_df[labels_df['Mars Locations'].str.contains('Chemin')] # Just Chemin labels
filepaths = glob('PDS_data/SAM_PDS/*/*.txt')

# --- Remove nobel gas and combustion files --- 
print('Number of files before removal:', len(filepaths))
for EID in nobelgas_EID + combustion_EID:
    filepaths = [path for path in filepaths if str(EID) not in path]
print('Number of files after removal:', len(filepaths))
print('files:', filepaths)

# --- Load Metadata for PDS --- 
datasum_df = pd.read_csv('PDS_SAM_file_summary.csv')
datasum_df = datasum_df[~datasum_df['Filename'].str.contains('lbl')] # Remove the label files
print('Number of PDS files excluding labels:', len(datasum_df))

datasum_df = datasum_df.dropna(subset=['Sample'])
QMS_datasum_df = datasum_df[datasum_df['Experiment'] == 'qms'] # Just QMS data

print('Number of QMS files with non-NaN locations:', len(QMS_datasum_df))

# --- Labels ---
mars_labels = ['carbonate', 'chloride', 'oxidized organic carbon', 'oxychlorine', 'sulfate', 'sulfide', 'nitrate', 'iron_oxide', 'phyllosilicate', 'silicate']

# --- Data Collation ---
data_frame = pd.DataFrame()
total_spectra_count = 0 

for i in range(QMS_datasum_df.shape[0]):
    print('File', i+1, 'of', QMS_datasum_df.shape[0])

    smp_loc = QMS_datasum_df.iloc[i]['Sample'].split(' ')[0] # EGM label file only uses first word
    EGA_row = EGA_df[EGA_df['Mars Locations'].str.contains(smp_loc)]

    if verbose: 
        print('=== %s ===' % i)
        print('Sample Location:', QMS_datasum_df.iloc[i]['Sample'])
        print('First word of Sample Location:', smp_loc)
        print(EGA_row['Mars Locations'])

    if smp_loc in ['GC', 'Blank', 'Cal']:
        print('Skipping...')
        continue
    else:
        labels = [key for key, value in EGA_row.iloc[0].items() if value == 'X']
        unsure = [key for key, value in EGA_row.iloc[0].items() if value == '~']

        labels_dict = {label: (1 if label in labels else 0) for label in mars_labels}
        unsure_dict = {label: (1 if label in unsure else 0) for label in mars_labels}

    file_name = QMS_datasum_df.iloc[i]['Filename']
    # - find matching filename in paths 
    filepath_idx = None
    for path in filepaths:
        if file_name in path:
            if verbose:
                print('Matched:', file_name, 'to', path)
            filepath_idx = path
            break
    if not filepath_idx:
        print('No match found for:', file_name)
        continue

    # - load the data - 
    time, amu, pryro_temp, col_temp, counts = np.loadtxt(filepath_idx, unpack=True, skiprows=1, delimiter=',')
    mask = (amu >= 10) & (amu <= 200)
    time = time[mask]; amu = amu[mask]; pryro_temp = pryro_temp[mask]; col_temp = col_temp[mask]; counts = counts[mask]
    counts = counts / np.max(counts) # Normalise the counts, overall. 

    start_idxes = np.where(amu[:-1] > amu[1:])[0].tolist() # vectorised version of the above loop.

    if verbose:
        print('=== %s ===' % file_name)
        print('AMU Entries:', len(amu))
        print('AMU Difference Mode:', stats.mode(np.diff(amu))[0])
        print('Mode Counts:', stats.mode(np.diff(amu))[1])
        print('Average difference in AMU:', np.mean(np.diff(amu)))

    if verbose: print('Loading data for:', file_name)

    if len(amu) == 0: 
        print('No data found for:', file_name)
        print('Shapes:')
        print('Time:', time.shape)
        print('AMU:', amu.shape)
        print('Pyro Temp:', pryro_temp.shape)
        print('Col Temp:', col_temp.shape)
        print('Counts:', counts.shape)
        continue

    for k in range(len(start_idxes)):
        if k == 0: 
            start_idx = 0; end_idx = start_idxes[k]
        if k + 1 < len(start_idxes):
            start_idx = start_idxes[k] + 1; end_idx = start_idxes[k+1]
        else:
            continue

        count_indv = counts[start_idx:end_idx]
        amu_indv = amu[start_idx:end_idx]

        if len(count_indv) == 0:
            print('No data found for:', file_name, 'at indexes:', start_idx, end_idx)
            exit('Exiting...')

        peaks = find_peaks(count_indv, height=0.1)[0]
        peaks_values = [count_indv[peak] for peak in peaks]
        spec_n = k

        if len(peaks) == 0:
            if verbose: print('No peaks found for indexes:', start_idx, end_idx)
            max_peak_amu = 0

        else: 
            max_peak_amu = amu_indv[np.argmax(peaks_values)]
    
        # - Data Dictionary - 
        data_dict = {
            'time': time[start_idx:end_idx].tolist(), 
            'amu': amu_indv.tolist(), 
            'pryro_temp': pryro_temp[start_idx:end_idx].tolist(), 
            'col_temp': col_temp[start_idx:end_idx].tolist(), 
            'counts': count_indv.tolist()
        }
        new_row = {
            'Sample ID': int(QMS_datasum_df.iloc[i]['EID']),  
            'Sample': QMS_datasum_df.iloc[i]['Sample'],
            'Location': QMS_datasum_df.iloc[i]['Where'],
            'Description': QMS_datasum_df.iloc[i]['DESCRIPTION'],
            'Filename': file_name,
            'Data': (data_dict),  
            'Labels': (labels_dict),  
            'Unsure': (unsure_dict),  
            'Peak Index': (peaks.tolist()),  
            'Peak Values': (peaks_values),  
            'Max Peak AMU': max_peak_amu,
            'Spectra Number': spec_n,
            'Transferred Pyrolysis Temperature': 'N/A', 
            'Blank': 1 if int(QMS_datasum_df.iloc[i]['EID']) in blanks_EID else 0
        }
        
        total_spectra_count += 1
        data_frame = data_frame.append(new_row, ignore_index=True)
    
print('Size of the data frame: %.9f Gb' % (sys.getsizeof(data_frame) / 1e9))
print('Total number of spectra:', total_spectra_count)
print('Number of rows in the data frame:', data_frame.shape[0])

# --- Save the data ---
hdf5_file = 'EGAMS_PDS_Data.h5'
data_frame.to_hdf(hdf5_file, key='EGAMS_PDS_Data', mode='w')

# # --- HTML for viewing --- 
results_df = pd.read_hdf(hdf5_file, 'EGAMS_PDS_Data')
html = results_df.to_html()
with open('EGAMS_PDS_Data.html', 'w') as f:
    f.write(html)