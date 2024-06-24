#%% 
'''
Author: Owen A. Johnson 
Date: 2024-06-18 
Code Purpose: To extract labels assigned by the GSFC Planetary Environments Lab Workhops on the MARS Mass Spectrometry data for use in FDL 2024 Mars challenge.
'''

import pandas as pd
import numpy as np
from glob import glob
from tqdm import tqdm
import webbrowser
import matplotlib.pyplot as plt

# --- Preamble --- 
verbose = False
verybose = False

# --- Load the Labels --- 
labels_df = pd.read_csv('EGAMS_workshop_labels.csv')
EGA_df = labels_df[labels_df['Mars Locations'].str.contains('EGA')] # Just EGA labels 
Chemin_df = labels_df[labels_df['Mars Locations'].str.contains('Chemin')] # Just Chemin labels
filepaths = glob('PDS_data/SAM_PDS/*/*.txt')

if verbose:
    print('Number of EGA labels:', len(EGA_df))
    print('Number of Chemin labels:', len(Chemin_df))
    print('Number of txt files found:', len(filepaths))

# --- Load Metadata for PDS --- 
datasum_df = pd.read_csv('PDS_SAM_file_summary.csv')
datasum_df = datasum_df[~datasum_df['Filename'].str.contains('lbl')] # Remove the label files
print('Number of PDS files excluding labels:', len(datasum_df))

datasum_df = datasum_df.dropna(subset=['Sample'])
QMS_datasum_df = datasum_df[datasum_df['Experiment'] == 'qms'] # Just QMS data

print('Number of QMS files with non-NaN locations:', len(QMS_datasum_df))

# --- Labels ---
mars_labels = ['carbonate', 'chloride', 'oxidized organic carbon', 'oxychlorine', 'sulfate', 'sulfide', 'nitrate', 'iron_oxide', 'phyllosilicate', 'silicate']

# --- Extract the Labels ---
results_df = pd.DataFrame(columns=['Sample ID', 'Location', 'Filename', 'Data', 'Labels', 'Unsure'])

for i in (range(QMS_datasum_df.shape[0])):
    smp_loc = QMS_datasum_df.iloc[i]['Sample'].split(' ')[0] # EGM label file only uses first word
    EGA_row = EGA_df[EGA_df['Mars Locations'].str.contains(smp_loc)]

    if verybose: 
        print('=== %s ===' % i)
        print('Sample Location:', QMS_datasum_df.iloc[i]['Sample'])
        print('First word of Sample Location:', smp_loc)
        print(EGA_row['Mars Locations'])

    if smp_loc in ['GC', 'Blank', 'Cal', 'Poutour']:
        print('Skipping...')
        continue
    else:
        labels = [key for key, value in EGA_row.iloc[0].items() if value == 'X']
        unsure = [key for key, value in EGA_row.iloc[0].items() if value == '~']

        labels_dict = {label: (1 if label in labels else 0) for label in mars_labels}
        unsure_dict = {label: (1 if label in unsure else 0) for label in mars_labels}

    file_name = QMS_datasum_df.iloc[i]['Filename']
    # - find matching filename in paths 
    for path in filepaths:
        if file_name in path:
            if verbose:
                print('Matched:', file_name, 'to', path)
            filepath_idx = path
            break
    else:
        print('No match found for:', file_name)
        print('')

    # - load the data - 
    time, amu, pryro_temp, col_temp, counts = np.loadtxt(filepath_idx, unpack=True, skiprows=1, delimiter=',')
    print('Loading data for:', file_name)

    if len(amu) == 0: 
        print('No data found for:', file_name)
        print('Shapes:')
        print('Time:', time.shape)
        print('AMU:', amu.shape)
        print('Pyro Temp:', pryro_temp.shape)
        print('Col Temp:', col_temp.shape)
        print('Counts:', counts.shape)

    # sort the data
    sorted_amu = np.argsort(amu); amu = amu[sorted_amu]; counts = counts[sorted_amu]

    # mask NaN values
    mask = ~np.isnan(counts) & ~np.isnan(amu) & ~np.isnan(time)
    counts = counts[mask]; amu = amu[mask]; time = time[mask]; pryro_temp = pryro_temp[mask]; col_temp = col_temp[mask]

    data_dict = {'time': time, 'amu': amu, 'pryro_temp': pryro_temp, 'col_temp': col_temp, 'counts': counts}

    plt.plot(amu, counts/np.max(counts))
    plt.title(file_name)
    plt.xlabel('AMU')
    plt.ylabel('Normalised Counts')
    plt.show() 

    new_row = {
        'Sample ID': QMS_datasum_df.iloc[i]['EID'],
        'Location': QMS_datasum_df.iloc[i]['Sample'],
        'Filename': file_name,
        'Data': data_dict,
        'Labels': labels_dict,
        'Unsure':unsure_dict
    }

    results_df = results_df.append(new_row, ignore_index=True)
    
# --- Writing to H5 --- 
results_df.to_hdf('PDS_Data_Labelled.h5', key='data', mode='w')

# --- HTML for viewing --- 
html = results_df.to_html()
with open('results.html', 'w') as f:
    f.write(html)