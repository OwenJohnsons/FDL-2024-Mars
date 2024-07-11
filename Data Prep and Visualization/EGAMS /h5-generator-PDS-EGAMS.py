#%%
'''
Author: Owen A. Johnson (ojohnson@tcd.ie) & Arushi Saxena
Last Major Update: 2024-06-28
Code Purpose: To generate a HDF5 file from the PDS data intaking raw .txt files from the NASA Planetary Data System (PDS) for the Curosity Rover and then cross referencing the data with the labels provided by the EGAMS workshop carried out by Victoria Da Poian (GSFC) and Eric Lyness (GSFC).
'''
import warnings
import pandas as pd
import numpy as np
from glob import glob
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import sys 
from scipy import stats
from scipy.interpolate import griddata
import os 
import scienceplots; plt.style.use(['science', 'ieee'])

warnings.filterwarnings("ignore", category=FutureWarning, message="The frame.append method is deprecated")
warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value encountered in divide")

# --- Preamble --- 
verbose = False
min_amu = 10; max_amu = 150
array_length = max_amu - min_amu 

nobelgas_EID = [25117, 25202, 25208, 25219, 25362, 25363, 25495]
combustion_EID = [25173, 25174, 25175]
blanks_EID = [25032, 25033, 25059, 25083, 25133, 25145, 25235, 2536, 25392, 25393, 25188, 25223, 25689]

# --- Load the Labels --- 
labels_df = pd.read_csv('EGAMS_workshop_labels.csv')
EGA_df = labels_df[labels_df['Mars Locations'].str.contains('EGA')] # Just EGA labels 
Chemin_df = labels_df[labels_df['Mars Locations'].str.contains('Chemin')] # Just Chemin labels
filepaths = glob('PDS_data/SAM_PDS/*/*.txt')
EGA_eids = np.loadtxt('EGAMS_EID_IDs.txt', dtype=int)
print('Number of EGA EIDs:', len(EGA_eids))

# --- Remove nobel gas and combustion files --
EGA_eids = [EID for EID in EGA_eids if EID not in nobelgas_EID + combustion_EID]
print('Number of EGA EIDs excluding nobel gas and combustion:', len(EGA_eids))

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
skip_cond = False

for i in tqdm(range(QMS_datasum_df.shape[0])):

    EID = QMS_datasum_df.iloc[i]['EID']

    if EID not in EGA_eids:
        continue

    else:
        smp_loc = QMS_datasum_df.iloc[i]['Sample'].split(' ')[0] # EGM label file only uses first word
        EGA_row = EGA_df[EGA_df['Mars Locations'].str.contains(smp_loc)]

        if verbose: 
            print('=== %s ===' % i)
            print('Sample Location:', QMS_datasum_df.iloc[i]['Sample'])
            print('First word of Sample Location:', smp_loc)
            print(EGA_row['Mars Locations'])

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
        mask = (amu >= min_amu) & (amu <= 151)
        time = time[mask]; amu = amu[mask]; pryro_temp = pryro_temp[mask]; col_temp = col_temp[mask]; counts = counts[mask]
        time = time[~np.isnan(counts)]; amu = amu[~np.isnan(counts)]; pryro_temp = pryro_temp[~np.isnan(counts)]; col_temp = col_temp[~np.isnan(counts)]; counts = counts[~np.isnan(counts)]
        time = time - time[0] # Start time at 0.
        max_count = np.max(counts)


        start_idxes = np.where(amu[:-1] > amu[1:])[0].tolist() # Finding the start of each spectrum, by finding where AMU decreases.
        # add the end to the start_idxes
        start_idxes = [0] + start_idxes + [len(amu)] # Adding the start and end indexes to the list.
        print(start_idxes)
        # break 
        bin_edges = np.arange(min_amu, max_amu + 2)

        amu_placeholder = np.arange(min_amu, max_amu + 1)
        counts_placeholder = np.zeros(len(amu_placeholder))
        smartscan_cond = False

        count_1 = 0; count_2 = 0; count_3 = 0 
        normal_scan = 0

        for k in range(len(start_idxes)-1):
      
            start_idx = start_idxes[k] + 1; end_idx = start_idxes[k+1]
    

            count_indv = counts[start_idx:end_idx]/max_count
            amu_indv = amu[start_idx:end_idx]
            if len(count_indv) < 3:
                continue

            if len(count_indv) == 0:
                raise ValueError('No data found for:', file_name, 'at indexes:', start_idx, end_idx)

            # --- Smart Scanning Adsorption ---
            start_amu = amu_indv[0]; end_amu = amu_indv[-1] # - min and max AMU values of a specific spectrum

            start_spectra_value = min(amu_indv); end_spectra_value = max(amu_indv)
            interpolate_amu     = np.arange(start_spectra_value, end_spectra_value + 1) # - interpolated AMU values
            interpolated_counts = griddata(amu_indv, count_indv, interpolate_amu, method='nearest') # - interpolated counts

            smartscan_cond = len(interpolated_counts) < 140

            if(not smartscan_cond):
                reference_amus = interpolate_amu; reference_counts = interpolated_counts
                smart_scan = 0
                normal_scan += 1
            else:
                cut_start_idx = np.where(amu_placeholder == np.ceil(start_spectra_value))[0][0] 
                cut_end_idx = np.where(amu_placeholder == np.ceil(end_spectra_value))[0][0] + 1
                reference_counts[cut_start_idx:cut_end_idx] =  interpolated_counts
                smart_scan += 1
                normal_scan = 0

            if ((smart_scan > 1 or normal_scan > 1)):
                count_indv = reference_counts; amu_indv = reference_amus
                eid = int(QMS_datasum_df.iloc[i]['EID'])
                
                # if os.path.exists(f"arushi/{eid}/{k}.png"):
                #     continue
                    
                # plt.plot(amu_indv, count_indv) 
                # plt.ylim(0, 1); plt.xlim(10, 150)
                # plt.savefig(f'arushi/{eid}/{k}.png')
                # plt.close()
     
            # --- Peak Analysis ---
            peaks = find_peaks(count_indv, height=0.1)[0]
            peaks_values = [count_indv[peak] for peak in peaks]
            spec_n = k

            if len(peaks) == 0:
                if verbose: print('No peaks found for indexes:', start_idx, end_idx)
                max_peak_amu = 0
            else: 
                max_peak_amu = amu_indv[np.argmax(peaks_values)]

            # --- Pyrolysis Temperature Analysis ---
            if pryro_temp[start_idx:end_idx].mean() < 0:
                real_temp = False
            else:
                real_temp = True

            # --- General Statistics ---
            mean_counts = np.mean(count_indv); std_counts = np.std(count_indv)
            skewness = stats.skew(count_indv); kurtosis = stats.kurtosis(count_indv)

            stat_dict = {
                'Mean Counts': mean_counts, 
                'Std Counts': std_counts, 
                'Skewness': skewness, 
                'Kurtosis': kurtosis
            }
        
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
                'Statistics': (stat_dict),
                'Labels': (labels_dict),  
                'Unsure': (unsure_dict),  
                'Peak Index': (peaks.tolist()),  
                'Peak Values': (peaks_values),  
                'Max Peak AMU': max_peak_amu,
                'Spectra Number': spec_n,
                'Orginial Pyrolysis Temperature': real_temp, 
                'Blank': 1 if int(QMS_datasum_df.iloc[i]['EID']) in blanks_EID else 0
            }
            
            total_spectra_count += 1
            if skip_cond:
                continue
            else: 
                data_frame = data_frame.append(new_row, ignore_index=True)

            previous_counts = count_indv
        # break 
      

print('Size of the data frame: %.9f Gb' % (sys.getsizeof(data_frame) / 1e9))
print('Total number of spectra:', total_spectra_count)
print('Number of rows in the data frame:', data_frame.shape[0])
print('Number of unique filenames:', len(data_frame['Filename'].unique()))

# --- Save the data ---
hdf5_file = 'PDS_EGAMS_H5_files/EGAMS_PDS_Data_AMU;%s-%s_IndvNorm;%s.h5' % (min_amu, max_amu, 'Final')
data_frame.to_hdf(hdf5_file, key='EGAMS_PDS_Data', mode='w')

# --- HTML for viewing --- 
results_df = pd.read_hdf(hdf5_file, 'EGAMS_PDS_Data')
html = results_df.to_html()
with open('PDS_EGAMS_PDS_Data.html', 'w') as f:
    f.write(html)