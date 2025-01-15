# Cleaning PDS Data

## Overview
The script `generate_hdf5.py` processes raw `.txt` files from the NASA Planetary Data System (PDS) related to the Curiosity Rover. It cross-references this data with labels provided by the EGAMS workshop (Victoria Da Poian and Eric Lyness, GSFC) to generate a comprehensive HDF5 file that is Machine Learning (ML) ready.

## Script Functionality

### Inputs
1. **Raw Data Files:** 
   - Directory: `PDS_data/SAM_PDS/*/*.txt`
   - Format: `.txt` files containing data from SAM directly from the Planatary Data System.
2. **Labels File:**
   - File: `EGAMS_workshop_labels.csv`
   - Description: Contains labels for Mars locations, categorizing geological and chemical features.
3. **EID IDs File:**
   - File: `EGAMS_EID_IDs.txt`
   - Description: List of EIDs relevant to the EGAMS study.
4. **Metadata File:**
   - File: `PDS_SAM_file_summary.csv`
   - Description: Metadata for the PDS files, including experiment types and sample locations.

### Outputs
1. **HDF5 File:**
   - File: `PDS_EGAMS_H5_files/EGAMS_PDS_Data_AMU;10-150_IndvNorm;Final.h5`
   - Content: A structured dataset containing processed spectra data, statistical summaries, and label associations.
2. **HTML File:**
   - File: `PDS_EGAMS_PDS_Data.html`
   - Content: A viewable HTML summary of the processed data.

## Script Workflow
1. **Load Labels:**
   - Reads `EGAMS_workshop_labels.csv` to extract labels for EGA and Chemin locations.

2. **Filter Data Files:**
   - Reads raw `.txt` files from the `PDS_data/SAM_PDS/` directory.
   - Excludes files related to noble gases, combustion, and blanks based on EID.

3. **Metadata Processing:**
   - Filters the metadata to only include QMS experiments and non-NaN sample locations. This can be easily removed for other SAM instruments to be used. 

4. **Data Collation:**
   - Iterates through each file to:
     - Match filenames to metadata.
     - Load data using `numpy.loadtxt`.
     - Filter data by AMU range (10-150).
     - Normalize and interpolate counts (optional argument) 
     - Extract peaks and calculate statistical summaries.
     - Generate dictionaries for data, labels, and statistics.

5. **Dataframe Construction:**
   - Appends processed data into a pandas DataFrame with relevant columns:
     - `Sample ID`, `Sample`, `Location`, `Description`, `Data`, `Statistics`, `Labels`, `Peak Index`, etc.

6. **Save Output Files:**
   - Writes the DataFrame to an HDF5 file for efficient storage and retrieval.
   - Exports the DataFrame to an HTML file for easy visualization.

## Usage

### Prerequisites
- Python 3.x
- Required Libraries: 
  - `numpy`
  - `pandas`
  - `matplotlib`
  - `scipy`
  - `tqdm`
  - `glob`
  - `scienceplots`

### Running the Script
1. Ensure the input files are available in the specified directories.
2. Run the script:
   ```
   python generate_hdf5.py
```

### Output Location
- HDF5 File: `PDS_EGAMS_H5_files/`
- HTML File: Current working directory.

### Notes
- The script ignores `.lbl` files in the metadata and skips files without a match in the raw data directory.
- Data is normalized by the maximum count in each file.

## Contact
- **Author:** Owen A. Johnson (ojohnson@tcd.ie) & Arushi Saxena
