# Extracting Labels for MARS Mass Spectrometry Data

## Overview
This script extracts and organizes labels assigned by the GSFC Planetary Environments Lab Workshops for MARS Mass Spectrometry data. It processes raw `.txt` files, associates them with relevant labels, and outputs a structured dataset for use in the FDL 2024 Mars Challenge.

## Script Functionality

### Inputs
1. **Raw Data Files:**
   - Directory: `PDS_data/SAM_PDS/*/*.txt`
   - Description: Contains `.txt` files with MARS Mass Spectrometry data.
2. **Labels File:**
   - File: `EGAMS_workshop_labels.csv`
   - Description: Provides labels for Mars locations with geological and chemical features.
3. **Metadata File:**
   - File: `PDS_SAM_file_summary.csv`
   - Description: Metadata for the data files, including experiment types and sample locations.

### Outputs
1. **HDF5 File:**
   - File: `PDS_Data_Labelled.h5`
   - Content: Contains the processed data with labels, organized into a structured dataset.
2. **HTML File:**
   - File: `results.html`
   - Content: Provides a browsable view of the processed data for easy verification.
  
### CSV Columns for Labels
The `EGAMS_workshop_labels.csv` file must contain the following columns:
- **Mars Locations:** Specifies the sample locations on Mars. This is a key column used for matching with the metadata and raw data files.
- **Geological/Chemical Labels:** Specific columns expected in the CSV file include:
  - `carbonate`
  - `chloride`
  - `oxidized organic carbon`
  - `oxychlorine`
  - `sulfate`
  - `sulfide`
  - `nitrate`
  - `iron_oxide`
  - `phyllosilicate`
  - `silicate`

  Each label column should contain:
  - `X`: Indicates the presence of the label.
  - `~`: Indicates uncertainty regarding the presence of the label.
  - Blank: Indicates the absence of the label.

## Script Workflow
1. **Load Labels:**
   - Reads `EGAMS_workshop_labels.csv` and filters for EGA and Chemin locations.

2. **Filter Metadata:**
   - Reads metadata from `PDS_SAM_file_summary.csv`, excluding label files and filtering for QMS experiments.

3. **Match Files to Metadata:**
   - Iterates through metadata to match filenames in the raw data directory.

4. **Load and Process Data:**
   - Reads and normalizes data from `.txt` files.
   - Filters and sorts data by AMU, removing NaN values.

5. **Extract Labels:**
   - Matches sample locations with labels.
   - Constructs dictionaries for confirmed and uncertain labels.

6. **Visualize Data:**
   - Generates AMU vs. normalized counts plots for each file.

7. **Build Dataset:**
   - Appends processed data and labels into a pandas DataFrame with columns:
     - `Sample ID`, `Location`, `Filename`, `Data`, `Labels`, `Unsure`.

8. **Save Output Files:**
   - Writes the dataset to an HDF5 file for further analysis.
   - Exports the dataset to an HTML file for easy visualization.

## Usage

### Prerequisites
- Python 3.x
- Required Libraries:
  - `numpy`
  - `pandas`
  - `matplotlib`
  - `tqdm`
  - `glob`

### Running the Script
1. Ensure the input files are available in the specified directories.
2. Run the script:
   ```
   python extract_labels.py
   ```.

### Notes
- The script skips files related to blanks, calibration, and certain locations.


# Extracting labels from SAM Workshop Results 

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
