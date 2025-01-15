# Data Augmentation Script

This script generates combinatorial data augmentations for a given dataset and saves the augmented data into specified directories.

## Features

- Combines base data augmentations into all possible subsets.
- Normalizes the augmented data.
- Saves the augmented data into separate folders.

## Requirements

- Python 3.x
- Required libraries:
  - numpy
  - pandas
  - matplotlib
  - itertools
  - argparse

## Usage

### Command-line Arguments

- `--input` (required): Path to the input HDF5 file containing the dataset.
- `--output` (required): Path to the directory where augmented data will be saved.

### Example

Run the script as follows:

```python augmentation_X.py --input path/to/input.hdf5 --output path/to/output_directory```

## Output Structure

The augmented data will be saved in the specified output directory, organized by augmentation types. Each folder corresponds to a specific augmentation and contains the associated dataframes.

## Functions

### `rSubset`
Generates subsets of a given size from an array.

### `get_total_comb_aug_type`
Generates all possible combinations of the base data augmentation columns.

### `get_comb_data_aug`
Performs combinatorial augmentations and normalizes the data.

### `save_dataframes`
Saves the generated dataframes into appropriate directories.


## Contact
- **Author(s):** Pugazhenthi Sivasankar and Owen A. Johnson (ojohnson@tcd.ie) 
