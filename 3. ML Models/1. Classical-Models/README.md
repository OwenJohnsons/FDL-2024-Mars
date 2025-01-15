# Multi-Class Multi-Label Classification

## Overview
This script performs multi-class, multi-label classification on EGAMS data for the Mars 2024 Frontier Development Lab (FDL) Challenge. It utilizes custom functions from `classifiers.py` and `data_processing.py` to preprocess data, train models, and evaluate their performance.

## Features
- Supports a variety of classifiers, including RandomForest, LogisticRegression, SVC, KNeighbors, and more.
- Computes accuracy, recall, and precision metrics for each label.
- Visualizes performance using confusion matrices.
- Saves predictions and evaluation metrics for further analysis.

## Prerequisites
Ensure the following dependencies are installed:
- Python 3.6+
- Required Python packages: `numpy`, `pandas`, `matplotlib`, `scikit-learn`, `scienceplots`

You also need the following files:
- `classifiers.py`: Contains functions for creating and training classifiers.
- `data_processing.py`: Contains functions for data processing and loading.

## Usage

### Command-Line Arguments
The script accepts the following command-line arguments:

| Argument          | Description                                                                                       | Required | Example                                                                                             |
|------------------ |---------------------------------------------------------------------------------------------------|----------|-----------------------------------------------------------------------------------------------------|
| `--database_path` | Path to the database.                                                                             | Yes      | `/path/to/database`                                                                                 |
| `--train_set`     | Path to the training set EIDs.                                                                    | Yes      | `/path/to/train_set.h5`                                                                             |
| `--test_set`      | Path to the testing set EIDs.                                                                     | Yes      | `/path/to/test_set.h5`                                                                              |
| `--max_length`    | Maximum length of the samples. If not provided, it will be calculated.                           | No       | `256`                                                                                               |
| `--classifier`    | Classifier to use. Choose from `RandomForest`, `LogisticRegression`, `SVC`, `KNeighbors`, etc.    | Yes      | `RandomForest`                                                                                      |
| `--params`        | Parameters for the classifier in JSON format.                                                    | Yes      | `'{\"n_estimators\": 100, \"max_depth\": 10}'`                                                       |
| `--ensemble`      | Enable ensemble classification. Default is `False`.                                              | No       | `True`                                                                                              |

### Example Usage
```
python train-evaluate.py \
    --database_path /path/to/database \
    --train_set /path/to/train_set.h5 \
    --test_set /path/to/test_set.h5 \
    --classifier RandomForest \
    --params '{"n_estimators": 100, "max_depth": 10}'
```

## Outputs
1. **Predictions with File Paths**:
   - Saved as `predictions_with_paths.csv` in the results directory.
   - Includes predicted labels and corresponding file paths.

2. **Evaluation Metrics**:
   - Saved as `metrics.txt` in the results directory.
   - Includes accuracy, recall, and precision for each label.

3. **Confusion Matrices**:
   - Saved as `.jpg` files in the results directory.
   - Visualize true vs predicted labels for each classification task.

4. **Performance Summary**:
   - Execution time for training data loading, testing data loading, and model training.

## Key Functions
### `parse_args()`
Parses command-line arguments.

### `plot_confusion_matrix()`
Plots and saves confusion matrices.

### `main()`
Main function that orchestrates the data loading, model training, and evaluation pipeline.

## Notes
- Ensure `classifiers.py` and `data_processing.py` are in the same directory as this script.
- Refer to the [scikit-learn documentation](https://scikit-learn.org/stable/documentation.html) for details on classifier parameters.
- Modify the `labels_string` variable in the script to match the specific labels in your dataset.

