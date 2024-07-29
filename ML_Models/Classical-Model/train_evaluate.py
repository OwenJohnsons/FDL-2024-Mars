"""
Code Purpose: 

Modules:

Note:
    - data-processing.py is a module that contains functions for data processing and loading
    - model.py is a module that contains functions for model building and training
    - train-evaluate.py is a module that contains functions for training and evaluating the model
  
Imports:

Author:
    [Owen A. Johnson]

Last updated:
    [2024-07-25]
"""

import numpy as np
import os
from data_processing import *
from classifiers import *
import argparse
import json 

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--database_path", required=True)
    parser.add_argument("--train_set", required=True)
    parser.add_argument("--test_set", required=True)
    parser.add_argument("--max_length", required=False)
    parser.add_argument("--classifier", required=True, choices=["RandomForest", "LogisticRegression", "SVC"]) 
    parser.add_argument("--params", required=True)
    return parser.parse_args()

def main():
    args = parse_args()
    print('Number of CPUs:', os.cpu_count())

    file_map = create_file_map(args.database_path)

    if args.max_length:
        max_length = int(args.max_length) 
    else:
        print('Fetching max sample length...')
        max_length = longest_sample(file_map)
        print(f"Longest sample length: {max_length}")

    file_paths = file_map['path'].values

    # --- Loading Training Data --- 
    train_ids = pd.read_hdf(args.train_set)['Sample ID'].values
    train_data, train_full_id_array = load_data_set(train_ids, file_map, max_length)
    train_labels = load_labels(args.train_set, train_full_id_array)

    print('Training data shape:', train_data.shape)
    print('Training labels shape:', train_labels.shape)

    # --- Loading Testing Data ---
    test_ids = pd.read_hdf(args.test_set)['Sample ID'].values
    test_data, test_full_id_array = load_data_set(test_ids, file_map, max_length)
    test_labels = load_labels(args.test_set, test_full_id_array)

    print('Training data shape:', train_data.shape)
    print('Training labels shape:', train_labels.shape)

    # --- Training the model ---
    params = json.loads(args.params)
    print("\nTraining the model...")
    classifier = get_classifier(args.classifier, params)
    multi_target_classifier = MultiOutputClassifier(classifier, n_jobs=2)
    multi_target_classifier.fit(train_data, train_labels)

    labels = multi_target_classifier.predict(test_data)

    accuracies = []

    for i in range(test_labels.shape[1]):
        accuracy = accuracy_score(test_labels[:, i], labels[:, i])
        accuracies.append(accuracy)
        print(f"Accuracy for label {i}: {accuracy}")

    print("Training complete. Accuracies for each label:")
    for i, accuracy in enumerate(accuracies):
        print(f"Label {i}: {accuracy}")

if __name__ == "__main__":
    print("Starting script...")
    main()
    print("Script completed.")