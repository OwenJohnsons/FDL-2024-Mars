"""
Code Purpose: 
    This script calls functions from classifiers.py and data_processing.py to carry out multi-class multi-label classification on EGAMS data for the Mars 2024 FDL Challenge 

Arguments:
    --database_path: Path to the database
    --train_set: Path to the training set EIDs 
    --test_set: Path to the testing set EIDs 
    --max_length: Maximum length of the samples (if not provided, the script will calculate it)
    --classifier: Classifier to use
    --params: Parameters for the classifier (see scikit-learn documentation for the classifier you choose for specifics on the parameters)

Note:
    - data-processing.py is a module that contains functions for data processing and loading.
    - classifiers.py is a module that contains functions for creating and training classifiers.
    - train-evaluate.py is a module that contains functions for training and evaluating the model and contains the main() function. 

Author:
    [Owen A. Johnson]

Last updated:
    [2024-08-01]
"""

import numpy as np
import os
import gc 
import pandas as pd  
from data_processing import *
from classifiers import *
import argparse
import json 
from sklearn.metrics import accuracy_score, recall_score, precision_score, ConfusionMatrixDisplay, confusion_matrix
from sklearn.multioutput import MultiOutputClassifier 
import matplotlib.pyplot as plt 
import scienceplots; plt.style.use(['science', 'no-latex'])
import time

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--database_path", required=True)
    parser.add_argument("--train_set", required=True)
    parser.add_argument("--test_set", required=True)
    parser.add_argument("--max_length", required=False)
    parser.add_argument("--classifier", required=True, choices=["RandomForest", "LogisticRegression", "SVC", "KNeighbors", "DecisionTree", "GradientBoosting", "AdaBoost", "XGBoost", "RidgeClassifier"]) 
    parser.add_argument("--params", required=True)
    parser.add_argument("--ensemble", required=False, default=False)
    return parser.parse_args()

def plot_confusion_matrix(true_labels, predicted_labels, title, all_labels, directory):

    cm = confusion_matrix(true_labels, predicted_labels, labels=all_labels)    
    cm_percentage = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100  # Convert the counts to percentages
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_percentage, display_labels=['False', 'True'])
    
    fig, ax = plt.subplots()
    disp.plot(ax=ax, cmap='Blues', values_format='.1f')
    disp.ax_.set_title(title)
    
    plt.savefig(f"{directory}/{title}_confusion_matrix.jpg", dpi=200)
    plt.close()

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

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
    
    gc.collect()
    file_paths = file_map['path'].values

    # --- Loading Training Data --- 
    tr_dl_start = time.time()
    train_ids = pd.read_hdf(args.train_set)['Sample ID'].values.astype(str)

    train_data, train_full_id_array = load_data_set(train_ids, file_map, max_length)
    train_labels = load_labels(args.train_set, train_full_id_array)
    print('Training Labels Distribution:', np.sum(train_labels, axis=0))
    
    print('Training data shape:', train_data.shape)
    print('Training labels shape:', train_labels.shape)
    tr_dl_time = time.time() - tr_dl_start

    # --- Loading Testing Data ---
    test_dl_time = time.time()
    test_ids = pd.read_hdf(args.test_set)['Sample ID'].values.astype(str)

    test_data, test_full_id_array = load_data_set(test_ids, file_map, max_length)
    test_labels = load_labels(args.test_set, test_full_id_array)
    print('Test Labels Distribution:', np.sum(test_labels, axis=0))
    test_dl_time = time.time() - test_dl_time

    print('Test data shape:', test_data.shape)
    print('Test labels shape:', test_labels.shape)

    # --- Training the model ---
    train_time = time.time()
    params = json.loads(args.params)
    print("\nTraining the model...")
    classifier = get_classifier(args.classifier, params)
    multi_target_classifier = MultiOutputClassifier(classifier, n_jobs=2)
    multi_target_classifier.fit(train_data, train_labels)

    labels = multi_target_classifier.predict(test_data)
    train_time = time.time() - train_time   

    accuracies = []

    for i in range(test_labels.shape[1]):
        accuracy = accuracy_score(test_labels[:, i], labels[:, i])
        accuracies.append(accuracy)
    
    print("\nTraining complete! \n--- Accuracies for each label ---")

    labels_string = ['carbonate', 'chloride', 'iron oxide', 'nitrate', 
    'oxidized organic carbon', 'oxychlorine', 'phyllosilicate', 
    'silicate', 'sulfate', 'sulfide']
    all_labels = [0, 1]

    # --- Make save directory ---
    classifier_name = args.classifier
    date = time.strftime("%Y-%m-%dT%H-%M")
    max_length = str(max_length)

    save_dir = f"results/{classifier_name}_{max_length}_{date}"

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # --- Save predictions and file paths to a DataFrame ---
    predictions_df = pd.DataFrame(labels, columns=labels_string)

    predictions_df['File Path'] = [
        "/".join(file_map.loc[file_map['id'] == id]['path'].values[0].split('/')[-2:])
        for id in test_full_id_array
    ]
    predictions_df.to_csv(f"{save_dir}/predictions_with_paths.csv", index=False)

    # --- Accuracy ---
    for i, accuracy in enumerate(accuracies):
        print(f"{labels_string[i]}: {100*accuracy:.2f}%")

    # --- Recall ---
    print("\n--- Recall ---")
    for i in range(test_labels.shape[1]):
        r_score = recall_score(test_labels[:, i], labels[:, i])
        print(f"{labels_string[i]}: {r_score:.2f}")

    # --- Precision ---
    print("\n--- Precision ---")
    for i in range(test_labels.shape[1]):
        pre_score = precision_score(test_labels[:, i], labels[:, i])
        print(f"{labels_string[i]}: {pre_score:.2f}")

    # --- Save metrics ---
    with open(f"{save_dir}/metrics.txt", 'w') as f:
        f.write(f"Classifier: {classifier_name}\n")
        f.write(f"Max Length: {max_length}\n")
        f.write(f"Date: {date}\n")
        f.write(f"Training Data: {args.train_set}\n")
        f.write(f"Testing Data: {args.test_set}\n")
        f.write(f"Parameters: {args.params}\n")

        f.write("\n--- Accuracies for each label ---\n")
        for i, accuracy in enumerate(accuracies):
            f.write(f"{labels_string[i]}: {100*accuracy:.2f}%\n")

        f.write("\n--- Recall ---\n")
        for i in range(test_labels.shape[1]):
            r_score = recall_score(test_labels[:, i], labels[:, i])
            f.write(f"{labels_string[i]}: {r_score:.2f}\n")

        f.write("\n--- Precision ---\n")
        for i in range(test_labels.shape[1]):
            pre_score = precision_score(test_labels[:, i], labels[:, i])
            f.write(f"{labels_string[i]}: {pre_score:.2f}\n")
    
    # --- Plot Confusion Matrix ---
    for i in range(test_labels.shape[1]):
        plot_confusion_matrix(test_labels[:, i], labels[:, i], labels_string[i], all_labels, save_dir)

    print('\n--- EXECUTION TIME BREAKDOWN ---')
    print(f"Training data load time: {tr_dl_time / 60} mins")
    print(f"Testing data load time: {test_dl_time / 60} mins")
    print(f"Training time: {train_time / 60} mins")

if __name__ == "__main__":
    print("Starting script...")
    main()
    print("Script completed.")