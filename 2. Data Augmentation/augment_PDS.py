import os
import numpy as np
import pandas as pd
from itertools import combinations
import math
import random
import matplotlib.pyplot as plt


def rSubset(arr, r):
    """
    Returns a list of all subsets of length r from the array.
    """
    return list(combinations(arr, r))


def get_total_comb_aug_type(data_aug_base_type):
    """
    Returns a list of combinatorial data augmentation column numbers in the master dataframe.

    Args:
        data_aug_base_type (dict): Dictionary of the base data augmentation types.

    Returns:
        list: List of combinatorial data augmentation columns.
    """
    bda_num = list(data_aug_base_type.keys())
    base_aug_num = [i for i in range(2, len(data_aug_base_type) + 1)]
    final_comb_aug_type = []
    total_comb = 0

    for i in range(len(base_aug_num)):
        comb_aug_type = rSubset(bda_num, base_aug_num[i])

        if len(comb_aug_type) == math.comb(len(bda_num), base_aug_num[i]):
            print(f"Number of possible {base_aug_num[i]} combinations out of {len(bda_num)} base data augmentations: {len(comb_aug_type)}")

        final_comb_aug_type += comb_aug_type
        total_comb += len(comb_aug_type)

    print(f"Total number of combinatorial augmentations: {total_comb}")
    return final_comb_aug_type


def get_comb_data_aug(df_base_aug, final_comb_aug_type):
    """
    Performs combinatorial data augmentations using the base augmentations.

    Args:
        df_base_aug (pd.DataFrame): DataFrame with preprocessed and normalized raw signal.
        final_comb_aug_type (list): List of tuples containing combinations of base augmentations.

    Returns:
        dict: Dictionary containing data frames for each combination.
    """
    col_interest = [i for i in range(6, len(df_base_aug.columns))]
    eid = df_base_aug["EID"][0].split(".")[0]
    total_aug_num = 1 + (len(df_base_aug.columns) - 7) + len(final_comb_aug_type)
    df_coll = {}

    for i in range(total_aug_num):
        if i == 0:
            df_coll[f"df_{eid}_norm"] = df_base_aug.iloc[:, [0, 1, 2, 3, 4, 5, col_interest[i]]]
        elif i in [1, 2, 3, 4, 5, 6]:
            df_coll[f"df_{eid}_base{i}"] = df_base_aug.iloc[:, [0, 1, 2, 3, 4, 5, col_interest[i]]]
        else:
            dummy = df_base_aug.iloc[:, [0, 1, 2, 3, 4, 5]]
            combination = final_comb_aug_type[i - 7]
            combo_name = f"combo_{'_'.join(map(str, combination))}"
            combo_val = np.mean([df_base_aug.iloc[:, col] for col in combination], axis=0)
            dummy[combo_name] = combo_val / max(combo_val)
            df_coll[f"df_{eid}_{combo_name}"] = dummy

    return df_coll


def save_dataframes(df_coll, main_folder_path):
    """
    Saves the generated dataframes into respective folders.

    Args:
        df_coll (dict): Dictionary of dataframes.
        main_folder_path (str): Path to save the dataframes.
    """
    for key, df in df_coll.items():
        folder_name = os.path.join(main_folder_path, key.split("_")[-1])
        os.makedirs(folder_name, exist_ok=True)
        filename = f"{key.split('_')[1]}.hdf"
        df.to_hdf(os.path.join(folder_name, filename), key="data")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate and save data augmentations.")
    parser.add_argument("--input", type=str, required=True, help="Path to input HDF5 file.")
    parser.add_argument("--output", type=str, required=True, help="Directory to save augmented data.")

    args = parser.parse_args()

    data_aug_base_type = {
        7: "shift",
        8: "shiftRan",
        9: "randInten",
        10: "aw_noise",
        11: "ag_noise",
        12: "stretch2"
    }

    master_df = pd.read_hdf(args.input)
    final_comb_aug_type = get_total_comb_aug_type(data_aug_base_type)

    for eid in master_df["Sample ID"].unique():
        sample_df = master_df[master_df["Sample ID"] == eid]
        df_coll = get_comb_data_aug(sample_df, final_comb_aug_type)
        save_dataframes(df_coll, args.output)
