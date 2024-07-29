import pandas as pd
import os
import data_description

'''
If you have not generated the training and test dataframes,
run the get_data.py function in this folder first
'''

def get_datasets(root_dir):
    # -------------------------------------- #
    #     training dirs, model paths,etc     #
    # We have already generated the data     # 
    # -------------------------------------- #

    train_dir = root_dir + "train/"
    test_dir  = root_dir + "test/"
    train_label_file = root_dir + "pd_train.hdf"
    test_label_file  = root_dir + "pd_test.hdf"

    # ------ For the training dataset ---------------------------
    train_df1 = pd.read_hdf(train_label_file)
    train_df = train_df1['Labels'].apply(pd.Series)
    train_df.insert(0, 'Image_Name', train_df1['Sample ID'])

    for i in range (10):
        train_df[train_df.columns[1+i]] = train_df[train_df.columns[1+i]].apply(int)

    #only because we started with the png so the hdf file contains all the names in png format
    train_df['Image_Name'] = train_df['Image_Name'].str.replace('png', 'jpg') 


    # ----------For the test dataset ------------------------------
    test_df1 = pd.read_hdf(test_label_file)
    test_df  = test_df1['Labels'].apply(pd.Series)

    test_df.insert(0, 'Image_Name', test_df1['Sample ID'])

    for i in range (10):
        test_df[test_df.columns[1+i]] = test_df[test_df.columns[1+i]].apply(int)

    #only because we started with the png so the hdf file contains all the names in png format
    test_df['Image_Name'] = test_df['Image_Name'].str.replace('png', 'jpg') 

    # everything below is to format the data based on the pretrained model -------------------
    train_df["Image_Name"] = train_df["Image_Name"].map(lambda x : os.path.join(train_dir,x))
    train_df.head(10)
    test_df["Image_Name"] = test_df["Image_Name"].map(lambda x : os.path.join(test_dir, x))
    test_df.head(10)

    # -------------------------------- #
    #        train , test              #
    # -------------------------------- #

    train_dataset = data_description.ImageDataset(train_df)
    test_dataset  = data_description.ImageDataset(test_df)

    return train_dataset, test_dataset