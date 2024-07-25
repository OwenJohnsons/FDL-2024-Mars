#!/bin/bash 

# Constants 
DBPATH="/home/owenj/bucket/samurai_data_base"
TRAINPATH="/home/owenj/fdl-2024-mars/ML_Models/data_split_EID/train_set_EID.hdf" 
TESTPATH="/home/owenj/fdl-2024-mars/ML_Models/data_split_EID/test_set_EID.hdf"

# Start the process 
python train_evaluate --database_path $DBPATH --train_path $TRAINPATH --test_path $TESTPATH