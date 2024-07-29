#!/bin/bash 

# Constants 
DBPATH="/home/owenj/bucket/samurai_data_mini"
TRAINPATH="/home/owenj/fdl-2024-mars/ML_Models/data_split_EID/train_set_EID.hdf" 
TESTPATH="/home/owenj/fdl-2024-mars/ML_Models/data_split_EID/test_set_EID.hdf"
MAXLEN=1420

# Arguments for classifier
CLASSIFIER=$1
PARAMS=$2

# Start the process 
# bash launch.sh RandomForest '{"n_estimators": 100, "random_state": 1, "verbose": 1}'
python train_evaluate.py --database_path $DBPATH --train_set $TRAINPATH --test_set $TESTPATH --max_length $MAXLEN --classifier $CLASSIFIER --params "$PARAMS"