#!/bin/bash

# Number of GPUs to use
# NUM_GPUS=2  # Replace with the number of GPUs you have
NUM_GPUS=1

# Path to main python script 
SCRIPT_PATH="main.py"

# Run the script with torch.distributed.launch
# python -m torch.distributed.launch --nproc_per_node=$NUM_GPUS $SCRIPT_PATH

# Run the script with torchrun 
torchrun --nproc_per_node=$NUM_GPUS $SCRIPT_PATH --verbose