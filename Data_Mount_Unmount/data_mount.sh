#! /usr/bin/bash

# This script mounts the data from the cloud storage buckets onto the virtual machine.
# Please run this from the home directory of your virtual machine so that the 
# destination folder is created in the home directory.

# Suggested usage:

# >> ./data_mount.sh <source_dir> <destination_dir>

# Suggested usage:


# To install GCS FUSE:
sudo apt-get update
sudo apt-get install fuse
export GCSFUSE_REPO=gcsfuse- 'lsb_release -c -s' 
echo "deb https://packages.cloud.google.com/apt $GCSFUSE_REPO main" | sudo tee /etc/apt/sources.list.d/gcsfuse.list
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add - 
sudo apt-get update
sudo apt-get install gcsfuse 

# Check if at least two arguments are provided:
if [ $# -lt 2 ]; then
    echo "Usage: $0 <first_input> <second_input>"
    exit 1
fi

# Assign the respective input values
source_dir=$1
destination_dir=$2

# Create a destination folder:
mkdir $destination_dir

# Mount the data in READ ONLY format:
gcsfuse --implicit-dirs -o ro $source_dir $destination_dir