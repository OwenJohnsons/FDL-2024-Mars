#! /usr/bin/bash

# This shell script unmounts the data from the cloud storage bucket, and 
# deletes the folder in which the data was mounted.
# Please run this from the home directory of your virtual machine.

# Suggested usage:

# >> ./data_unmount.sh <dir_to_unmount>

# Suggested usage:

# Directory to unmount:
dir_to_unmount=$1

# Command to unmount the directory:
fusermount -u $dir_to_unmount

# Remove the empty directory:
rm -r $dir_to_unmount

