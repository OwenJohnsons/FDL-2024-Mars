#!/bin/bash

# Check if gcloud is installed
if ! command -v gcloud &> /dev/null
then
    echo "gcloud command could not be found. Please install Google Cloud SDK and authenticate."
    exit
fi

# Constants
VM_COUNT=4
VM_NAME_PREFIX="samurai-training-vm" # Corrected to follow the naming conventions
IMAGE_FAMILY="pytorch-latest-gpu" # PyTorch image family
IMAGE_PROJECT="deeplearning-platform-release" # PyTorch image project
DISK_SIZE="500GB" # Disk size

# List of machine types
MACHINE_TYPES=("a3-highgpu-8g" "a3-megagpu-8g" "a2-highgpu-1g" "a2-highgpu-2g" "a2-highgpu-4g" "a2-highgpu-8g" "a2-megagpu-1g" "a2-megagpu-2g" "a2-megagpu-4g" "a2-megagpu-8g")

# List of US zones
US_ZONES=("us-central1-a" "us-central1-b" "us-central1-c" "us-central1-f" "us-east1-b" "us-east1-c" "us-east1-d" "us-east4-a" "us-east4-b" "us-east4-c" "us-west1-a" "us-west1-b" "us-west1-c" "us-west2-a" "us-west2-b" "us-west2-c")

# Function to create a VM
create_vm() {
    local vm_name=$1
    local zone=$2
    local machine_type=$3

    echo "Creating VM: $vm_name in zone: $zone with machine type: $machine_type and disk size: $DISK_SIZE"

    gcloud compute instances create $vm_name \
        --machine-type=$machine_type \
        --zone=$zone \
        --image-family=$IMAGE_FAMILY \
        --image-project=$IMAGE_PROJECT \
        --boot-disk-size=$DISK_SIZE \
        --maintenance-policy=TERMINATE

    return $?
}

# Loop to create 4 VMs successfully
success_count=0
attempted_count=0

while [ $success_count -lt $VM_COUNT ]
do
    vm_name="${VM_NAME_PREFIX}-${attempted_count}"
    zone=${US_ZONES[$RANDOM % ${#US_ZONES[@]}]}
    machine_type=${MACHINE_TYPES[$RANDOM % ${#MACHINE_TYPES[@]}]}

    create_vm $vm_name $zone "$machine_type"
    if [ $? -eq 0 ]; then
        echo "VM $vm_name created successfully in zone $zone."
        ((success_count++))
    fi

    ((attempted_count++))
done

echo "Successfully created $success_count VMs."