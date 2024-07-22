#!/bin/bash

# Check if gcloud is installed
if ! command -v gcloud &> /dev/null
then
    echo "gcloud command could not be found. Please install Google Cloud SDK and authenticate."
    exit
fi

# Constants
VM_COUNT=4
VM_NAME_PREFIX="SAMurAI-Training-VM"
IMAGE_FAMILY="pytorch-latest-gpu"
IMAGE_PROJECT="deeplearning-platform-release"
ADD_GPUS="yes"
GPU_TYPES=("nvidia-tesla-a100" "nvidia-tesla-h100" "nvidia-tesla-b100")
GPU_COUNT=2
DISK_SIZE="500GB" 

# List of US zones
US_ZONES=("us-central1-a" "us-central1-b" "us-central1-c" "us-central1-f" "us-east1-b" "us-east1-c" "us-east1-d" "us-east4-a" "us-east4-b" "us-east4-c" "us-west1-a" "us-west1-b" "us-west1-c" "us-west2-a" "us-west2-b" "us-west2-c")

create_vm() {
    local vm_name=$1
    local zone=$2
    local gpu_flag=$3

    echo "Creating VM: $vm_name in zone: $zone with GPU flag: $gpu_flag"

    gcloud compute instances create $vm_name \
        --zone=$zone \
        --image-family=$IMAGE_FAMILY \
        --image-project=$IMAGE_PROJECT \
        --boot-disk-size=$DISK_SIZE \
        $gpu_flag

    return $?
}

success_count=0
attempted_count=0

while [ $success_count -lt $VM_COUNT ]
do
    vm_name="${VM_NAME_PREFIX}-${attempted_count}"
    zone=${US_ZONES[$RANDOM % ${#US_ZONES[@]}]}

    if [[ "$ADD_GPUS" == "yes" ]]; then
        for gpu_type in "${GPU_TYPES[@]}"
        do
            gpu_flag="--accelerator type=$gpu_type,count=$GPU_COUNT"
            create_vm $vm_name $zone "$gpu_flag"
            if [ $? -eq 0 ]; then
                echo "VM $vm_name created successfully."
                ((success_count++))
                break
            fi
        done
    else
        create_vm $vm_name $zone ""
        if [ $? -eq 0 ]; then
            echo "VM $vm_name created successfully."
            ((success_count++))
        fi
    fi

    ((attempted_count++))
done

echo "Successfully created $success_count VMs."