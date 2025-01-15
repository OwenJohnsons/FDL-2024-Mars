import torch
import numpy as np
import os
import random

# ---------------------------------------------------- #
#    Configuration class contains training configs     #
#                  &  model configs                    #
# ---------------------------------------------------- #

class Config:
        BATCH_SIZE = 32
        KFOLD = 5
        FOLD = 0
        SEED = 21
        LEARNING_RATE = 1e-5
        TYPE  = "Multi-label Image Classification"
        DATA_SOURCE = "multi-label-image-classification-dataset"
        MODEL_NAME = 'resnetv2_50'
        CRITERION_ = "Binary Cross Entropy"
        OPTIMIZER_ = "AdamW"
        DATA_TYPE = 'image'
        DATA_FORMAT = '.jpg'
        IMG_SIZE = (602, 602)
        EPOCHS = 10
        
        def __init__(self):
            print("configuration set!")
        
        def check_cuda(self):
            print("Scanning for CUDA")
            if torch.cuda.is_available():
                print("GPU is available , training will be accelerated! : )\n")
            else:
                print("NO GPUs found : / \n")
        
        def seed_everything(self):
            print("Seeding...")
            np.random.seed(self.SEED)
            random.seed(self.SEED)
            os.environ['PYTHONHASHSEED'] = str(self.SEED)
            torch.manual_seed(self.SEED)
            torch.cuda.manual_seed(self.SEED)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            print("Seeded everything!")