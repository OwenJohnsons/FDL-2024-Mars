''' 
Code Authors: Owen A. Johnson and Arushi Saxena 
Date: 22/07/2024
Code Purpose: This code loads .mp4 videos and converts them into 3D tensors as to be ingested by the CNN model. 
'''

import cv2
import torch
import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from glob import glob

class VideoDataset(Dataset):
    def __init__(self, video_path, transform=None):
        self.video_path = video_path
        self.transform = transform
        self.frames = self._load_video()

    def __len__(self):
        return len(self.frames)

    def _load_video(self):
        frames = []
        cap = cv2.VideoCapture(self.video_path)
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #  grayscale if needed
            frames.append(frame)
        cap.release()
        return np.array(frames)
    
    def __getitem__(self, idx):
        frame = self.frames[idx]
        frame = torch.tensor(frame, dtype=torch.float32).unsqueeze(0)  # Add channel dimension
        if self.transform:
            frame = self.transform(frame)
        return frame
    
dataset = VideoDataset(video_path=glob('/*/*.mp4'))
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)