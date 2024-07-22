import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import VideoDataset
from glob import glob

def test_dataloader(video_path, num_samples=5):

    dataset = VideoDataset(video_path=video_path)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    print(f"Total frames in video: {len(dataset)}")

    for i, frame in enumerate(dataloader):
        if i >= num_samples:
            break
        print(f"Sample {i + 1}:")
        print(f"Frame shape: {frame.shape}")
        print(f"Frame tensor: {frame}")

if __name__ == "__main__":
    video_path = glob('/*/*.mp4')  
    test_dataloader(video_path)
