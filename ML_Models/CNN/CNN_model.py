'''
Code Authors: Owen A. Johnson and Arushi Saxena 
Date: 22/07/2024
Code Purpose: This code utlizes pytorch to create a 3D CNN model to analyse .mp4 videos created from Mass Spectrometry Data. 
'''

# import torch 
# import torch.nn as nn
# import torch.nn.functional as F

# class CNN(nn.Module):
#     def __init__(self):
#         super(CNN, self).__init__()
#         self.conv1 = nn.Conv3d(1, 32, kernel_size=3, stride=1, padding=1)
#         self.conv2 = nn.Conv3d(32, 64, kernel_size=3, stride=1, padding=1)
#         self.conv3 = nn.Conv3d(64, 128, kernel_size=3, stride=1, padding=1)
#         self.conv4 = nn.Conv3d(128, 256, kernel_size=3, stride=1, padding=1)
#         self.conv5 = nn.Conv3d(256, 512, kernel_size=3, stride=1, padding=1)
#         self.pool = nn.MaxPool3d(kernel_size=2, stride=2, padding=0)
#         self.fc1 = nn.Linear(512*3*3*3, 512)
#         self.fc2 = nn.Linear(512, 256)
#         self.fc3 = nn.Linear(256, 128)
#         self.fc4 = nn.Linear(128, 64)
#         self.fc5 = nn.Linear(64, 2)
#         self.dropout = nn.Dropout(0.5)
        
#     def forward(self, x):
#         x = F.relu(self.conv1(x))
#         x = self.pool(x)
#         x = F.relu(self.conv2(x))
#         x = self.pool(x)
#         x = F.relu(self.conv3(x))
#         x = self.pool(x)
#         x = F.relu(self.conv4(x))
#         x = self.pool(x)
#         x = F.relu(self.conv5(x))
#         x = self.pool(x)
#         x = x.view(-1, 512*3*3*3)
#         x = self.dropout(F.relu(self.fc1(x)))
#         x = self.dropout(F.relu(self.fc2(x)))
#         x = self.dropout(F.relu(self.fc3(x)))
#         x = self.dropout(F.relu(self.fc4(x)))
#         x = self.fc5(x)
#         return x