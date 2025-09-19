"""
model.py
-----------

Defines the convolutional neural network used to classify images from the CIFAR-10 dataset.  The architecture is intentionally simple but modular so that it can be easily modified or replaced.
"""

import torch.nn as nn
import torch.nn.functional as F


class CIFAR10CNN(nn.Module):
    """A simple Convolutional Neural Network for CIFAR-10 classification."""

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)

        # Batch normalization to speed up training and improve stability
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)

        # Fully connected layers
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, num_classes)

        # Dropout layer to reduce overfitting
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        # First conv layer + ReLU + MaxPool
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)

        # Second conv layer + ReLU + MaxPool
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)

        # Third conv layer + ReLU + MaxPool
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)

        # Flatten before feeding into fully connected layers
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


__all__ = ["CIFAR10CNN"]
