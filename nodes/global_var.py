import os
import re
import time
import copy
import json
import torch
import signal
import requests
import threading
import numpy as np
from torch import nn
import concurrent.futures
import matplotlib.pyplot as plt
from torchvision import datasets
from flask import Flask, request
from collections import defaultdict
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
import torchvision.transforms as transforms


class ServerNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_stack2 = nn.Sequential(
            nn.Conv2d(256, 384, kernel_size=(3, 3), stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, kernel_size=(3, 3), stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=(3, 3), stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3, 3), stride=2)
        )

        self.classification_stack = nn.Sequential(
            nn.Flatten(),
            nn.Linear(6*6*256, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 62),
            nn.Softmax(1)
        )

    def forward(self, data):
        x = self.conv_stack2(data)
        return self.classification_stack(x)


class ClientNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_stack1 = nn.Sequential(
            nn.Conv2d(1, 96, kernel_size=(11, 11), stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3, 3), stride=2),
            nn.Conv2d(96, 256, kernel_size=(5, 5), stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3, 3), stride=2)
        )

    def forward(self, data):
        x = self.conv_stack1(data)
        return x


class CustomImageDataset(Dataset):
    def __init__(self, data):
        x = torch.Tensor(np.array(data['x']).reshape((-1, 28, 28)))
        resize_transform = transforms.Resize((224, 224))
        self.x = torch.stack([resize_transform(img.unsqueeze(0)) for img in x])
        self.y = torch.Tensor(np.array(data['y']))

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        data = self.x[idx]
        label = self.y[idx].long()
        return data, label
