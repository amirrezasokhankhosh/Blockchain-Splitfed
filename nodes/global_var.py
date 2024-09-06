import os
import re
import sys
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



class ClientNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()

    def forward(self, data):
        x = self.pool(self.relu(self.conv1(data)))
        return x

class ServerNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()

    def forward(self, data):
        x = self.pool(self.relu(self.conv2(data)))  # Conv2 -> ReLU -> Pool
        x = x.view(-1, 64 * 7 * 7)              # Flatten
        x = self.relu(self.fc1(x))               # Fully connected -> ReLU
        x = self.fc2(x)                          # Fully connected
        return x
