import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class receiptPrediction(nn.Module):
    def __init__(self):
        super(receiptPrediciton, self).__init__()
        self.fc1 = nn.Linear(1, 16)
        self.fc2 = nn.Linear(16, 8)
        self.fc3 = nn.Linear(8, 1)

