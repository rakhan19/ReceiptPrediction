import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np

def preprocessing(file):
    data = pd.read_csv(file)

    #extract the data
    data['# Date'] = pd.to_datetime(data['# Date'])
    data['Month'] = data['# Date'].dt.month

    monthlyData = data.groupby('Month')['receiptCount'].sum().reset_index()
    return monthlyData


#NN defined below
class receiptPrediction(nn.Module):
    #input is month, output is count
    def __init__(self):
        super(receiptPrediction, self).__init__()
        self.fc1 = nn.Linear(1, 8)
        self.fc2 = nn.Linear(8, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
#Training
def train(monthlyData):
    X = torch.tensor(monthlyData['Month'].values, )
