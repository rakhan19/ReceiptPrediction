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
    X = torch.tensor(monthlyData['Month'].values, dtype=torch.float32).view(-1,1)
    y = torch.tensor(monthlyData['Receipt_Count'].values, dtype=torch.float32).view(-1,1)
    #Initialize model, loss, and optim
    model = receiptPrediction()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    #training iterations
    for epoch in range(50):
        model.train()
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
    
    #save model
    torch.save(model.state_dict(), 'model.pth')
    return model
