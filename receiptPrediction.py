import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import json

def preprocessing(file):
    data = pd.read_csv(file)

    #extract the data for month
    data['# Date'] = pd.to_datetime(data['# Date'])
    data['Month'] = data['# Date'].dt.month
    #sum up the months and group them
    monthlyData = data.groupby('Month')['Receipt_Count'].sum().reset_index()
    return monthlyData


#NN defined below
class receiptPrediction(nn.Module):
    #input is month, output is count
    def __init__(self):
        super(receiptPrediction, self).__init__()
        self.fc1 = nn.Linear(1, 16)
        self.fc2 = nn.Linear(16, 8)
        self.fc3 = nn.Linear(8, 1)
    #relu is very popular option and sufficient for this task
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
#Training
def train(monthlyData):
    #normalize the data, leads to better estimation
    maxCount = monthlyData['Receipt_Count'].max()
    monthlyData['Receipt_Count'] /= maxCount

    #load up the data
    X = torch.tensor(monthlyData['Month'].values, dtype=torch.float32).view(-1,1)
    y = torch.tensor(monthlyData['Receipt_Count'].values, dtype=torch.float32).view(-1,1)
    #Initialize model, loss, and optim
    #MSE and Adam both chosen
    model = receiptPrediction()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    #training iterations (200)
    for epoch in range(200):
        model.train()
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
    
    #save model
    torch.save(model.state_dict(), 'model.pth')
    #save maxCount
    with open('config.json', 'w') as config_file:
        json.dump({"maxCount": maxCount}, config_file)
    return model

#load model
def loadModel():
    model = receiptPrediction()
    model.load_state_dict(torch.load('model.pth'))
    model.eval()
    #grab maxCount
    with open('config.json', 'r') as config_file:
        config = json.load(config_file)
        maxCount = config["maxCount"]
    return model, maxCount

#training routine
def main():
    #process the data
    monthlyData = preprocessing('data_daily.csv')
    print(monthlyData)
    #train the model
    model = train(monthlyData)
    print('Training complete')

if __name__ == '__main__':
    main()