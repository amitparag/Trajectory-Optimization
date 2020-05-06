import torch
import numpy as np
import crocoddyl
from tqdm import tqdm
import torch.optim as optim
from torch.utils import data
import torch.nn as nn
from valueNetwork import ValueNet
from residualNetwork import ResidualNet
from dataset import Datagen

def train_on_values(feedforward:bool = True):

    # Training data
    datagen = Datagen()
    positions, values = datagen.positions_values(5000) 
    # Torch dataloader
    dataset = torch.utils.data.TensorDataset(positions,values)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size = 1000) 


    if feedforward:
        net = ValueNet()
        
    else:
        net = ResidualNet()
    # Initialize loss and optimizer
    criterion = torch.nn.MSELoss(reduction='sum')
    optimizer = optim.Adam(net.parameters(), lr=0.001)  

    # Set net to training mode
    net = net.float()
    net.train()
    print("\n Training ... \n")
    for epoch in tqdm(range(10000)):        
        for data, target in dataloader: 

            outputs = net(data)
            loss = criterion(outputs, target)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    xtest, ytest = datagen.positions_values(1000)
    net.eval()
    with torch.no_grad():
        ypred = net(xtest)
    error = (ytest - ypred)
    print(f"Mean Squared Error = {torch.mean(error ** 2)}")
    if feedforward:
        torch.save(net, "valueNet.pth")
    else:
        torch.save(net, "resNet.pth")
        
        
        
def train_on_cost():

    # Training data
    datagen = Datagen()
    positions, values = datagen.positions_costs(5000) 
    # Torch dataloader
    dataset = torch.utils.data.TensorDataset(positions,values)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size = 700) 


    net = ResidualNet()

    # Initialize loss and optimizer
    criterion = torch.nn.MSELoss(reduction='sum')
    optimizer = optim.Adam(net.parameters(), lr=0.001)  

    # Set net to training mode
    net = net.float()
    net.train()
    print("\n Training ... \n")
    for epoch in tqdm(range(10000)):        
        for data, target in dataloader: 

            outputs = net(data)
            loss = criterion(outputs, target)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    xtest, ytest = datagen.positions_costs(1000)
    net.eval()
    with torch.no_grad():
        ypred = net(xtest)
    error = (ytest - ypred)
    print(f"Mean Squared Error = {torch.mean(error ** 2)}")
    torch.save(net, "rescostNet.pth")
    
    
if __name__=='__main__':
    #train_on_values(feedforward=True)
    #train_on_values(feedforward=False)
    train_on_cost()