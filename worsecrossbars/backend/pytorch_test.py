#Imports from external libraries
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
import time

#Imports from internal libraries

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.hidden1 = nn.Linear(784, 128)
        self.hidden2 = nn.Linear(128,10)
        self.dropout = nn.Dropout(0.2)


    def forward(self, x):
        x = F.relu(self.hidden1(x))
        x = self.dropout(x)
        x = self.hidden2(x)
        return x
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
model = MyModel()
model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


train_dataset = torchvision.datasets.MNIST(
    root='./data_mnist',
    train=True,
    download=True,
    transform=transforms.ToTensor()
)

val_dataset = torchvision.datasets.MNIST(
    root='./data_mnist',
    train=False,
    download=True,
    transform=transforms.ToTensor()
)

train_loader = DataLoader(
    train_dataset,
    batch_size=1000,
    shuffle=True,
    num_workers=8
)

val_loader = DataLoader(
    val_dataset,
    batch_size=100,
    shuffle=False,
    num_workers=20
)

start_time = time.time()
torch.backends.cudnn.benchmark = True

for epoch in range(30):
    train_loss = 0.
    val_loss = 0.
    train_acc = 0.
    val_acc = 0.
    
    num_samples = 0
    for data, target in train_loader:
        data = data.to(device)
        target = target.to(device)
        optimizer.zero_grad()
        num_samples += data.size(0)
        output = model(data.view(data.size(0), -1))
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        train_acc += (torch.argmax(output, 1) == target).float().sum()
    
    print(f"Num datapoints in trainset: {num_samples}")
        
    # with torch.no_grad():     
    #     for data, target in val_loader:
    #         data = data.to(device)
    #         target = target.to(device)
    #         output = model(data.view(data.size(0), -1))
    #         loss = criterion(output, target)            
    #         val_loss += loss.item()
    #         val_acc += (torch.argmax(output, 1) == target).float().sum()
    
    # train_loss /= len(train_loader)
    # train_acc /= len(train_dataset)
    # val_loss /= len(val_loader)
    # val_acc /= len(val_dataset)

    print('Epoch {}, train_loss {}, val_loss {}, train_acc {}, val_acc {}'.format(
        epoch, train_loss, val_loss, train_acc, val_acc))
end_time = time.time()

print(f"pytorch training took: {end_time-start_time} s")