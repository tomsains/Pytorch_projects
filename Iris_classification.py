import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec



iris = pd.read_csv("https://gist.githubusercontent.com/netj/8836201/raw/6f9306ad21398ea43cba4f7d537619d0e07d5ae3/iris.csv")
iris.head()

num_epochs = 200

X = torch.tensor(iris.drop("variety", axis=1).values, dtype=torch.float)
y = torch.tensor(
    [0 if vty == "Setosa" else 1 if vty == "Versicolor" else 2 for vty in iris["variety"]], 
    dtype=torch.long
)

print(X.shape, y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=42)

train_data = TensorDataset(X_train, y_train)
test_data = TensorDataset(X_test, y_test)

train_loader = DataLoader(train_data, shuffle=True, batch_size=12)
test_loader = DataLoader(test_data, batch_size=len(test_data.tensors[0]))

print("Training data batches:")
for X, y in train_loader:
    print(X.shape, y.shape)
    
print("\nTest data batches:")
for X, y in test_loader:
    print(X.shape, y.shape)
    
    
class classifier(nn.Module):
    def __init__(self, n_inputs =4, n_hidden=16, n_outputs =3):
        super().__init__()
        self.l1 = nn.Linear(in_features = n_inputs, out_features =n_hidden)
        self.l2 = nn.Linear(in_features = n_hidden, out_features =n_hidden)
        self.l3 = nn.Linear(in_features = n_hidden, out_features = n_outputs)
        
    def forward(self, x):
        x =F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        return(self.l3(x))
    
model = classifier()

loss_func = nn.CrossEntropyLoss()

optimiser = torch.optim.Adam(params=model.parameters(), lr = 0.01)

train_accuracies, test_accuracies = [], []

for i in range(num_epochs):
    for X, y in train_loader:
        predictions =  model(X)
        pred_labels = torch.argmax(predictions, axis=1)
        loss = loss_func(predictions, y)
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
        
        train_accuracies.append(
            100 * torch.mean((pred_labels == y).float()).item()
        )
        
        if i % 10 == 0:
            print("accuracy:", train_accuracies [-1])
            
with  torch.no_grad():
    for X, y in test_loader:
        predictions =  model(X)
        pred_labels = torch.argmax(predictions, axis=1)
        
    train_accuracies.append(
            100 * torch.mean((pred_labels == y).float()).item()
        )
        
    
    
    


        