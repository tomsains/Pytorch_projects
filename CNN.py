import torch
import torch.nn as nn
import torch.nn.functional as F 
import torchvision.transforms as transforms
import matplotlib.pyplot as plt 
import numpy as np 


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# hyperparams
num_epoch = 4
batch_size = 4 
learning_rate = 0.001


transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,0.5,0.5),
                                                     (0.5,0.5,0.5))])


train_dataset = torchvision.datasets.CIFAR10(root='./data/',
                                             train=True, download=True,
                                             transform=transform)

test_dataset = torchvision.datasets.CIFAR10(root='./data/',
                                             train=False, download=True,
                                             transform=transform)


train_loader = torch.utils.data.Dataloader(train_dataset, batch_size =batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.Dataloader(test_dataset, batch_size =batch_size,
                                           shuffle=True)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog',
           'frog', 'horse', 'ship', 'truck')


class CNN(nn.Module):
    def __init__(self):
        self.cnn1 = nn.Conv2d(3, 6, 5)
        self.mp1 = nn.MaxPool2d(2,2)
        self.cnn2 = nn.Conv2d(6,16,5)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
    
    def forward(self, x):
        pass
    
    
model = CNN().to(device)


loss_func = nn.CrossEntropyLoss()
optimiser = torch.optim.SGD(model.parameters(), lr=learning_rate)


n_total_steps = len(train_loader)

for epoch in range(num_epoch):
    for i, (images, labels) in enumerate(train_loader):
        images =images.to(device)
        labels = labels.to(device)
        
        
        outputs = model(images)
        loss = loss_func(outputs, labels)
        
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
        
        if (i + 1) % 100 == 0:
            print("epoch", epoch, " step: ", i+1, " loss: ", 
                  f'{loss.item():.4}')
            
            
with torch.no_grad():
    n_correct = 0
    n_samples = 0
    for images, labels in test_loader:
        images = images.reshape(-1, 28*28).to(device)
        labels = labels.to(device)
        outputs = model(images)
        # max returns (value ,index)
        _, predicted = torch.max(outputs.data, 1)
        n_samples += labels.size(0)
        n_correct += (predicted == labels).sum().item()

    acc = 100.0 * n_correct / n_samples
    print(f'Model accuracy on test set: {acc} %')