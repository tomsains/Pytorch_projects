import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# device config
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

input_size = 784
hidden_size = 100
num_classes = 10

num_epochs = 2
batch_size = 100
learning_rate = 0.001

train_dataset = torchvision.datasets.MNIST(root="./data",
                                           train=True, download=True, transform=transforms.ToTensor())


test_dataset = torchvision.datasets.MNIST(root="./data",
                                           train=False, transform=transforms.ToTensor())


train_loader =  torch.utils.data.DataLoader(dataset=train_dataset, 
                                            batch_size=batch_size, shuffle=True)


test_loader =  torch.utils.data.DataLoader(dataset=test_dataset, 
                                            batch_size=batch_size, shuffle=True)

examples = iter(train_loader)
samples, labels = next(examples)
print(samples.shape, labels.shape)



class perceptron(nn.Module):
    def __init__(self, input_size,  hidden_size, num_classes):
        super(perceptron, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        return(out)
    
model =  perceptron(input_size, hidden_size, num_classes)

loss_fun = nn.CrossEntropyLoss()
optimiser = torch.optim.Adam(model.parameters(), lr=0.01)

# training loop
n_total_steps =  len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.reshape(-1, 28*28).to(device)
        labels = labels.to(device)
        
        outputs = model(images)
        loss = loss_fun(outputs, labels)
        
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
        output = model(images)
        
        # returns value and index but we want the index aka the class
        _, predictions = torch.max(outputs, 1)
        n_samples += labels.shape[0]
        n_correct = (predictions == labels).sum().item()
        
    acc = 100.0 * n_correct / n_samples
    print("acc", f'{acc:.4}')
        
        
        
        
        
        
        
        


