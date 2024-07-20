import torch
import torch.nn as nn
import numpy as np

import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
X_num, y_num = make_regression(n_samples=200, n_features=1, noise=20, random_state=42)

#plt.scatter(X[:,0], y)
#plt.show()


X =  torch.from_numpy(X_num.astype(np.float32))
y = torch.from_numpy(y_num.astype(np.float32))
y =  y.view(y.shape[0], 1)


n_samples, n_features = X.shape

# 2) loss and optimiser 
input_size =  n_features
output_size = 1
model = nn.Linear(input_size, output_size)

learning_rate = 0.01
loss_func = nn.MSELoss()
optimiser = torch.optim.SGD(model.parameters(), lr=learning_rate)


# 4) train on the data in loop
num_epochs = 500

for epoch in range(num_epochs):
    #forward pass and loss
    y_predicted = model(X)
    loss = loss_func(y_predicted, y)
    
    loss.backward()
    
    optimiser.step()
    
    optimiser.zero_grad()
    
    if (epoch + 1) % 10 == 0:
       print("epoch", epoch, " loss: ",  f'{loss.item():.4}')
       
predicted = model(X).detach()

plt.scatter(X, y)
plt.plot(X, predicted)
plt.show()




