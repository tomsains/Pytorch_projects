import torch
import torch.nn as nn
import numpy as np

import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

bc =  datasets.load_breast_cancer()

#plt.scatter(X[:,0], y)
#plt.show()

X, y = bc.data, bc.target

n_samples, n_features =  X.shape

X_train, X_test, y_train, y_test = train_test_split(X ,y, test_size=.2)

# scale the data - always a good idea in logisitc regression
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

X_train =  torch.from_numpy(X_train.astype(np.float32))
X_test =  torch.from_numpy(X_test.astype(np.float32))
y_train =  torch.from_numpy(y_train.astype(np.float32))
y_test =  torch.from_numpy(y_test.astype(np.float32))

y_train =  y_train.view(y_train.shape[0], 1)
y_test =  y_test.view(y_test.shape[0], 1)

# setup the model
class LogisticRegression(nn.Module):
    def __init__(self, n_input_features):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(n_input_features, 1) 
        
    # forward pass
    def forward(self, x):
        y_pred =  torch.sigmoid(self.linear(x))
        return y_pred

# init model and losses 
model = LogisticRegression(n_input_features=n_features)

loss_func =  nn.BCELoss()
optimiser = torch.optim.SGD(model.parameters(), lr=0.01)

num_epochs = 500

loss_que = []

for epoch in range(num_epochs):
    # forward pass
    y_pred = model.forward(x=X_train)
    #print(y_pred)
    
    # calculated the loss
    loss = loss_func(y_pred, y_train)
    loss_que.append(loss.item())
    # back prop
    loss.backward()
    optimiser.step()
    
    # zero the gradients
    optimiser.zero_grad()
    y_pred_round  = y_pred.round()
    acc = y_pred_round.eq(y_train).sum()/float(y_train.shape[0])
    if (epoch + 1) % 10 == 0:
       print("epoch", epoch, " loss: ",  f'{loss.item():.4}', "acc: ", f'{acc:.4}')
       
plt.plot(loss_que)
plt.show()
       
# calculate accuracy
with torch.no_grad():
    y_prediction = model.forward(X_test)
    y_predicted_round  = y_prediction.round()
    #print(y_prediction.shape)
    #print(y_test.shape)
    acc = y_predicted_round.eq(y_test).sum()/float(y_test.shape[0])
    print("accuracy: ", f'{acc:.4}')
    

        
