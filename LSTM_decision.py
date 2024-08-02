import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# Load your data frame
num_trials = 10
time_steps_per_trial = 50
data = pd.read_pickle("/Users/thomassainsbury/Documents/Mathis_lab/Aug_Reg/Marmot_2024-07-29_1.pickle")

# Flatten state data and construct the dataframe
data["state"] = np.vstack(data["state"])
df = pd.DataFrame({
    "time_step": data["step_time"],
    "trial": data["episode"],
    "x_position": data["state"][:, 0],
    "y_position": data["state"][:, 1],
    "heading_direction": data["state"][:, 2],
    "photodiode_state": data["state"][:, -2],
    "at_criterion": data["state"][:, -1],
    "ITI": data["state"][:, 4],
})

df["choice"] = (df["ITI"].diff() == 1) & (df["x_position"] > 0)
df["choice"] = df["choice"].astype(int)  # Convert boolean to int

# Parameters
window_size = 5  # Number of previous time steps to consider

# Data Preparation
features = df[['time_step', 'x_position', 'y_position', 'heading_direction']]
target = df['choice']

# Normalize the features
scaler = MinMaxScaler()
features_scaled = scaler.fit_transform(features)

# Reshape the data to (trials, time_steps, features)
grouped = df.groupby('trial')
X = []
y = []

for _, group in grouped:
    group_features = group[['time_step', 'x_position', 'y_position', 'heading_direction']].values
    group_target = group['choice'].values
    for i in range(window_size, len(group)):
        X.append(group_features[i-window_size:i])
        y.append(group_target[i])

X = np.array(X)
y = np.array(y)

# Convert to PyTorch tensors
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_tensor, y_tensor, test_size=0.2, random_state=42)

# Define the LSTM Model
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        h_0 = torch.zeros(1, x.size(0), hidden_size).to(x.device)
        c_0 = torch.zeros(1, x.size(0), hidden_size).to(x.device)
        out, _ = self.lstm(x, (h_0, c_0))
        out = self.fc(out[:, -1, :])  # Use only the output of the last time step
        out = self.sigmoid(out)
        return out

input_size = 4  # time_step, x_position, y_position, heading_direction
hidden_size = 64
output_size = 1
num_layers = 1

model = LSTMModel(input_size, hidden_size, output_size, num_layers)

# Train the Model
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 10
batch_size = 32

train_loader = torch.utils.data.DataLoader(
    dataset=list(zip(X_train, y_train)), batch_size=batch_size, shuffle=True)

test_loader = torch.utils.data.DataLoader(
    dataset=list(zip(X_test, y_test)), batch_size=batch_size, shuffle=False)

for epoch in range(num_epochs):
    model.train()
    for i, (x_batch, y_batch) in enumerate(train_loader):
        outputs = model(x_batch)
        loss = criterion(outputs.squeeze(), y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# Evaluate the Model and Plot Probabilities
model.eval()
with torch.no_grad():
    all_probs = []
    all_labels = []
    for x_batch, y_batch in test_loader:
        outputs = model(x_batch).squeeze().cpu().numpy()
        all_probs.append(outputs)
        all_labels.append(y_batch.cpu().numpy())

    all_probs = np.concatenate(all_probs)
    all_labels = np.concatenate(all_labels)

    # Plot the results
    plt.figure(figsize=(12, 6))
    plt.plot(all_probs, label='Predicted Choice Probability')
    plt.plot(all_labels, label='Actual Choice', alpha=0.5)
    plt.xlabel('Sample Index')
    plt.ylabel('Choice Probability')
    plt.title('Choice Probability Across Test Set Samples')
    plt.legend()
    plt.show()
