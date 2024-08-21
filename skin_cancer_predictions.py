import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm_notebook as tqdm
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SkinLesionDataset(Dataset):
    def __init__(self, root, metadata_csv, image_folder, transform=None):
        self.metadata = pd.read_csv(root + metadata_csv)
        self.image_folder = root + image_folder
        self.transform = transform

    def __len__(self):
        return 20000#len(self.metadata)

    def __getitem__(self, idx):
        img_name = self.metadata.iloc[idx]['isic_id']
        img_path = f"{self.image_folder}/image/{img_name}.jpg"
        image = Image.open(img_path)

        if self.transform:
            image = self.transform(image)

        # Extract tabular data
        tabular_data = self.metadata.iloc[idx][['age_approx', 'tbp_lv_areaMM2']].values.astype(np.float32)

        # Convert 'sex' to a numerical value: 1 for 'male', 0 for 'female'
        sex = self.metadata.iloc[idx]['sex']
        sex_numeric = 1.0 if sex == 'male' else 0.0
        
        tabular_data = np.nan_to_num(tabular_data, nan=0)

        # Append the numeric 'sex' value to tabular data
        tabular_data = np.append(tabular_data, sex_numeric)

        # Convert to tensor
        tabular_data = torch.tensor(tabular_data, dtype=torch.float32)

        # Target label
        label = torch.tensor(self.metadata.iloc[idx]['target'], dtype=torch.float32)
        
        return image, tabular_data, label
    
    
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])



class CnnNetTabular(nn.Module):
    def __init__(self):
        super(CnnNetTabular, self).__init__()
        
        self.cnn = models.efficientnet_b0(pretrained=False)
        self.cnn.load_state_dict(torch.load("/kaggle/input/efficent-net-b0/pytorch/default/1/efficientnet_b0.pth"))

        # Replace the final classification layer with an identity layer
        self.cnn.classifier[1] = nn.Identity()

        # Simple feedforward for tabular data
        self.tabular_net = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
        )

        # Final combined classifier
        self.classifier = nn.Sequential(
            nn.Linear(1280 + 32, 128),  # Adjust the input size to match the concatenated feature size
            nn.ReLU(),
            nn.Linear(128, 3)  # Adjust to output the correct number of classes
        )

    def forward(self, image, tabular_data):
        img_features = self.cnn(image)  # Features from the CNN
        tab_features = self.tabular_net(tabular_data)  # Features from tabular data
        combined_features = torch.cat((img_features, tab_features), dim=1)
        output = self.classifier(combined_features)
        return output

root = '/kaggle/input/isic-2024-challenge/'

# Initialize dataset and dataloader
dataset = SkinLesionDataset(root=root, metadata_csv='train-metadata.csv', image_folder='train-image', transform=transform)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

# Initialize model, loss function, and optimizer
model = CnnNetTabular().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(2):  # Number of epochs
    model.train()
    running_loss = 0.0
    loss_through_training = []
    i = 0
    for images, tabular_data, labels in tqdm(dataloader):
        
        if (i % 100) == 0.:
            print("batch: ", i)
        i += 1
        images = images.to(device)
        tabular_data = tabular_data.to(device)
        
        labels = labels.to(device)
        labels=labels.to(torch.int64)
        optimizer.zero_grad()
        outputs = model(images, tabular_data)
        #print(outputs [0,:])
       
        loss = criterion(outputs.squeeze(), labels)
        loss.backward()
        #print(loss)
        optimizer.step()
        running_loss += loss.item()
        lo
        
        if torch.isnan(images).any() or torch.isnan(tabular_data).any():
            print("NaN detected in input data!")
            break

        if torch.isinf(images).any() or torch.isinf(tabular_data).any():
            print("Inf detected in input data!")
            break
        
        #print(outputs.size)
        if torch.isnan(loss).any():
            print("Warning: Loss has become NaN!")
            break
       

    print(f"Epoch {epoch+1}, Loss: {running_loss/len(dataloader)}")