import torchvision.models as models
from PIL import Image
import pandas as pd
import torch
import os
import time
from torchvision import transforms
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import requests
import zipfile

torch.manual_seed(0)

# -----------------------------
# DOWNLOAD DATA
# -----------------------------

urls = [
    "https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DL0321EN/data/images/Positive_tensors.zip",
    "https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DL0321EN/data/images/Negative_tensors.zip"
]

for url in urls:
    print(f"[INFO] Downloading from: {url}")
    r = requests.get(url, allow_redirects=True)
    open(url.split('/')[-1], 'wb').write(r.content)
    print(f"[INFO] Downloaded {url.split('/')[-1]} successfully.")

file_names = ["Positive_tensors.zip", "Negative_tensors.zip"]

for file_name in file_names:
    print(f"[INFO] Extracting {file_name} ...")
    with zipfile.ZipFile(file_name, 'r') as zip_ref:
        zip_ref.extractall('.')
    print(f"[INFO] Extracted {file_name} successfully.")


print("[INFO] Data downloaded and extracted successfully.")
# -----------------------------
# DATASET CLASS
# -----------------------------

class Dataset(Dataset):
    def __init__(self, transform=None, train=True):
        directory = "C:\\Users\\marcnune\\Documents\\GitHub\\-crack-detector-resnet-pytorch"
        positive = "Positive_tensors"
        negative = 'Negative_tensors'

        positive_file_path = os.path.join(directory, positive)
        negative_file_path = os.path.join(directory, negative)
        positive_files = [os.path.join(positive_file_path, file) for file in os.listdir(positive_file_path) if file.endswith(".pt")]
        negative_files = [os.path.join(negative_file_path, file) for file in os.listdir(negative_file_path) if file.endswith(".pt")]
        
        number_of_samples = len(positive_files) + len(negative_files)
        self.all_files = [None] * number_of_samples
        self.all_files[::2] = positive_files
        self.all_files[1::2] = negative_files
        
        self.transform = transform
        self.Y = torch.zeros([number_of_samples]).type(torch.LongTensor)
        self.Y[::2] = 1
        self.Y[1::2] = 0
        
        if train:
            self.all_files = self.all_files[0:30000]
            self.Y = self.Y[0:30000]
            self.len = len(self.all_files)
        else:
            self.all_files = self.all_files[30000:]
            self.Y = self.Y[30000:]
            self.len = len(self.all_files)

    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        image = torch.load(self.all_files[idx])
        y = self.Y[idx]
        if self.transform:
            image = self.transform(image)
        return image, y

train_dataset = Dataset(train=True)
validation_dataset = Dataset(train=False)

# -----------------------------
# PRE-TRAINED MODEL: RESNET18
# -----------------------------

print("[INFO] Loading pre-trained ResNet18...")

# Load pre-trained ResNet18
model = models.resnet18(pretrained=True)

# Set parameters to non-trainable
for param in model.parameters():
    param.requires_grad = False

# Modify the output layer
model.fc = nn.Linear(512, 2)  # As we have two classes: positive and negative

print("[INFO] Modified the last layer for binary classification.")

# -----------------------------
# TRAIN THE MODEL
# -----------------------------

# Loss function
criterion = nn.CrossEntropyLoss()

# DataLoaders
train_loader = DataLoader(dataset=train_dataset, batch_size=100)
validation_loader = DataLoader(dataset=validation_dataset, batch_size=100)

# Optimizer
optimizer = torch.optim.Adam([parameters for parameters in model.parameters() if parameters.requires_grad], lr=0.001)

# Training loop
n_epochs = 2
loss_list = []
accuracy_list = []
N_test = len(validation_dataset)
start_time = time.time()

for epoch in range(n_epochs):
    for x, y in train_loader:
        model.train()
        optimizer.zero_grad()
        z = model(x)
        loss = criterion(z, y)
        loss.backward()
        optimizer.step()
        loss_list.append(loss.data)
    
    correct = 0
    for x_test, y_test in validation_loader:
        model.eval()
        z = model(x_test)
        _, yhat = torch.max(z.data, 1)
        correct += (yhat == y_test).sum().item()
    accuracy = correct / N_test

# Plotting
print("Accuracy: ", accuracy)
plt.plot(loss_list)
plt.xlabel("iteration")
plt.ylabel("loss")
plt.show()

# -----------------------------
# MISCLASSIFIED SAMPLES
# -----------------------------

misclassified_samples = []
for x, y in validation_loader:
    z = model(x)
    _, yhat = torch.max(z.data, 1)
    misclassified_indices = torch.where(yhat != y)[0]
    misclassified_samples += [(x[i], yhat[i], y[i]) for i in misclassified_indices]

# Displaying first 4 misclassified samples
for index, (img, pred, actual) in enumerate(misclassified_samples[:4]):
    print(f"Sample {index + 1} - Predicted: {pred}, Actual: {actual}")
    plt.imshow(img[0], cmap='gray')
    plt.show()
