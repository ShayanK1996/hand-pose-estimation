# ==========================================
# Baseline CNN model
# ELE 588: Applied Machine Learning
# Author : Shayan Khodabakhsh
# ==========================================
# STEP 1: IMPORTS
# ==========================================
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import json
import os
import random

# 1. Setup GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"1. Setup complete. Using device: {device}")

# 2. Hyperparameters
LEARNING_RATE = 0.001
BATCH_SIZE = 32
EPOCHS = 5
DATA_PATH = "Data/FreiHAND_pub_v2" 

# ==========================================
# STEP 2: TRANSFORMS
# ==========================================
# A. Training Transform: Includes "Safe" Augmentation (Color Only)
train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2), # Safe: Doesn't move pixels
    transforms.ToTensor(), 
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# B. Validation Transform: PURE CLEAN DATA (No Augmentation)
val_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(), 
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ==========================================
# STEP 3: DATASET CLASS
# ==========================================
class FreiHANDDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform 
        
        json_path = os.path.join(root_dir, "training_xyz.json")
        with open(json_path, 'r') as f:
            self.joints_list = json.load(f)

    def __len__(self):
        return len(self.joints_list)

    def __getitem__(self, idx):
        img_name = f"{idx:08d}.jpg" 
        img_path = os.path.join(self.root_dir, "training", "rgb", img_name)
        image = Image.open(img_path).convert('RGB')
        
       
        # This converts the PIL Image to a Tensor
        if self.transform:
            image = self.transform(image)
        
        joints = torch.tensor(self.joints_list[idx], dtype=torch.float32).flatten()
        return image, joints

# ==========================================
# STEP 4: MODEL ARCHITECTURE
# ==========================================
class HandPoseModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Feature Extractor
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        
        # Regressor
        self.fc1 = nn.Linear(256 * 16 * 16, 512) 
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 256)
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(256, 63) 

    def forward(self, x):
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))
        x = self.pool(torch.relu(self.bn3(self.conv3(x))))
        x = self.pool(torch.relu(self.bn4(self.conv4(x))))
        x = x.view(x.size(0), -1)
        x = self.dropout1(torch.relu(self.fc1(x)))
        x = self.dropout2(torch.relu(self.fc2(x)))
        x = self.fc3(x)
        return x

# ==========================================
# STEP 5: SETUP & SPLITTING
# ==========================================
print("2. Preparing Data...")

# A. Create TWO dataset objects with SPECIFIC transforms
train_set_full = FreiHANDDataset(root_dir=DATA_PATH, transform=train_transform)
val_set_full   = FreiHANDDataset(root_dir=DATA_PATH, transform=val_transform)

# B. Generate Indices for Splitting
dataset_size = len(train_set_full)
indices = list(range(dataset_size))
train_size = int(0.8 * dataset_size)
val_size = dataset_size - train_size

# Random shuffle
torch.manual_seed(42)
indices = torch.randperm(dataset_size).tolist()

train_indices = indices[:train_size]
val_indices = indices[train_size:]

# C. Create Subsets using the correct dataset objects
train_dataset = Subset(train_set_full, train_indices)
val_dataset   = Subset(val_set_full, val_indices)

print(f"   Data Split: {len(train_dataset)} Training (Augmented), {len(val_dataset)} Validation (Clean).")

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

print("3. Building Model...")
model = HandPoseModel().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# SCHEDULER: Drops speed at Epoch 20 and 35
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 35], gamma=0.1)

train_loss_history = []
val_loss_history = []
best_val_loss = float('inf') 

# ==========================================
# STEP 6: THE LOOP
# ==========================================
print(f"4. Starting Training ({EPOCHS} Epochs)...")

for epoch in range(EPOCHS):
    # --- PHASE 1: TRAINING ---
    model.train()
    train_loss = 0.0
    
    for i, (images, targets) in enumerate(train_loader):
        images, targets = images.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        
        if i % 200 == 0:
            print(f"   Epoch {epoch+1} | Batch {i} | Train Loss: {loss.item():.4f}")

    avg_train_loss = train_loss / len(train_loader)
    train_loss_history.append(avg_train_loss)

    # --- PHASE 2: VALIDATION ---
    model.eval() 
    val_loss = 0.0
    
    with torch.no_grad():
        for images, targets in val_loader:
            images, targets = images.to(device), targets.to(device)
            outputs = model(images)
            loss = criterion(outputs, targets)
            val_loss += loss.item()
            
    avg_val_loss = val_loss / len(val_loader)
    val_loss_history.append(avg_val_loss)
    
    scheduler.step()

    print(f"=== RESULT EPOCH {epoch+1}: Train Loss = {avg_train_loss:.4f} | Val Loss = {avg_val_loss:.4f} ===")

    # --- PHASE 3: SAVE ---
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), "best_model.pth")
        print(f"   >>> New High Score! Saved model to 'best_model.pth'")

# ==========================================
# FINAL PLOT
# ==========================================
print("Training Complete. Saving plot...")
plt.figure(figsize=(10, 5))
plt.plot(train_loss_history, label='Training Loss')
plt.plot(val_loss_history, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('MSE Loss')
plt.legend()
plt.title('Training vs Validation Loss')
plt.grid(True)
plt.savefig('loss_curve.png')
print("Done.")