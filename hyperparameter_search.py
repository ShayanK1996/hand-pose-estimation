# ==========================================
# Hyperparameter Search for Hand Pose Estimation
# ELE 588: Applied Machine Learning
# 
# This finds the best hyperparameters using grid search
# Run time: ~3-4 hours (tests 12 combinations)
# ==========================================
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from torch.cuda.amp import autocast, GradScaler
from torchvision import transforms, models
from PIL import Image
import numpy as np
import json
import os
import time
from itertools import product

# ==========================================
# CONFIGURATION
# ==========================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# Fixed settings
DATA_PATH = "Data/FreiHAND_pub_v2"
EPOCHS = 30  # Enough to see trends, not too long
BATCH_SIZE = 32
USE_MIXED_PRECISION = True
EARLY_STOPPING = 8

# ==========================================
# HYPERPARAMETER SEARCH SPACE
# ==========================================
SEARCH_SPACE = {
    'learning_rate': [1e-3, 3e-4, 1e-4],
    'weight_decay': [1e-3, 1e-4, 1e-5, 0],
}

# This creates 3 × 4 = 12 combinations to test

# ==========================================
# TRANSFORMS & DATASET (same as before)
# ==========================================
train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


class FreiHANDDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        
        json_path = os.path.join(root_dir, "training_xyz.json")
        with open(json_path, 'r') as f:
            self.joints_list = json.load(f)
        
        self._compute_normalization_stats()
    
    def _compute_normalization_stats(self):
        all_joints = []
        for i in range(min(10000, len(self.joints_list))):
            joints = np.array(self.joints_list[i])
            root = joints[0:1, :]
            all_joints.append(joints - root)
        
        all_joints = np.concatenate(all_joints, axis=0)
        self.coord_mean = torch.tensor(all_joints.mean(axis=0), dtype=torch.float32)
        self.coord_std = torch.tensor(all_joints.std(axis=0) + 1e-8, dtype=torch.float32)

    def __len__(self):
        return len(self.joints_list)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, "training", "rgb", f"{idx:08d}.jpg")
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        joints = torch.tensor(self.joints_list[idx], dtype=torch.float32)
        root = joints[0:1, :]
        joints_centered = joints - root
        joints_normalized = (joints_centered - self.coord_mean) / self.coord_std
        
        return image, joints_normalized.flatten(), root.flatten()

    def denormalize(self, normalized_joints, root):
        joints = normalized_joints.view(-1, 21, 3)
        joints = joints * self.coord_std.to(joints.device) + self.coord_mean.to(joints.device)
        return joints + root.view(-1, 1, 3)


# ==========================================
# MODELS
# ==========================================
class CustomCNN(nn.Module):
    """4-block CNN as specified in proposal"""
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True), nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True), nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True), nn.MaxPool2d(2, 2),
        )
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 16 * 16, 512), nn.ReLU(inplace=True), nn.Dropout(0.5),
            nn.Linear(512, 256), nn.ReLU(inplace=True), nn.Dropout(0.3),
            nn.Linear(256, 63)
        )

    def forward(self, x):
        return self.regressor(self.features(x))


class ResNet50Model(nn.Module):
    """ResNet50 with pretrained ImageNet weights"""
    def __init__(self):
        super().__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2048, 512), nn.BatchNorm1d(512), nn.ReLU(inplace=True), nn.Dropout(0.3),
            nn.Linear(512, 256), nn.ReLU(inplace=True), nn.Dropout(0.2),
            nn.Linear(256, 63)
        )

    def forward(self, x):
        return self.head(self.backbone(x))


# ==========================================
# METRICS
# ==========================================
def calculate_mpjpe(pred, target):
    pred = pred.view(-1, 21, 3)
    target = target.view(-1, 21, 3)
    return torch.sqrt(((pred - target) ** 2).sum(dim=-1)).mean().item() * 1000


def calculate_pck(pred, target, threshold_mm=20):
    pred = pred.view(-1, 21, 3)
    target = target.view(-1, 21, 3)
    distances = torch.sqrt(((pred - target) ** 2).sum(dim=-1))
    threshold_m = threshold_mm / 1000.0
    return (distances < threshold_m).float().mean().item() * 100


# ==========================================
# TRAIN WITH ONE HYPERPARAMETER SET
# ==========================================
def train_with_params(model_type, lr, weight_decay, train_loader, val_loader, dataset_val):
    """Train model with given hyperparameters, return best MPJPE"""
    
    if model_type == "resnet50":
        model = ResNet50Model().to(device)
    else:
        model = CustomCNN().to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
    scaler = GradScaler() if USE_MIXED_PRECISION else None
    
    best_mpjpe = float('inf')
    best_pck = 0.0
    no_improve = 0
    
    for epoch in range(EPOCHS):
        # Train
        model.train()
        for images, targets, _ in train_loader:
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            
            if USE_MIXED_PRECISION:
                with autocast():
                    loss = criterion(model(images), targets)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss = criterion(model(images), targets)
                loss.backward()
                optimizer.step()
        
        # Validate
        model.eval()
        all_preds, all_targets, all_roots = [], [], []
        with torch.no_grad():
            for images, targets, roots in val_loader:
                images = images.to(device)
                with autocast():
                    outputs = model(images)
                all_preds.append(outputs.cpu())
                all_targets.append(targets)
                all_roots.append(roots)
        
        pred_denorm = dataset_val.denormalize(torch.cat(all_preds), torch.cat(all_roots))
        target_denorm = dataset_val.denormalize(torch.cat(all_targets), torch.cat(all_roots))
        mpjpe = calculate_mpjpe(pred_denorm, target_denorm)
        pck = calculate_pck(pred_denorm, target_denorm)
        
        scheduler.step()
        
        if mpjpe < best_mpjpe:
            best_mpjpe = mpjpe
            best_pck = pck
            no_improve = 0
        else:
            no_improve += 1
        
        if no_improve >= EARLY_STOPPING:
            break
    
    return best_mpjpe, best_pck


# ==========================================
# MAIN: GRID SEARCH
# ==========================================
print("="*60)
print("HYPERPARAMETER SEARCH")
print("="*60)

# Load data
print("\nLoading data...")
dataset_train = FreiHANDDataset(root_dir=DATA_PATH, transform=train_transform)
dataset_val = FreiHANDDataset(root_dir=DATA_PATH, transform=val_transform)
dataset_val.coord_mean = dataset_train.coord_mean
dataset_val.coord_std = dataset_train.coord_std

# Train/val split (80/20)
dataset_size = len(dataset_train)
torch.manual_seed(42)
indices = torch.randperm(dataset_size).tolist()
train_size = int(0.8 * dataset_size)

train_loader = DataLoader(
    Subset(dataset_train, indices[:train_size]),
    batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True
)
val_loader = DataLoader(
    Subset(dataset_val, indices[train_size:]),
    batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True
)

print(f"Train: {train_size}, Val: {dataset_size - train_size}")

# Generate all combinations
param_combinations = list(product(
    SEARCH_SPACE['learning_rate'],
    SEARCH_SPACE['weight_decay']
))

# Models to test
MODELS = ['custom_cnn', 'resnet50']

total_combinations = len(param_combinations) * len(MODELS)
print(f"\nTesting {total_combinations} total combinations ({len(param_combinations)} per model × {len(MODELS)} models)...")
print(f"Search space:")
print(f"  Models: {MODELS}")
print(f"  Learning rates: {SEARCH_SPACE['learning_rate']}")
print(f"  Weight decays: {SEARCH_SPACE['weight_decay']}")

# Run search for BOTH models
all_results = {}
total_start = time.time()

for model_type in MODELS:
    print(f"\n{'='*60}")
    print(f"  SEARCHING: {model_type.upper()}")
    print(f"{'='*60}")
    
    results = []
    
    for i, (lr, wd) in enumerate(param_combinations):
        print(f"\n  [{i+1}/{len(param_combinations)}] LR={lr}, WD={wd}")
        start = time.time()
        
        mpjpe, pck = train_with_params(model_type, lr, wd, train_loader, val_loader, dataset_val)
        elapsed = (time.time() - start) / 60
        
        results.append({
            'learning_rate': lr,
            'weight_decay': wd,
            'mpjpe': mpjpe,
            'pck': pck
        })
        
        print(f"      Result: MPJPE = {mpjpe:.2f}mm, PCK@20mm = {pck:.1f}% ({elapsed:.1f}min)")
    
    all_results[model_type] = results

total_time = (time.time() - total_start) / 60

# ==========================================
# RESULTS
# ==========================================
print("\n" + "="*60)
print("HYPERPARAMETER SEARCH RESULTS")
print("="*60)
print(f"Total time: {total_time:.1f} minutes ({total_time/60:.1f} hours)")

best_params = {}

for model_type in MODELS:
    results = all_results[model_type]
    results_sorted = sorted(results, key=lambda x: x['mpjpe'])
    
    print(f"\n{'='*60}")
    print(f"  {model_type.upper()} RESULTS")
    print(f"{'='*60}")
    
    print(f"\n{'Rank':<6} {'LR':<12} {'Weight Decay':<15} {'MPJPE (mm)':<12} {'PCK@20mm':<10}")
    print("-"*60)
    for rank, r in enumerate(results_sorted, 1):
        marker = " ← BEST" if rank == 1 else ""
        print(f"{rank:<6} {r['learning_rate']:<12} {r['weight_decay']:<15} {r['mpjpe']:<12.2f} {r['pck']:<10.1f}{marker}")
    
    best_params[model_type] = results_sorted[0]

# Summary
print("\n" + "="*60)
print("BEST HYPERPARAMETERS SUMMARY")
print("="*60)
print(f"\n{'Model':<15} {'Best LR':<12} {'Best WD':<15} {'MPJPE (mm)':<12} {'PCK@20mm':<10}")
print("-"*64)
for model_type, best in best_params.items():
    print(f"{model_type:<15} {best['learning_rate']:<12} {best['weight_decay']:<15} {best['mpjpe']:<12.2f} {best['pck']:<10.1f}")

print("\n" + "="*60)
print("FOR YOUR CV SCRIPT - UPDATE THESE VALUES:")
print("="*60)
print(f"\n# Custom CNN")
print(f"CUSTOM_CNN_LR = {best_params['custom_cnn']['learning_rate']}")
print(f"CUSTOM_CNN_WD = {best_params['custom_cnn']['weight_decay']}")
print(f"\n# ResNet50")
print(f"RESNET50_LR = {best_params['resnet50']['learning_rate']}")
print(f"RESNET50_WD = {best_params['resnet50']['weight_decay']}")

print("\n--- For Your Report ---")
print("Hyperparameters were selected via grid search over")
print(f"learning rates {SEARCH_SPACE['learning_rate']} and")
print(f"weight decays {SEARCH_SPACE['weight_decay']}.")
print(f"\nCustom CNN: LR={best_params['custom_cnn']['learning_rate']}, WD={best_params['custom_cnn']['weight_decay']}")
print(f"ResNet50: LR={best_params['resnet50']['learning_rate']}, WD={best_params['resnet50']['weight_decay']}")

# Save results
torch.save({
    'all_results': all_results,
    'best_params': best_params,
    'search_space': SEARCH_SPACE
}, 'hyperparam_search_results.pth')
print("\nSaved: hyperparam_search_results.pth")

print("\n" + "="*60)
print("DONE!")
print("="*60)