# ==========================================
# Hand Pose Estimation - Complete Ablation Study
# K-Fold Cross-Validation on BOTH Models
# ELE 588: Applied Machine Learning
# Author: Shayan Khodabakhsh
#
# UPDATED with optimal hyperparameters from grid search
# ==========================================
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from torch.cuda.amp import autocast, GradScaler
from torchvision import transforms, models
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import json
import os
import time
from sklearn.model_selection import KFold

# ==========================================
# CONFIGURATION
# ==========================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# Cross-validation settings
K_FOLDS = 5
EPOCHS_PER_FOLD = 40

# ==========================================
# HYPERPARAMETERS - OPTIMIZED FROM GRID SEARCH
# ==========================================
BATCH_SIZE = 32

# Custom CNN optimal (from grid search)
LR_CNN = 0.001
WD_CNN = 0.0001

# ResNet50 optimal (from grid search)
LR_RESNET = 0.0003
WD_RESNET = 1e-05

DATA_PATH = "Data/FreiHAND_pub_v2"
USE_MIXED_PRECISION = True
EARLY_STOPPING_PATIENCE = 10

# ==========================================
# TRANSFORMS
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

# ==========================================
# DATASET
# ==========================================
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
        sample_size = min(10000, len(self.joints_list))
        
        for i in range(sample_size):
            joints = np.array(self.joints_list[i])
            root = joints[0:1, :]
            joints_centered = joints - root
            all_joints.append(joints_centered)
        
        all_joints = np.concatenate(all_joints, axis=0)
        self.coord_mean = torch.tensor(all_joints.mean(axis=0), dtype=torch.float32)
        self.coord_std = torch.tensor(all_joints.std(axis=0) + 1e-8, dtype=torch.float32)

    def __len__(self):
        return len(self.joints_list)

    def __getitem__(self, idx):
        img_name = f"{idx:08d}.jpg"
        img_path = os.path.join(self.root_dir, "training", "rgb", img_name)
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
        joints = joints + root.view(-1, 1, 3)
        return joints


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
    distances = torch.sqrt(((pred - target) ** 2).sum(dim=-1))
    return distances.mean().item() * 1000


def calculate_pck(pred, target, threshold_mm=20):
    pred = pred.view(-1, 21, 3)
    target = target.view(-1, 21, 3)
    distances = torch.sqrt(((pred - target) ** 2).sum(dim=-1))
    threshold_m = threshold_mm / 1000.0
    return (distances < threshold_m).float().mean().item() * 100


# ==========================================
# TRAIN SINGLE FOLD
# ==========================================
def train_fold(fold, train_idx, val_idx, dataset_train, dataset_val, model_type):
    """Train one fold, return best metrics"""
    
    # Use model-specific hyperparameters
    if model_type == "resnet50":
        lr = LR_RESNET
        weight_decay = WD_RESNET
    else:
        lr = LR_CNN
        weight_decay = WD_CNN
    
    train_loader = DataLoader(Subset(dataset_train, train_idx), batch_size=BATCH_SIZE, 
                              shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(Subset(dataset_val, val_idx), batch_size=BATCH_SIZE,
                            shuffle=False, num_workers=4, pin_memory=True)
    
    # Fresh model
    if model_type == "resnet50":
        model = ResNet50Model().to(device)
    else:
        model = CustomCNN().to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS_PER_FOLD, eta_min=1e-6)
    scaler = GradScaler() if USE_MIXED_PRECISION else None
    
    best_mpjpe = float('inf')
    best_pck = 0.0
    no_improve = 0
    
    for epoch in range(EPOCHS_PER_FOLD):
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
                if USE_MIXED_PRECISION:
                    with autocast():
                        outputs = model(images)
                else:
                    outputs = model(images)
                all_preds.append(outputs.cpu())
                all_targets.append(targets)
                all_roots.append(roots)
        
        all_preds = torch.cat(all_preds)
        all_targets = torch.cat(all_targets)
        all_roots = torch.cat(all_roots)
        
        pred_denorm = dataset_val.denormalize(all_preds, all_roots)
        target_denorm = dataset_val.denormalize(all_targets, all_roots)
        
        mpjpe = calculate_mpjpe(pred_denorm, target_denorm)
        pck = calculate_pck(pred_denorm, target_denorm)
        
        scheduler.step()
        
        if mpjpe < best_mpjpe:
            best_mpjpe = mpjpe
            best_pck = pck
            no_improve = 0
        else:
            no_improve += 1
        
        if no_improve >= EARLY_STOPPING_PATIENCE:
            break
        
        # Progress every 10 epochs
        if (epoch + 1) % 10 == 0:
            print(f"      Epoch {epoch+1}: MPJPE={mpjpe:.2f}mm, PCK={pck:.1f}%")
    
    return best_mpjpe, best_pck


# ==========================================
# RUN CV FOR ONE MODEL
# ==========================================
def run_cross_validation(model_type, dataset_train, dataset_val, kfold_splits):
    """Run k-fold CV for a single model type"""
    
    if model_type == "resnet50":
        lr, wd = LR_RESNET, WD_RESNET
    else:
        lr, wd = LR_CNN, WD_CNN
    
    print(f"\n{'='*60}")
    print(f"  {model_type.upper()} - {K_FOLDS}-Fold Cross-Validation")
    print(f"  Hyperparameters: LR={lr}, WD={wd}")
    print(f"{'='*60}")
    
    results = {'mpjpe': [], 'pck': []}
    
    for fold, (train_idx, val_idx) in enumerate(kfold_splits):
        print(f"\n  Fold {fold+1}/{K_FOLDS}...")
        fold_start = time.time()
        
        mpjpe, pck = train_fold(
            fold, train_idx.tolist(), val_idx.tolist(),
            dataset_train, dataset_val, model_type
        )
        
        results['mpjpe'].append(mpjpe)
        results['pck'].append(pck)
        
        fold_time = (time.time() - fold_start) / 60
        print(f"    Fold {fold+1} Result: MPJPE={mpjpe:.2f}mm, PCK={pck:.1f}% ({fold_time:.1f}min)")
    
    return results


# ==========================================
# MAIN
# ==========================================
print("="*60)
print("COMPLETE ABLATION STUDY")
print(f"{K_FOLDS}-Fold Cross-Validation on Both Models")
print("="*60)
print("\nHyperparameters (from grid search):")
print(f"  Custom CNN: LR={LR_CNN}, WD={WD_CNN}")
print(f"  ResNet50:   LR={LR_RESNET}, WD={WD_RESNET}")

# Load data once
print("\nLoading dataset...")
dataset_train = FreiHANDDataset(root_dir=DATA_PATH, transform=train_transform)
dataset_val = FreiHANDDataset(root_dir=DATA_PATH, transform=val_transform)
dataset_val.coord_mean = dataset_train.coord_mean
dataset_val.coord_std = dataset_train.coord_std
print(f"Total samples: {len(dataset_train)}")

# Create k-fold splits (same splits for both models = fair comparison)
kfold = KFold(n_splits=K_FOLDS, shuffle=True, random_state=42)
kfold_splits = list(kfold.split(range(len(dataset_train))))

total_start = time.time()

# ==========================================
# RUN BOTH MODELS
# ==========================================
all_results = {}

# 1. Custom CNN (baseline)
all_results['custom_cnn'] = run_cross_validation(
    model_type="custom_cnn",
    dataset_train=dataset_train,
    dataset_val=dataset_val,
    kfold_splits=kfold_splits
)

# 2. ResNet50 (improved)
all_results['resnet50'] = run_cross_validation(
    model_type="resnet50",
    dataset_train=dataset_train,
    dataset_val=dataset_val,
    kfold_splits=kfold_splits
)

total_time = (time.time() - total_start) / 60

# ==========================================
# FINAL COMPARISON
# ==========================================
print("\n" + "="*60)
print("FINAL RESULTS - ABLATION STUDY")
print("="*60)
print(f"Total runtime: {total_time:.1f} minutes ({total_time/60:.1f} hours)")

# Calculate statistics
stats = {}
for model_name, results in all_results.items():
    mpjpe_arr = np.array(results['mpjpe'])
    pck_arr = np.array(results['pck'])
    stats[model_name] = {
        'mpjpe_mean': mpjpe_arr.mean(),
        'mpjpe_std': mpjpe_arr.std(),
        'pck_mean': pck_arr.mean(),
        'pck_std': pck_arr.std(),
        'mpjpe_all': results['mpjpe'],
        'pck_all': results['pck']
    }

# Print comparison table
print("\n" + "-"*60)
print(f"{'Model':<15} {'MPJPE (mm)':<20} {'PCK@20mm (%)':<20}")
print("-"*60)
for model_name, s in stats.items():
    mpjpe_str = f"{s['mpjpe_mean']:.2f} ± {s['mpjpe_std']:.2f}"
    pck_str = f"{s['pck_mean']:.1f} ± {s['pck_std']:.1f}"
    print(f"{model_name:<15} {mpjpe_str:<20} {pck_str:<20}")
print("-"*60)

# Improvement calculation
improvement = (stats['custom_cnn']['mpjpe_mean'] - stats['resnet50']['mpjpe_mean']) / stats['custom_cnn']['mpjpe_mean'] * 100
print(f"\nResNet50 vs Custom CNN: {improvement:.1f}% reduction in MPJPE")

# Per-fold breakdown
print("\n--- Per-Fold Results ---")
print(f"{'Fold':<6} {'Custom CNN':<15} {'ResNet50':<15}")
print("-"*36)
for i in range(K_FOLDS):
    cnn = stats['custom_cnn']['mpjpe_all'][i]
    res = stats['resnet50']['mpjpe_all'][i]
    print(f"{i+1:<6} {cnn:<15.2f} {res:<15.2f}")

# Success criteria
print("\n--- Success Criteria ---")
resnet_mpjpe = stats['resnet50']['mpjpe_mean']
resnet_pck = stats['resnet50']['pck_mean']

if resnet_mpjpe <= 20:
    print(f"✓ Primary Goal: {resnet_mpjpe:.2f}mm ≤ 20mm")
else:
    print(f"✗ Primary Goal: {resnet_mpjpe:.2f}mm > 20mm")

if resnet_mpjpe <= 15:
    print(f"★ Stretch Goal: {resnet_mpjpe:.2f}mm ≤ 15mm")

if resnet_pck >= 70:
    print(f"✓ PCK Goal: {resnet_pck:.1f}% ≥ 70%")
else:
    print(f"✗ PCK Goal: {resnet_pck:.1f}% < 70%")

# What to report
print("\n" + "="*60)
print("FOR YOUR REPORT (copy this):")
print("="*60)
print(f"Custom CNN:  MPJPE = {stats['custom_cnn']['mpjpe_mean']:.2f} ± {stats['custom_cnn']['mpjpe_std']:.2f} mm")
print(f"             PCK@20mm = {stats['custom_cnn']['pck_mean']:.1f} ± {stats['custom_cnn']['pck_std']:.1f}%")
print(f"ResNet50:    MPJPE = {stats['resnet50']['mpjpe_mean']:.2f} ± {stats['resnet50']['mpjpe_std']:.2f} mm")
print(f"             PCK@20mm = {stats['resnet50']['pck_mean']:.1f} ± {stats['resnet50']['pck_std']:.1f}%")
print(f"\n({K_FOLDS}-fold cross-validation)")
print(f"\nHyperparameters selected via grid search:")
print(f"  Custom CNN: LR={LR_CNN}, WD={WD_CNN}")
print(f"  ResNet50:   LR={LR_RESNET}, WD={WD_RESNET}")

# ==========================================
# VISUALIZATION
# ==========================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle(f'Ablation Study: Custom CNN vs ResNet50 ({K_FOLDS}-Fold CV)', fontsize=14, fontweight='bold')

x = np.arange(K_FOLDS)
width = 0.35

# MPJPE comparison
ax1 = axes[0]
bars1 = ax1.bar(x - width/2, stats['custom_cnn']['mpjpe_all'], width, label='Custom CNN', color='#1f77b4', alpha=0.8)
bars2 = ax1.bar(x + width/2, stats['resnet50']['mpjpe_all'], width, label='ResNet50', color='#2ca02c', alpha=0.8)

ax1.axhline(y=stats['custom_cnn']['mpjpe_mean'], color='#1f77b4', linestyle='--', alpha=0.7)
ax1.axhline(y=stats['resnet50']['mpjpe_mean'], color='#2ca02c', linestyle='--', alpha=0.7)
ax1.axhline(y=20, color='red', linestyle=':', linewidth=2, label='Target (20mm)')
ax1.axhline(y=15, color='orange', linestyle=':', linewidth=2, label='Stretch (15mm)')

ax1.set_xlabel('Fold')
ax1.set_ylabel('MPJPE (mm)')
ax1.set_title('Mean Per-Joint Position Error')
ax1.set_xticks(x)
ax1.set_xticklabels([f'Fold {i+1}' for i in range(K_FOLDS)])
ax1.legend()
ax1.grid(True, alpha=0.3, axis='y')

# Add value labels
for bar in bars1:
    ax1.annotate(f'{bar.get_height():.1f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                 xytext=(0, 3), textcoords='offset points', ha='center', va='bottom', fontsize=8)
for bar in bars2:
    ax1.annotate(f'{bar.get_height():.1f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                 xytext=(0, 3), textcoords='offset points', ha='center', va='bottom', fontsize=8)

# PCK comparison
ax2 = axes[1]
bars3 = ax2.bar(x - width/2, stats['custom_cnn']['pck_all'], width, label='Custom CNN', color='#1f77b4', alpha=0.8)
bars4 = ax2.bar(x + width/2, stats['resnet50']['pck_all'], width, label='ResNet50', color='#2ca02c', alpha=0.8)

ax2.axhline(y=stats['custom_cnn']['pck_mean'], color='#1f77b4', linestyle='--', alpha=0.7)
ax2.axhline(y=stats['resnet50']['pck_mean'], color='#2ca02c', linestyle='--', alpha=0.7)
ax2.axhline(y=70, color='red', linestyle=':', linewidth=2, label='Target (70%)')

ax2.set_xlabel('Fold')
ax2.set_ylabel('PCK@20mm (%)')
ax2.set_title('Percentage of Correct Keypoints')
ax2.set_xticks(x)
ax2.set_xticklabels([f'Fold {i+1}' for i in range(K_FOLDS)])
ax2.legend()
ax2.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('ablation_study_cv_results.png', dpi=150, bbox_inches='tight')
print("\nSaved: ablation_study_cv_results.png")

# Save all results
torch.save({
    'stats': stats,
    'all_results': all_results,
    'config': {
        'k_folds': K_FOLDS,
        'epochs_per_fold': EPOCHS_PER_FOLD,
        'batch_size': BATCH_SIZE,
        'hyperparameters': {
            'custom_cnn': {'lr': LR_CNN, 'weight_decay': WD_CNN},
            'resnet50': {'lr': LR_RESNET, 'weight_decay': WD_RESNET}
        }
    }
}, 'ablation_study_cv_results.pth')
print("Saved: ablation_study_cv_results.pth")

print("\n" + "="*60)
print("DONE! Go check your results!")
print("="*60)