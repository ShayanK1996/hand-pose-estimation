# ==========================================
# ELE 588: Applied Machine Learning
# Author: Shayan Khodabakhsh
# Usage:
#   Run 1: USE_RESNET = False  → Custom CNN (proposal baseline)
#   Run 2: USE_RESNET = True   → ResNet50 (improved version)
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

# ==========================================
# CONFIGURATION
# ==========================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"1. Setup complete. Using device: {device}")

# ============================================
# >>> TOGGLE THIS FOR ABLATION STUDY <<<
# ============================================
USE_RESNET = True  # False = Custom CNN (baseline), True = ResNet50 (improved)
# ============================================

# Hyperparameters
LEARNING_RATE = 3e-4 if USE_RESNET else 1e-3
BATCH_SIZE = 32
EPOCHS = 60
DATA_PATH = "Data/FreiHAND_pub_v2"

# Training settings
WEIGHT_DECAY = 1e-4
EARLY_STOPPING_PATIENCE = 15
USE_MIXED_PRECISION = True

# Output naming based on model type
MODEL_NAME = "resnet50" if USE_RESNET else "custom_cnn"
SAVE_PREFIX = f"results_{MODEL_NAME}"

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
# MODEL 1: Custom CNN (Proposal Baseline)
# ==========================================
class CustomCNN(nn.Module):
    """
    4-block CNN as specified in proposal Figure 1.
    - Conv blocks: 32 → 64 → 128 → 256 filters
    - FC layers: 512 → 256 → 63
    """
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            # Block 2
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            # Block 3
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            # Block 4
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 16 * 16, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 63)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.regressor(x)
        return x


# ==========================================
# MODEL 2: ResNet50 (Improved Version)
# ==========================================
class ResNet50Model(nn.Module):
    """
    ResNet50 backbone with pretrained ImageNet weights.
    Demonstrates the impact of transfer learning.
    """
    def __init__(self):
        super().__init__()
        
        # Load pretrained ResNet50
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        
        # Remove the final classification layer
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        
        # Custom regression head
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, 63)
        )
        
        # Initialize head weights
        for m in self.head.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.backbone(x)
        x = self.head(x)
        return x


# ==========================================
# EVALUATION METRICS
# ==========================================
def calculate_mpjpe(pred, target):
    """Mean Per-Joint Position Error in millimeters"""
    pred = pred.view(-1, 21, 3)
    target = target.view(-1, 21, 3)
    distances = torch.sqrt(((pred - target) ** 2).sum(dim=-1))
    return distances.mean().item() * 1000


def calculate_pck(pred, target, threshold_mm=20):
    """Percentage of Correct Keypoints within threshold"""
    pred = pred.view(-1, 21, 3)
    target = target.view(-1, 21, 3)
    distances = torch.sqrt(((pred - target) ** 2).sum(dim=-1))
    threshold_m = threshold_mm / 1000.0
    correct = (distances < threshold_m).float()
    return correct.mean().item() * 100


def calculate_per_joint_error(pred, target):
    """Per-joint error for analysis"""
    pred = pred.view(-1, 21, 3)
    target = target.view(-1, 21, 3)
    distances = torch.sqrt(((pred - target) ** 2).sum(dim=-1))
    return distances.mean(dim=0) * 1000


# ==========================================
# SETUP
# ==========================================
print("2. Preparing Data...")

train_set_full = FreiHANDDataset(root_dir=DATA_PATH, transform=train_transform)
val_set_full = FreiHANDDataset(root_dir=DATA_PATH, transform=val_transform)
val_set_full.coord_mean = train_set_full.coord_mean
val_set_full.coord_std = train_set_full.coord_std

# Split
dataset_size = len(train_set_full)
torch.manual_seed(42)
indices = torch.randperm(dataset_size).tolist()

train_size = int(0.8 * dataset_size)
train_dataset = Subset(train_set_full, indices[:train_size])
val_dataset = Subset(val_set_full, indices[train_size:])

print(f"   Train: {len(train_dataset)}, Val: {len(val_dataset)}")

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=4, pin_memory=True, persistent_workers=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=4, pin_memory=True, persistent_workers=True)

# Model selection
print(f"\n3. Building Model...")
print(f"   >>> Using: {MODEL_NAME.upper()} <<<")

if USE_RESNET:
    model = ResNet50Model().to(device)
else:
    model = CustomCNN().to(device)

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"   Total parameters: {total_params:,}")
print(f"   Trainable: {trainable_params:,}")

# Training setup
criterion = nn.MSELoss()
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=15, T_mult=2, eta_min=1e-6)
scaler = GradScaler() if USE_MIXED_PRECISION else None

# History
history = {'train_loss': [], 'val_loss': [], 'mpjpe': [], 'pck': [], 'lr': []}
best_val_loss = float('inf')
best_mpjpe = float('inf')
epochs_without_improvement = 0

# ==========================================
# TRAINING LOOP
# ==========================================
print(f"\n4. Starting Training ({EPOCHS} Epochs)...")
print(f"   Model: {MODEL_NAME}")
print(f"   LR: {LEARNING_RATE}, Weight Decay: {WEIGHT_DECAY}")
print("=" * 70)

start_time = time.time()

for epoch in range(EPOCHS):
    epoch_start = time.time()
    
    # --- TRAINING ---
    model.train()
    train_loss = 0.0
    
    for i, (images, targets, roots) in enumerate(train_loader):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        
        optimizer.zero_grad(set_to_none=True)
        
        if USE_MIXED_PRECISION:
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, targets)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        
        train_loss += loss.item()
        
        if i % 200 == 0:
            print(f"   Epoch {epoch+1} | Batch {i}/{len(train_loader)} | Loss: {loss.item():.4f}")

    avg_train_loss = train_loss / len(train_loader)
    history['train_loss'].append(avg_train_loss)

    # --- VALIDATION ---
    model.eval()
    val_loss = 0.0
    all_preds, all_targets, all_roots = [], [], []
    
    with torch.no_grad():
        for images, targets, roots in val_loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            
            if USE_MIXED_PRECISION:
                with autocast():
                    outputs = model(images)
                    loss = criterion(outputs, targets)
            else:
                outputs = model(images)
                loss = criterion(outputs, targets)
            
            val_loss += loss.item()
            all_preds.append(outputs.cpu())
            all_targets.append(targets.cpu())
            all_roots.append(roots)
    
    avg_val_loss = val_loss / len(val_loader)
    history['val_loss'].append(avg_val_loss)
    
    # Metrics
    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    all_roots = torch.cat(all_roots)
    
    pred_denorm = val_set_full.denormalize(all_preds, all_roots)
    target_denorm = val_set_full.denormalize(all_targets, all_roots)
    
    epoch_mpjpe = calculate_mpjpe(pred_denorm, target_denorm)
    epoch_pck = calculate_pck(pred_denorm, target_denorm)
    
    history['mpjpe'].append(epoch_mpjpe)
    history['pck'].append(epoch_pck)
    
    scheduler.step()
    current_lr = optimizer.param_groups[0]['lr']
    history['lr'].append(current_lr)
    
    epoch_time = time.time() - epoch_start

    # --- RESULTS ---
    print(f"\n{'='*70}")
    print(f"EPOCH {epoch+1}/{EPOCHS} [{MODEL_NAME.upper()}] ({epoch_time:.1f}s)")
    print(f"  Loss: Train={avg_train_loss:.4f}, Val={avg_val_loss:.4f}")
    print(f"  MPJPE: {epoch_mpjpe:.2f}mm | PCK@20mm: {epoch_pck:.1f}%")
    
    if epoch_mpjpe <= 20:
        print(f"  ✓ TARGET ACHIEVED (≤20mm)!")
    if epoch_mpjpe <= 15:
        print(f"  ★ STRETCH GOAL (≤15mm)!")
    
    # --- CHECKPOINT ---
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        best_mpjpe = epoch_mpjpe
        epochs_without_improvement = 0
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'mpjpe': epoch_mpjpe,
            'pck': epoch_pck,
            'model_type': MODEL_NAME,
            'coord_mean': train_set_full.coord_mean,
            'coord_std': train_set_full.coord_std,
        }, f"best_model_{MODEL_NAME}.pth")
        print(f"  >>> SAVED best_model_{MODEL_NAME}.pth (MPJPE: {epoch_mpjpe:.2f}mm)")
    else:
        epochs_without_improvement += 1
        print(f"  No improvement: {epochs_without_improvement}/{EARLY_STOPPING_PATIENCE}")
    
    if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
        print(f"\n*** EARLY STOPPING ***")
        break
    
    print("=" * 70)

total_time = time.time() - start_time

# ==========================================
# FINAL RESULTS
# ==========================================
print("\n" + "=" * 70)
print(f"TRAINING COMPLETE - {MODEL_NAME.upper()}")
print("=" * 70)
print(f"Time: {total_time/60:.1f} min | Epochs: {len(history['train_loss'])}")
print(f"\n>>> BEST MPJPE: {best_mpjpe:.2f} mm <<<")
print(f">>> BEST PCK@20mm: {max(history['pck']):.1f}% <<<")

# Per-joint analysis
print(f"\n--- Per-Joint Error Analysis ---")
per_joint = calculate_per_joint_error(pred_denorm, target_denorm)
joint_names = ['Wrist', 'T_CMC', 'T_MCP', 'T_IP', 'T_Tip',
               'I_MCP', 'I_PIP', 'I_DIP', 'I_Tip',
               'M_MCP', 'M_PIP', 'M_DIP', 'M_Tip',
               'R_MCP', 'R_PIP', 'R_DIP', 'R_Tip',
               'P_MCP', 'P_PIP', 'P_DIP', 'P_Tip']

# Show worst 5 joints
sorted_idx = torch.argsort(per_joint, descending=True)
print("Hardest joints:")
for i in range(5):
    idx = sorted_idx[i].item()
    print(f"  {joint_names[idx]}: {per_joint[idx]:.2f}mm")

# ==========================================
# SAVE PLOTS
# ==========================================
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle(f'Training Results - {MODEL_NAME.upper()}', fontsize=14, fontweight='bold')

# Loss
axes[0,0].plot(history['train_loss'], label='Train')
axes[0,0].plot(history['val_loss'], label='Val')
axes[0,0].set_xlabel('Epoch')
axes[0,0].set_ylabel('Loss')
axes[0,0].set_title('Loss Curves')
axes[0,0].legend()
axes[0,0].grid(True, alpha=0.3)

# MPJPE
axes[0,1].plot(history['mpjpe'], color='green', linewidth=2)
axes[0,1].axhline(y=20, color='orange', linestyle='--', label='Target (20mm)')
axes[0,1].axhline(y=15, color='red', linestyle='--', label='Stretch (15mm)')
axes[0,1].set_xlabel('Epoch')
axes[0,1].set_ylabel('MPJPE (mm)')
axes[0,1].set_title(f'MPJPE (Best: {best_mpjpe:.2f}mm)')
axes[0,1].legend()
axes[0,1].grid(True, alpha=0.3)

# PCK
axes[1,0].plot(history['pck'], color='purple', linewidth=2)
axes[1,0].axhline(y=70, color='orange', linestyle='--', label='Target (70%)')
axes[1,0].set_xlabel('Epoch')
axes[1,0].set_ylabel('PCK@20mm (%)')
axes[1,0].set_title(f'PCK@20mm (Best: {max(history["pck"]):.1f}%)')
axes[1,0].legend()
axes[1,0].grid(True, alpha=0.3)

# Per-joint bar chart
colors = ['#d62728' if i in [4,8,12,16,20] else '#1f77b4' for i in range(21)]
axes[1,1].bar(range(21), per_joint.numpy(), color=colors)
axes[1,1].set_xticks(range(21))
axes[1,1].set_xticklabels(joint_names, rotation=45, ha='right', fontsize=8)
axes[1,1].set_ylabel('Error (mm)')
axes[1,1].set_title('Per-Joint Error (red=fingertips)')
axes[1,1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(f'{SAVE_PREFIX}_plots.png', dpi=150, bbox_inches='tight')
print(f"\nSaved: {SAVE_PREFIX}_plots.png")

# Save history for comparison
torch.save(history, f'{SAVE_PREFIX}_history.pth')
print(f"Saved: {SAVE_PREFIX}_history.pth")

print("\n" + "=" * 70)
print("DONE!")
print("=" * 70)