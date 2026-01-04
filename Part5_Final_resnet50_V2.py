# ==========================================
# Hand Pose Estimation - Test Set Evaluation
# ELE 588: Applied Machine Learning
# Author: Shayan Khodabakhsh
#
# This script:
# 1. Loads the best pre-trained ResNet50 model from cross-validation
# 2. Evaluates on the held-out FreiHAND test set
# 3. Generates comprehensive visualizations for the report
# ==========================================
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast
from torchvision import transforms, models
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import json
import os

# ==========================================
# CONFIGURATION
# ==========================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# Paths
TRAIN_PATH = "Data/FreiHAND_pub_v2"
EVAL_PATH = "Data/FreiHAND_pub_v2_eval"
MODEL_PATH = "best_model_resnet50.pth"  # From cross-validation

BATCH_SIZE = 32
USE_MIXED_PRECISION = True

# ==========================================
# TRANSFORMS
# ==========================================
train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ==========================================
# DATASET
# ==========================================
class FreiHANDDataset(Dataset):
    def __init__(self, root_dir, transform=None, is_eval=False):
        self.root_dir = root_dir
        self.transform = transform
        self.is_eval = is_eval
        
        if is_eval:
            # Eval set: individual JSON files
            self.img_folder = os.path.join(root_dir, "evaluation", "rgb")
            self.anno_folder = os.path.join(root_dir, "evaluation", "anno")
            self.num_samples = len([f for f in os.listdir(self.img_folder) if f.endswith('.jpg')])
            self.joints_list = None
        else:
            # Training set: single JSON file
            json_path = os.path.join(root_dir, "training_xyz.json")
            self.img_folder = os.path.join(root_dir, "training", "rgb")
            with open(json_path, 'r') as f:
                self.joints_list = json.load(f)
            self.num_samples = len(self.joints_list)
        
        self._compute_normalization_stats()
    
    def _compute_normalization_stats(self):
        all_joints = []
        sample_size = min(10000, self.num_samples)
        
        for i in range(sample_size):
            if self.is_eval:
                anno_path = os.path.join(self.anno_folder, f"{i:08d}.json")
                with open(anno_path, 'r') as f:
                    anno = json.load(f)
                joints = np.array(anno['xyz'])
            else:
                joints = np.array(self.joints_list[i])
            
            root = joints[0:1, :]
            joints_centered = joints - root
            all_joints.append(joints_centered)
        
        all_joints = np.concatenate(all_joints, axis=0)
        self.coord_mean = torch.tensor(all_joints.mean(axis=0), dtype=torch.float32)
        self.coord_std = torch.tensor(all_joints.std(axis=0) + 1e-8, dtype=torch.float32)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        img_name = f"{idx:08d}.jpg"
        img_path = os.path.join(self.img_folder, img_name)
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        # Load joints
        if self.is_eval:
            anno_path = os.path.join(self.anno_folder, f"{idx:08d}.json")
            with open(anno_path, 'r') as f:
                anno = json.load(f)
            joints = torch.tensor(anno['xyz'], dtype=torch.float32)
            K = torch.tensor(anno['K'], dtype=torch.float32)
        else:
            joints = torch.tensor(self.joints_list[idx], dtype=torch.float32)
            K = torch.zeros(3, 3)
        
        root = joints[0:1, :]
        joints_centered = joints - root
        joints_normalized = (joints_centered - self.coord_mean) / self.coord_std
        
        return image, joints_normalized.flatten(), root.flatten(), idx, K

    def denormalize(self, normalized_joints, root):
        joints = normalized_joints.view(-1, 21, 3)
        joints = joints * self.coord_std.to(joints.device) + self.coord_mean.to(joints.device)
        joints = joints + root.view(-1, 1, 3)
        return joints
    
    def get_raw_image(self, idx):
        """Get original image for visualization"""
        img_name = f"{idx:08d}.jpg"
        img_path = os.path.join(self.img_folder, img_name)
        return Image.open(img_path).convert('RGB')


# ==========================================
# MODEL
# ==========================================
class ResNet50Model(nn.Module):
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
    """Mean Per-Joint Position Error in mm"""
    pred = pred.view(-1, 21, 3)
    target = target.view(-1, 21, 3)
    distances = torch.sqrt(((pred - target) ** 2).sum(dim=-1))
    return distances.mean().item() * 1000


def calculate_pck(pred, target, threshold_mm=20):
    """Percentage of Correct Keypoints"""
    pred = pred.view(-1, 21, 3)
    target = target.view(-1, 21, 3)
    distances = torch.sqrt(((pred - target) ** 2).sum(dim=-1))
    threshold_m = threshold_mm / 1000.0
    return (distances < threshold_m).float().mean().item() * 100


def calculate_per_joint_error(pred, target):
    """Per-joint error in mm"""
    pred = pred.view(-1, 21, 3)
    target = target.view(-1, 21, 3)
    distances = torch.sqrt(((pred - target) ** 2).sum(dim=-1))
    return distances.mean(dim=0).numpy() * 1000


def calculate_auc(pred, target, thresholds=np.arange(0, 51, 1)):
    """Area Under PCK Curve"""
    pred = pred.view(-1, 21, 3)
    target = target.view(-1, 21, 3)
    distances = torch.sqrt(((pred - target) ** 2).sum(dim=-1)).numpy() * 1000
    
    pck_values = []
    for thresh in thresholds:
        pck = (distances < thresh).mean() * 100
        pck_values.append(pck)
    
    auc = np.trapz(pck_values, thresholds) / (thresholds[-1] - thresholds[0])
    return auc, thresholds, pck_values


# ==========================================
# TESTING
# ==========================================
def test_model(model, test_dataset):
    """Run inference on test set and collect all predictions"""
    print("\n" + "="*60)
    print("TESTING ON EVALUATION SET")
    print("="*60)
    
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, 
                             shuffle=False, num_workers=4, pin_memory=True)
    
    model.eval()
    all_preds = []
    all_targets = []
    all_roots = []
    all_indices = []
    all_K = []
    
    with torch.no_grad():
        for images, targets, roots, indices, K in test_loader:
            images = images.to(device)
            
            if USE_MIXED_PRECISION:
                with autocast():
                    outputs = model(images)
            else:
                outputs = model(images)
            
            all_preds.append(outputs.cpu())
            all_targets.append(targets)
            all_roots.append(roots)
            all_indices.extend(indices.tolist())
            all_K.append(K)
    
    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    all_roots = torch.cat(all_roots)
    all_K = torch.cat(all_K)
    
    # Denormalize
    pred_denorm = test_dataset.denormalize(all_preds, all_roots)
    target_denorm = test_dataset.denormalize(all_targets, all_roots)
    
    return pred_denorm, target_denorm, all_indices, all_K


# ==========================================
# HAND SKELETON CONNECTIONS
# ==========================================
JOINT_NAMES = [
    'Wrist',
    'Thumb_CMC', 'Thumb_MCP', 'Thumb_IP', 'Thumb_Tip',
    'Index_MCP', 'Index_PIP', 'Index_DIP', 'Index_Tip',
    'Middle_MCP', 'Middle_PIP', 'Middle_DIP', 'Middle_Tip',
    'Ring_MCP', 'Ring_PIP', 'Ring_DIP', 'Ring_Tip',
    'Pinky_MCP', 'Pinky_PIP', 'Pinky_DIP', 'Pinky_Tip'
]

SKELETON_CONNECTIONS = [
    # Thumb
    (0, 1), (1, 2), (2, 3), (3, 4),
    # Index
    (0, 5), (5, 6), (6, 7), (7, 8),
    # Middle
    (0, 9), (9, 10), (10, 11), (11, 12),
    # Ring
    (0, 13), (13, 14), (14, 15), (15, 16),
    # Pinky
    (0, 17), (17, 18), (18, 19), (19, 20),
    # Palm connections
    (5, 9), (9, 13), (13, 17)
]

FINGER_COLORS = {
    'thumb': '#FF6B6B',    # Red
    'index': '#4ECDC4',    # Teal
    'middle': '#45B7D1',   # Blue
    'ring': '#96CEB4',     # Green
    'pinky': '#FFEAA7',    # Yellow
    'palm': '#DDA0DD'      # Plum
}


# ==========================================
# VISUALIZATION FUNCTIONS
# ==========================================
def plot_training_curve(train_losses, save_path='test_training_curve.png'):
    """Plot training loss curve (placeholder if no training)"""
    if len(train_losses) == 0:
        print("Skipped: No training was performed")
        return
    
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, 'b-', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss (MSE)', fontsize=12)
    plt.title('Training Loss Curve - ResNet50', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_per_joint_error(per_joint_errors, save_path='test_per_joint_error.png'):
    """Plot per-joint error bar chart"""
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Color by finger
    colors = []
    for i in range(21):
        if i == 0:
            colors.append('#808080')  # Wrist - gray
        elif i <= 4:
            colors.append(FINGER_COLORS['thumb'])
        elif i <= 8:
            colors.append(FINGER_COLORS['index'])
        elif i <= 12:
            colors.append(FINGER_COLORS['middle'])
        elif i <= 16:
            colors.append(FINGER_COLORS['ring'])
        else:
            colors.append(FINGER_COLORS['pinky'])
    
    bars = ax.bar(range(21), per_joint_errors, color=colors, edgecolor='black', linewidth=0.5)
    
    ax.set_xlabel('Joint', fontsize=12)
    ax.set_ylabel('MPJPE (mm)', fontsize=12)
    ax.set_title('Per-Joint Position Error on Test Set', fontsize=14, fontweight='bold')
    ax.set_xticks(range(21))
    ax.set_xticklabels(JOINT_NAMES, rotation=45, ha='right', fontsize=8)
    ax.axhline(y=20, color='red', linestyle='--', linewidth=2, label='Target (20mm)')
    ax.axhline(y=per_joint_errors.mean(), color='blue', linestyle='--', linewidth=2, 
               label=f'Mean ({per_joint_errors.mean():.1f}mm)')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, val in zip(bars, per_joint_errors):
        ax.annotate(f'{val:.1f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                   xytext=(0, 3), textcoords='offset points', ha='center', va='bottom', fontsize=7)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_pck_curve(thresholds, pck_values, auc, save_path='test_pck_curve.png'):
    """Plot PCK curve at different thresholds"""
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, pck_values, 'b-', linewidth=2)
    plt.fill_between(thresholds, pck_values, alpha=0.3)
    
    plt.axvline(x=20, color='red', linestyle='--', linewidth=2, label='Threshold = 20mm')
    plt.axhline(y=pck_values[20], color='green', linestyle=':', linewidth=2, 
                label=f'PCK@20mm = {pck_values[20]:.1f}%')
    
    plt.xlabel('Threshold (mm)', fontsize=12)
    plt.ylabel('PCK (%)', fontsize=12)
    plt.title(f'Percentage of Correct Keypoints (AUC = {auc:.1f})', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 50)
    plt.ylim(0, 100)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_error_distribution(pred, target, save_path='test_error_distribution.png'):
    """Plot histogram of joint errors"""
    pred = pred.view(-1, 21, 3)
    target = target.view(-1, 21, 3)
    errors = torch.sqrt(((pred - target) ** 2).sum(dim=-1)).numpy().flatten() * 1000
    
    plt.figure(figsize=(10, 6))
    plt.hist(errors, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
    plt.axvline(x=errors.mean(), color='red', linestyle='--', linewidth=2, 
                label=f'Mean = {errors.mean():.1f}mm')
    plt.axvline(x=np.median(errors), color='green', linestyle='--', linewidth=2,
                label=f'Median = {np.median(errors):.1f}mm')
    plt.axvline(x=20, color='orange', linestyle=':', linewidth=2, label='Target = 20mm')
    
    plt.xlabel('Error (mm)', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.title('Distribution of Joint Position Errors', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_finger_comparison(per_joint_errors, save_path='test_finger_comparison.png'):
    """Compare average error by finger"""
    finger_errors = {
        'Wrist': per_joint_errors[0],
        'Thumb': per_joint_errors[1:5].mean(),
        'Index': per_joint_errors[5:9].mean(),
        'Middle': per_joint_errors[9:13].mean(),
        'Ring': per_joint_errors[13:17].mean(),
        'Pinky': per_joint_errors[17:21].mean(),
    }
    
    colors = ['#808080', FINGER_COLORS['thumb'], FINGER_COLORS['index'], 
              FINGER_COLORS['middle'], FINGER_COLORS['ring'], FINGER_COLORS['pinky']]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(finger_errors.keys(), finger_errors.values(), color=colors, 
                  edgecolor='black', linewidth=1)
    
    ax.axhline(y=20, color='red', linestyle='--', linewidth=2, label='Target (20mm)')
    ax.set_ylabel('Average MPJPE (mm)', fontsize=12)
    ax.set_title('Average Error by Finger', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    for bar, val in zip(bars, finger_errors.values()):
        ax.annotate(f'{val:.1f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                   xytext=(0, 3), textcoords='offset points', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def visualize_predictions(test_dataset, pred_denorm, target_denorm, indices, all_K,
                         num_samples=6, save_path='test_prediction_samples.png'):
    """Visualize predicted vs ground truth skeletons on sample images"""
    
    # Select random samples
    np.random.seed(42)
    sample_indices = np.random.choice(len(indices), min(num_samples, len(indices)), replace=False)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for ax_idx, sample_idx in enumerate(sample_indices):
        img_idx = indices[sample_idx]
        img = test_dataset.get_raw_image(img_idx)
        img = img.resize((224, 224))
        
        pred = pred_denorm[sample_idx].numpy()
        gt = target_denorm[sample_idx].numpy()
        K = all_K[sample_idx].numpy()
        
        # Calculate error for this sample
        error = np.sqrt(((pred - gt) ** 2).sum(axis=-1)).mean() * 1000
        
        # Project 3D to 2D using camera intrinsics
        def project_to_2d(joints_3d, K):
            fx, fy = K[0, 0], K[1, 1]
            cx, cy = K[0, 2], K[1, 2]
            
            x, y, z = joints_3d[:, 0], joints_3d[:, 1], joints_3d[:, 2]
            
            u = fx * x / z + cx
            v = fy * y / z + cy
            
            return np.stack([u, v], axis=1)
        
        pred_2d = project_to_2d(pred, K)
        gt_2d = project_to_2d(gt, K)
        
        # Draw on image
        ax = axes[ax_idx]
        ax.imshow(img)
        
        # Draw GT skeleton (green)
        for i, j in SKELETON_CONNECTIONS:
            ax.plot([gt_2d[i, 0], gt_2d[j, 0]], [gt_2d[i, 1], gt_2d[j, 1]], 
                   'g-', linewidth=2, alpha=0.7)
        ax.scatter(gt_2d[:, 0], gt_2d[:, 1], c='green', s=30, zorder=5, label='Ground Truth')
        
        # Draw Pred skeleton (red)
        for i, j in SKELETON_CONNECTIONS:
            ax.plot([pred_2d[i, 0], pred_2d[j, 0]], [pred_2d[i, 1], pred_2d[j, 1]], 
                   'r-', linewidth=2, alpha=0.7)
        ax.scatter(pred_2d[:, 0], pred_2d[:, 1], c='red', s=30, zorder=5, label='Prediction')
        
        ax.set_title(f'Sample {img_idx} | Error: {error:.1f}mm', fontsize=10)
        ax.axis('off')
        if ax_idx == 0:
            ax.legend(loc='upper right', fontsize=8)
    
    plt.suptitle('Predicted (Red) vs Ground Truth (Green) Hand Poses', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


# ==========================================
# MAIN
# ==========================================
if __name__ == "__main__":
    print("="*60)
    print("HAND POSE ESTIMATION - TEST SET EVALUATION")
    print("="*60)
    
    # Load datasets
    print("\nLoading datasets...")
    train_dataset = FreiHANDDataset(TRAIN_PATH, transform=train_transform, is_eval=False)
    test_dataset = FreiHANDDataset(EVAL_PATH, transform=test_transform, is_eval=True)
    
    # Use training set normalization stats for test set
    test_dataset.coord_mean = train_dataset.coord_mean
    test_dataset.coord_std = train_dataset.coord_std
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    # Load pre-trained model
    print("\n" + "="*60)
    print("LOADING PRE-TRAINED MODEL")
    print("="*60)
    
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: Model file '{MODEL_PATH}' not found!")
        print("Please run 'part5_cross_validation.py' first to generate the model.")
        exit(1)
    
    model = ResNet50Model().to(device)
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    
    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded model (CV MPJPE: {checkpoint.get('mpjpe', 'N/A'):.2f}mm)")
    else:
        model.load_state_dict(checkpoint)
        print("Loaded model checkpoint")
    
    # Test model
    pred_denorm, target_denorm, indices, all_K = test_model(model, test_dataset)
    
    # Calculate metrics
    print("\n" + "="*60)
    print("TEST SET METRICS")
    print("="*60)
    
    mpjpe = calculate_mpjpe(pred_denorm, target_denorm)
    pck = calculate_pck(pred_denorm, target_denorm, threshold_mm=20)
    per_joint = calculate_per_joint_error(pred_denorm, target_denorm)
    auc, thresholds, pck_curve = calculate_auc(pred_denorm, target_denorm)
    
    # All errors for histogram
    all_errors = torch.sqrt(((pred_denorm.view(-1, 21, 3) - target_denorm.view(-1, 21, 3)) ** 2).sum(dim=-1)).numpy().flatten() * 1000
    
    print(f"\nMPJPE: {mpjpe:.2f} mm")
    print(f"PCK@20mm: {pck:.1f}%")
    print(f"AUC (0-50mm): {auc:.1f}")
    
    # Success criteria
    print("\n--- Success Criteria ---")
    if mpjpe <= 20:
        print(f"✓ Primary Goal: {mpjpe:.2f}mm ≤ 20mm")
    else:
        print(f"✗ Primary Goal: {mpjpe:.2f}mm > 20mm")
    
    if mpjpe <= 15:
        print(f"★ Stretch Goal: {mpjpe:.2f}mm ≤ 15mm")
    
    if pck >= 70:
        print(f"✓ PCK Goal: {pck:.1f}% ≥ 70%")
    else:
        print(f"✗ PCK Goal: {pck:.1f}% < 70%")
    
    # Store results
    results = {
        'mpjpe': mpjpe,
        'pck': pck,
        'per_joint': per_joint,
        'auc': auc,
        'thresholds': thresholds,
        'pck_curve': pck_curve,
        'all_errors': all_errors,
        'n_samples': len(test_dataset)
    }
    
    # Generate all visualizations
    print("\n" + "="*60)
    print("GENERATING VISUALIZATIONS")
    print("="*60)
    
    plot_per_joint_error(per_joint, 'test_per_joint_error.png')
    plot_pck_curve(thresholds, pck_curve, auc, 'test_pck_curve.png')
    plot_error_distribution(pred_denorm, target_denorm, 'test_error_distribution.png')
    plot_finger_comparison(per_joint, 'test_finger_comparison.png')
    visualize_predictions(test_dataset, pred_denorm, target_denorm, indices, all_K,
                         num_samples=6, save_path='test_prediction_samples.png')
    
    # Save results
    torch.save({
        'results': results,
        'pred_denorm': pred_denorm,
        'target_denorm': target_denorm,
        'indices': indices,
        'all_K': all_K
    }, 'test_results.pth')
    print("\nSaved: test_results.pth")
    
    # Print summary for report
    print("\n" + "="*60)
    print("FOR YOUR REPORT:")
    print("="*60)
    print(f"""
Test Set Evaluation (n = {len(test_dataset)} samples):
- MPJPE: {mpjpe:.2f} mm
- PCK@20mm: {pck:.1f}%
- AUC (0-50mm): {auc:.1f}

Model: ResNet50 with ImageNet pretraining
""")
    
    print("\n" + "="*60)
    print("DONE! All visualizations saved.")
    print("="*60)