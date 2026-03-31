
## THE MODEL WAS TRAINED IN GOOGLE COLAB SO THEREFORE THERE WILL BE SOME DISRUPTIONS WHILE RUNNING IT LOCALLY


# =============================================================================
# CELL 1 — DOWNLOAD BDD100K
# =============================================================================
import os
os.system('mkdir -p ~/.kaggle && cp /content/kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json')

print(" Downloading BDD100K (This will take 2-4 minutes)...")
!kaggle datasets download solesensei/solesensei_bdd100k -p /content

print(" Extracting dataset...")
!unzip -q /content/solesensei_bdd100k.zip -d /content/bdd100k
print(" BDD100K Ready!")

# =============================================================================
# CELL 2 — IMPORTS
# =============================================================================
import os
import cv2
import time
import glob
import random
import warnings
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
import segmentation_models_pytorch as smp
from tqdm import tqdm

warnings.filterwarnings("ignore")
print(" Imports loaded.")


# =============================================================================
# CELL 3 — HIGH-SPEED DATA MATCHING (19-CLASS SEGMENTATION TARGET)
# =============================================================================
import os
import glob
import random
import albumentations as A
from albumentations.pytorch import ToTensorV2

all_jpgs = glob.glob('/content/bdd100k/bdd100k_seg/bdd100k/seg/images/**/*.jpg', recursive=True)
all_pngs = glob.glob('/content/bdd100k/bdd100k_seg/bdd100k/seg/labels/**/*.png', recursive=True)

if not all_jpgs or not all_pngs:
    raise FileNotFoundError("Could not find the 'seg' images or labels. Check extraction path!")

mask_lookup = {os.path.basename(m).split('_')[0]: m for m in all_pngs}

train_pairs = []
val_pairs = []

print("🔗 Linking images to their ground truth masks...")
for img_path in all_jpgs:
    base_id = os.path.basename(img_path).replace('.jpg', '')

    if base_id in mask_lookup:
        mask_path = mask_lookup[base_id]
        
        if '/train/' in img_path.lower():
            train_pairs.append((img_path, mask_path))
        elif '/val/' in img_path.lower():
            val_pairs.append((img_path, mask_path))
        if '/test/' in img_path.lower():
            val_pairs.append((img_path, mask_path))

random.shuffle(train_pairs)
random.shuffle(val_pairs)

train_sample = train_pairs[:min(5000, len(train_pairs))]
val_sample = val_pairs[:min(1000, len(val_pairs))]

train_imgs, train_masks = zip(*train_sample) if train_sample else ([], [])
val_imgs, val_masks = zip(*val_sample) if val_sample else ([], [])

print(f"Final Training Set: {len(train_imgs)} pairs")
print(f"Final Validation Set: {len(val_imgs)} pairs")

IMG_H, IMG_W = 512, 896

train_tfm = A.Compose([
    A.Resize(IMG_H, IMG_W),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.3),
    A.HueSaturationValue(p=0.3),
    A.RandomRain(p=0.1),
    A.RandomFog(p=0.1),
    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=5, p=0.3),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

val_tfm = A.Compose([
    A.Resize(IMG_H, IMG_W),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])


# =============================================================================
# CELL 4 — DATASET CLASS & DATALOADERS (TRUE 19-CLASS LOGIC)
# =============================================================================
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import torch

class BDDDrivableDataset(Dataset):
    def __init__(self, imgs, masks, transforms=None):
        self.imgs       = imgs
        self.masks      = masks
        self.transforms = transforms

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img = cv2.cvtColor(cv2.imread(self.imgs[idx]), cv2.COLOR_BGR2RGB)


        mask_raw = cv2.imread(self.masks[idx], cv2.IMREAD_UNCHANGED)

        if mask_raw is None:
            
            binary_mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.float32)

        elif len(mask_raw.shape) == 3:
            
            binary_mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.float32)

        else:
            binary_mask = (mask_raw == 0).astype(np.float32)

        if self.transforms:
            aug         = self.transforms(image=img, mask=binary_mask)
            img         = aug['image']
            binary_mask = aug['mask']

        return img, binary_mask.unsqueeze(0).float()

train_loader = DataLoader(
    BDDDrivableDataset(train_imgs, train_masks, train_tfm),
    batch_size=8,
    shuffle=True,
    num_workers=2,
    pin_memory=True,
    drop_last=True 
)

val_loader = DataLoader(
    BDDDrivableDataset(val_imgs, val_masks, val_tfm),
    batch_size=8,
    shuffle=False,
    num_workers=2,
    pin_memory=True
)

print("DataLoaders Initialized targeting EXACTLY Class 0 (Road).")


# =============================================================================
# CELL 5 — ARCHITECTURE & HYPERPARAMETERS
# =============================================================================
import torch.nn.functional as F
print("Initializing DeepLabV3+ (MobileNetV3-Large)...")

model = smp.DeepLabV3Plus(
    encoder_name="timm-mobilenetv3_large_100",
    encoder_weights="imagenet",
    in_channels=3,
    classes=1,
    activation=None,
).cuda()

# Dice + Focal handles class imbalance perfectly
dice_loss  = smp.losses.DiceLoss(mode='binary')
focal_loss = smp.losses.FocalLoss(mode='binary', gamma=3.0, alpha=0.75)

def boundary_loss(y_pred, y_true):
    """
    Penalizes the model specifically at the edges of the road.
    Uses a Laplacian kernel to find the 'boundary' pixels.
    """
    kernel = torch.tensor([[-1,-1,-1], [-1,8,-1], [-1,-1,-1]], dtype=torch.float32).view(1,1,3,3).cuda()
    edges = F.conv2d(y_true, kernel, padding=1).abs().clamp(0, 1)

    bce_edge = F.binary_cross_entropy_with_logits(y_pred, y_true, weight=edges)
    return bce_edge

def criterion(y_pred, y_true):
    return (
        dice_loss(y_pred, y_true)
        + 0.5 * focal_loss(y_pred, y_true)
        + 0.3 * boundary_loss(y_pred, y_true)   
    )
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-3)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=15, eta_min=1e-6)

print("Model, Loss, and Optimizer ready.")


# =============================================================================
# CELL 7 — TRAINING SPRINT (FIXED: TRUE mIoU OVER BOTH CLASSES)
# =============================================================================
NUM_EPOCHS = 15
THRESHOLD  = 0.5
scaler     = GradScaler()
best_miou  = 0.0
history    = {'train_loss': [], 'val_miou': [], 'lr': []}

print("Starting High-Speed Training on BDD100K...")

for epoch in range(NUM_EPOCHS):

    # ---- TRAIN ----
    model.train()
    train_loss = 0.0
    for imgs, masks in tqdm(train_loader, desc=f"Epoch {epoch+1:02d}/{NUM_EPOCHS} [train]"):
        imgs, masks = imgs.cuda().float(), masks.cuda().float()
        optimizer.zero_grad()

        with autocast():
            preds = model(imgs)
            loss  = criterion(preds, masks)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        train_loss += loss.item()

    scheduler.step()
    avg_loss = train_loss / len(train_loader)

    model.eval()
    val_iou = 0.0
    with torch.no_grad():
        for imgs, masks in tqdm(val_loader, desc=f"Epoch {epoch+1:02d}/{NUM_EPOCHS} [val]  "):
            imgs, masks = imgs.cuda().float(), masks.cuda().float()

            with autocast():
                preds = model(imgs)

            pred_bin = (torch.sigmoid(preds) > THRESHOLD).float()

            # --- Class 1: Drivable ---
            tp1  = (pred_bin * masks).sum(dim=(1,2,3))
            fp1  = (pred_bin * (1 - masks)).sum(dim=(1,2,3))
            fn1  = ((1 - pred_bin) * masks).sum(dim=(1,2,3))
            iou1 = (tp1 + 1e-6) / (tp1 + fp1 + fn1 + 1e-6)

            # --- Class 0: Non-Drivable ---
            inv_pred = 1 - pred_bin
            inv_mask = 1 - masks
            tp0  = (inv_pred * inv_mask).sum(dim=(1,2,3))
            fp0  = (inv_pred * masks).sum(dim=(1,2,3))
            fn0  = ((1 - inv_pred) * inv_mask).sum(dim=(1,2,3))
            iou0 = (tp0 + 1e-6) / (tp0 + fp0 + fn0 + 1e-6)
            val_iou += ((iou1 + iou0) / 2).mean().item()

    val_miou = val_iou / len(val_loader)
    history['train_loss'].append(avg_loss)
    history['val_miou'].append(val_miou)
    history['lr'].append(scheduler.get_last_lr()[0])

    print(f"Epoch {epoch+1:02d} | Loss: {avg_loss:.4f} | Val mIoU: {val_miou:.4f}")

    if val_miou > best_miou:
        best_miou = val_miou
        torch.save(model.state_dict(), '/content/best_bdd_model.pth')
        print(f"NEW BEST mIoU: {best_miou:.4f} Saved!")

print(f"\nTraining complete. Final Best mIoU: {best_miou:.4f}")

# =============================================================================
# CELL 7 — EVALUATION, VISUALS & ONNX EXPORT
# =============================================================================
import onnxruntime as ort

# 1. Load Best Weights
model.load_state_dict(torch.load('/content/best_bdd_model.pth'))
model.eval()

# 2. Qualitative Results (Plotting)
def predict_and_overlay(img_path):
    img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    tensor = val_tfm(image=img)['image'].unsqueeze(0).cuda().float()
    with torch.no_grad(), autocast():
        pred = (torch.sigmoid(model(tensor)) > THRESHOLD).squeeze().cpu().numpy()

    overlay = cv2.resize(img, (IMG_W, IMG_H)).copy()
    overlay[pred > 0] = 0.5 * overlay[pred > 0] + 0.5 * np.array([0, 255, 100])
    return cv2.resize(img, (IMG_W, IMG_H)), overlay
fig, axes = plt.subplots(2, 3, figsize=(15, 7))
for i in range(3):
    orig, pred_overlay = predict_and_overlay(val_imgs[i])
    axes[0][i].imshow(orig)
    axes[0][i].set_title("Input"), axes[0][i].axis('off')
    axes[1][i].imshow(pred_overlay.astype(np.uint8))
    axes[1][i].set_title("Prediction"), axes[1][i].axis('off')
plt.suptitle(f"BDD100K Predictions (mIoU: {best_miou:.4f})", fontsize=16)
plt.show()

# 3. ONNX Export
model_float = model.float().eval()
dummy_input = torch.randn(1, 3, IMG_H, IMG_W).cuda()
torch.onnx.export(
    model_float, dummy_input, '/content/bdd_drivable_opt.onnx',
    export_params=True, opset_version=13, input_names=['image'], output_names=['mask'],
    do_constant_folding=True
)
print("\n ONNX Model Exported to /content/bdd_drivable_opt.onnx")

# 4. FPS Benchmark
ort_session = ort.InferenceSession('/content/bdd_drivable_opt.onnx', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
dummy_np = np.random.randn(1, 3, IMG_H, IMG_W).astype(np.float32)

for _ in range(10): ort_session.run(None, {'image': dummy_np})

t0 = time.time()
for _ in range(100): ort_session.run(None, {'image': dummy_np})
fps = 100 / (time.time() - t0)
print(f"ONNX Inference Speed: {fps:.1f} FPS")
