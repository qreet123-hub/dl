"""训练第一阶段 U-Net-Locator。"""

from __future__ import annotations

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import CFG, ensure_dirs
from dataset import ACDCDataset, split_dataset
from models import BinaryDiceLoss, build_locator
from utils import set_seed


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    for images, masks in tqdm(loader, desc="训练定位器", leave=False):
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, masks)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_dice = 0.0
    for images, masks in tqdm(loader, desc="验证定位器", leave=False):
        images = images.to(device)
        masks = masks.to(device)
        logits = model(images)
        loss = criterion(logits, masks)
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).float()
        intersection = (preds * masks).sum(dim=(1, 2, 3))
        denom = preds.sum(dim=(1, 2, 3)) + masks.sum(dim=(1, 2, 3))
        dice = ((2 * intersection + 1e-6) / (denom + 1e-6)).mean()
        total_loss += loss.item() * images.size(0)
        total_dice += dice.item() * images.size(0)
    return total_loss / len(loader.dataset), total_dice / len(loader.dataset)


def train_locator():
    ensure_dirs()
    set_seed()
    device = torch.device(CFG.DEVICE)
    print(f"定位器训练设备：{device}")

    full_dataset = ACDCDataset(CFG.ACDC_ROOT, training=True)
    train_set, val_set = split_dataset(full_dataset, val_ratio=CFG.VAL_RATIO)

    train_loader = DataLoader(train_set, batch_size=CFG.BATCH_SIZE, shuffle=True, num_workers=CFG.NUM_WORKERS, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_set, batch_size=CFG.BATCH_SIZE, shuffle=False, num_workers=CFG.NUM_WORKERS, pin_memory=torch.cuda.is_available())

    model = build_locator().to(device)
    criterion = BinaryDiceLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=CFG.LEARNING_RATE)

    best_dice = -1.0
    history = []
    history_path = CFG.CURVE_DIR / "locator_history.csv"

    for epoch in range(1, CFG.EPOCHS_LOCATOR + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_dice = validate(model, val_loader, criterion, device)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "val_dice": val_dice})
        pd.DataFrame(history).to_csv(history_path, index=False, encoding="utf-8-sig")

        print(f"[Locator] Epoch {epoch:03d}/{CFG.EPOCHS_LOCATOR} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | val_dice={val_dice:.4f}")
        if val_dice > best_dice:
            best_dice = val_dice
            torch.save({"model": model.state_dict(), "best_dice": best_dice, "epoch": epoch}, CFG.LOCATOR_CKPT)
            print(f"保存最佳定位器：{CFG.LOCATOR_CKPT}，Dice={best_dice:.4f}")

    return CFG.LOCATOR_CKPT


if __name__ == "__main__":
    train_locator()
