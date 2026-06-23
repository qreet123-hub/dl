"""训练第二阶段分割器，并同时训练三通道标准 U-Net 基线。"""

from __future__ import annotations

import pandas as pd
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from config import CFG, ensure_dirs
from dataset import MyoPSDataset, split_dataset
from models import DiceCrossEntropyLoss, build_locator, build_segmentor
from utils import dice_per_class, set_seed


def load_locator(device):
    """加载已经训练好的第一阶段定位器，并冻结参数。"""
    locator = build_locator().to(device)
    checkpoint = torch.load(CFG.LOCATOR_CKPT, map_location=device)
    locator.load_state_dict(checkpoint["model"])
    locator.eval()
    for param in locator.parameters():
        param.requires_grad = False
    return locator


def _make_input(images, locator, use_locator_mask: bool):
    """根据模型类型构造三通道 Baseline 输入或四通道 Cascade 输入。"""
    if not use_locator_mask:
        return images
    with torch.no_grad():
        bssfp = images[:, 2:3]
        loc_mask = torch.sigmoid(locator(bssfp))
    return torch.cat([images, loc_mask], dim=1)


def train_one_epoch(model, loader, criterion, optimizer, device, locator=None, use_locator_mask=True):
    model.train()
    total_loss = 0.0
    for images, masks in tqdm(loader, desc="训练分割器", leave=False):
        images = images.to(device)
        masks = masks.to(device)
        inputs = _make_input(images, locator, use_locator_mask)

        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs)
        loss = criterion(logits, masks)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def validate(model, loader, criterion, device, locator=None, use_locator_mask=True):
    model.eval()
    total_loss = 0.0
    dice_sum = torch.zeros(CFG.SEGMENTOR_CLASSES, dtype=torch.float64)
    sample_count = 0
    for images, masks in tqdm(loader, desc="验证分割器", leave=False):
        images = images.to(device)
        masks = masks.to(device)
        inputs = _make_input(images, locator, use_locator_mask)
        logits = model(inputs)
        loss = criterion(logits, masks)
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        gts = masks.cpu().numpy()
        for pred, gt in zip(preds, gts):
            dice_sum += torch.tensor(dice_per_class(pred, gt), dtype=torch.float64)
            sample_count += 1
        total_loss += loss.item() * images.size(0)
    dice_mean = (dice_sum / max(sample_count, 1)).tolist()
    return total_loss / len(loader.dataset), dice_mean


def train_single_model(name: str, train_loader, val_loader, device, locator=None, use_locator_mask=True):
    model = build_segmentor(with_locator_mask=use_locator_mask).to(device)
    criterion = DiceCrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=CFG.LEARNING_RATE)
    ckpt_path = CFG.SEGMENTOR_CKPT if use_locator_mask else CFG.BASELINE_CKPT
    safe_name = "cascade" if use_locator_mask else "baseline"
    history_path = CFG.CURVE_DIR / f"{safe_name}_history.csv"

    best_score = -1.0
    history = []
    for epoch in range(1, CFG.EPOCHS_SEGMENTOR + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, locator, use_locator_mask)
        val_loss, val_dice = validate(model, val_loader, criterion, device, locator, use_locator_mask)
        lesion_score = (val_dice[2] + val_dice[3]) / 2.0
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_dice_bg": val_dice[0],
            "val_dice_myo": val_dice[1],
            "val_dice_edema": val_dice[2],
            "val_dice_scar": val_dice[3],
            "val_dice_lesion_mean": lesion_score,
        }
        history.append(row)
        pd.DataFrame(history).to_csv(history_path, index=False, encoding="utf-8-sig")

        dice_text = " | ".join([f"{cls}={score:.4f}" for cls, score in zip(CFG.MYOPS_CLASS_NAMES, val_dice)])
        print(f"[{name}] Epoch {epoch:03d}/{CFG.EPOCHS_SEGMENTOR} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | {dice_text}")
        if lesion_score > best_score:
            best_score = lesion_score
            torch.save({"model": model.state_dict(), "best_lesion_dice": best_score, "epoch": epoch}, ckpt_path)
            print(f"保存最佳{name}：{ckpt_path}，水肿/疤痕平均Dice={best_score:.4f}")
    return ckpt_path


def build_lesion_sampler(train_set):
    """提高包含水肿、疤痕切片的采样概率，缓解小目标类别不平衡。"""
    if not CFG.USE_LESION_WEIGHTED_SAMPLER:
        return None

    weights = []
    old_training = getattr(train_set, "training", None)
    if old_training is not None:
        train_set.training = False

    for idx in range(len(train_set)):
        _, mask = train_set[idx]
        unique_labels = set(mask.unique().tolist())
        weight = 1.0
        if 1 in unique_labels:
            weight = max(weight, CFG.NORMAL_SAMPLE_WEIGHT)
        if 2 in unique_labels:
            weight = max(weight, CFG.EDEMA_SAMPLE_WEIGHT)
        if 3 in unique_labels:
            weight = max(weight, CFG.SCAR_SAMPLE_WEIGHT)
        weights.append(weight)

    if old_training is not None:
        train_set.training = old_training

    return WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)


def train_segmentors():
    ensure_dirs()
    set_seed()
    device = torch.device(CFG.DEVICE)
    print(f"分割器训练设备：{device}")

    if not CFG.LOCATOR_CKPT.exists():
        raise FileNotFoundError(f"未找到定位器权重：{CFG.LOCATOR_CKPT}，请先运行 train_locator.py")

    full_dataset = MyoPSDataset(CFG.MYOPS_ROOT, training=True)
    train_set, val_set, _ = split_dataset(full_dataset, val_ratio=CFG.VAL_RATIO, test_ratio=CFG.TEST_RATIO)

    sampler = build_lesion_sampler(train_set)
    train_loader = DataLoader(
        train_set,
        batch_size=CFG.BATCH_SIZE,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=CFG.NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(val_set, batch_size=CFG.BATCH_SIZE, shuffle=False, num_workers=CFG.NUM_WORKERS, pin_memory=torch.cuda.is_available())

    locator = load_locator(device)
    baseline_ckpt = train_single_model("Baseline-3ch-UNet", train_loader, val_loader, device, locator=None, use_locator_mask=False)
    cascade_ckpt = train_single_model("Cascade-4ch-UNet", train_loader, val_loader, device, locator=locator, use_locator_mask=True)
    return baseline_ckpt, cascade_ckpt


if __name__ == "__main__":
    train_segmentors()
