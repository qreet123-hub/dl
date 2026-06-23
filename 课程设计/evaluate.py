"""模型评估脚本：计算 Baseline 与级联模型的 Dice 和 HD95，并输出对比表格。"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import CFG, ensure_dirs
from dataset import MyoPSDataset, split_dataset
from models import build_locator, build_segmentor
from utils import dice_per_class, hd95_per_class, save_metrics_csv, set_seed


def load_models(device):
    if not CFG.LOCATOR_CKPT.exists():
        raise FileNotFoundError(f"缺少定位器权重：{CFG.LOCATOR_CKPT}")
    if not CFG.BASELINE_CKPT.exists():
        raise FileNotFoundError(f"缺少基线模型权重：{CFG.BASELINE_CKPT}")
    if not CFG.SEGMENTOR_CKPT.exists():
        raise FileNotFoundError(f"缺少级联分割器权重：{CFG.SEGMENTOR_CKPT}")

    locator = build_locator().to(device)
    baseline = build_segmentor(with_locator_mask=False).to(device)
    cascade = build_segmentor(with_locator_mask=True).to(device)

    locator.load_state_dict(torch.load(CFG.LOCATOR_CKPT, map_location=device)["model"])
    baseline.load_state_dict(torch.load(CFG.BASELINE_CKPT, map_location=device)["model"])
    cascade.load_state_dict(torch.load(CFG.SEGMENTOR_CKPT, map_location=device)["model"])

    locator.eval()
    baseline.eval()
    cascade.eval()
    return locator, baseline, cascade


@torch.no_grad()
def predict_batch(images, locator, baseline, cascade):
    base_logits = baseline(images)
    base_pred = torch.argmax(base_logits, dim=1)

    loc_mask = torch.sigmoid(locator(images[:, 2:3]))
    cascade_input = torch.cat([images, loc_mask], dim=1)
    cascade_logits = cascade(cascade_input)
    cascade_pred = torch.argmax(cascade_logits, dim=1)
    return base_pred, cascade_pred, loc_mask


def evaluate_models(save_csv: bool = True):
    ensure_dirs()
    set_seed()
    device = torch.device(CFG.DEVICE)
    print(f"评估设备：{device}")

    full_dataset = MyoPSDataset(CFG.MYOPS_ROOT, training=False, return_meta=True)
    _, _, test_set = split_dataset(full_dataset, val_ratio=CFG.VAL_RATIO, test_ratio=CFG.TEST_RATIO, return_meta=True)
    loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=CFG.NUM_WORKERS)

    locator, baseline, cascade = load_models(device)
    accum = {
        "Baseline": {"dice": [], "hd95": []},
        "Cascade": {"dice": [], "hd95": []},
    }

    for images, masks, _ in tqdm(loader, desc="评估测试集"):
        images = images.to(device)
        masks_np = masks.numpy()[0]
        base_pred, cascade_pred, _ = predict_batch(images, locator, baseline, cascade)
        base_np = base_pred.cpu().numpy()[0]
        cascade_np = cascade_pred.cpu().numpy()[0]

        accum["Baseline"]["dice"].append(dice_per_class(base_np, masks_np))
        accum["Baseline"]["hd95"].append(hd95_per_class(base_np, masks_np))
        accum["Cascade"]["dice"].append(dice_per_class(cascade_np, masks_np))
        accum["Cascade"]["hd95"].append(hd95_per_class(cascade_np, masks_np))

    rows = []
    for model_name, metrics in accum.items():
        dice_arr = np.asarray(metrics["dice"], dtype=np.float64)
        hd95_arr = np.asarray(metrics["hd95"], dtype=np.float64)
        hd95_arr[~np.isfinite(hd95_arr)] = np.nan
        for class_idx, class_name in enumerate(CFG.MYOPS_CLASS_NAMES):
            rows.append(
                {
                    "模型": model_name,
                    "类别": class_name,
                    "Dice均值": np.nanmean(dice_arr[:, class_idx]),
                    "Dice标准差": np.nanstd(dice_arr[:, class_idx]),
                    "HD95均值": np.nanmean(hd95_arr[:, class_idx]),
                    "HD95标准差": np.nanstd(hd95_arr[:, class_idx]),
                }
            )

    df = save_metrics_csv(rows, CFG.TABLE_DIR / "metrics_comparison.csv") if save_csv else rows
    print("\n评估结果：")
    print(df)
    return df


if __name__ == "__main__":
    evaluate_models()
