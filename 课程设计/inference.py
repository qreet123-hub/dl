"""推理与可视化脚本。

输出内容包括：
1. Baseline 与级联模型的预测 npy 文件；
2. 四联图：原始 bSSFP、真实标签、基线预测、级联预测；
3. 定位 mask 叠加图，便于观察解剖先验位置。
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import CFG, ensure_dirs
from dataset import MyoPSDataset, split_dataset
from evaluate import load_models, predict_batch
from utils import overlay_mask, save_comparison_figure, set_seed


@torch.no_grad()
def run_inference(max_cases: int = 20):
    ensure_dirs()
    set_seed()
    device = torch.device(CFG.DEVICE)
    print(f"推理设备：{device}")

    full_dataset = MyoPSDataset(CFG.MYOPS_ROOT, training=False, return_meta=True)
    _, _, test_set = split_dataset(full_dataset, val_ratio=CFG.VAL_RATIO, test_ratio=CFG.TEST_RATIO, return_meta=True)
    loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=CFG.NUM_WORKERS)

    locator, baseline, cascade = load_models(device)

    for index, (images, masks, case_ids) in enumerate(tqdm(loader, desc="推理并保存可视化")):
        if index >= max_cases:
            break
        images = images.to(device)
        base_pred, cascade_pred, loc_mask = predict_batch(images, locator, baseline, cascade)

        image_np = images.cpu().numpy()[0]
        bssfp = image_np[2]
        gt = masks.numpy()[0]
        base_np = base_pred.cpu().numpy()[0]
        cascade_np = cascade_pred.cpu().numpy()[0]
        loc_np = (loc_mask.cpu().numpy()[0, 0] > 0.5).astype(np.uint8)
        case_id = str(case_ids[0])

        np.save(CFG.PRED_DIR / f"{case_id}_baseline.npy", base_np)
        np.save(CFG.PRED_DIR / f"{case_id}_cascade.npy", cascade_np)
        np.save(CFG.PRED_DIR / f"{case_id}_locator_mask.npy", loc_np)

        save_comparison_figure(bssfp, gt, base_np, cascade_np, CFG.FIGURE_DIR / f"{case_id}_comparison.png")

        overlay = overlay_mask(bssfp, loc_np.astype(np.int64))
        plt.figure(figsize=(5, 5))
        plt.imshow(overlay)
        plt.title("定位 mask 叠加图", fontproperties="SimHei")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(CFG.FIGURE_DIR / f"{case_id}_locator_overlay.png", dpi=200, bbox_inches="tight")
        plt.close()

    print(f"推理结果已保存到：{CFG.PRED_DIR}")
    print(f"可视化图片已保存到：{CFG.FIGURE_DIR}")


if __name__ == "__main__":
    run_inference()
