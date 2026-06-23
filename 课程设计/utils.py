"""通用工具函数：随机种子、医学图像读取、指标计算和可视化辅助。"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.ndimage import binary_erosion, distance_transform_edt

from config import CFG


def set_seed(seed: int = CFG.RANDOM_SEED) -> None:
    """固定随机种子，方便复现实验结果。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def is_supported_file(path: Path) -> bool:
    """判断文件是否为项目支持的图像或数组格式。"""
    name = path.name.lower()
    return any(name.endswith(ext) for ext in CFG.SUPPORTED_EXTENSIONS)


def list_supported_files(root: Path) -> List[Path]:
    """递归列出目录下所有支持格式文件。"""
    if not root.exists():
        raise FileNotFoundError(f"数据路径不存在：{root}")
    return [p for p in root.rglob("*") if p.is_file() and is_supported_file(p)]


def contains_any(text: str, keys: Sequence[str]) -> bool:
    """大小写不敏感地判断字符串是否包含任一关键词。"""
    text = text.lower()
    return any(k.lower() in text for k in keys)


def read_image(path: Path) -> np.ndarray:
    """读取 2D/3D 医学图像、普通图像或 numpy 数组。

    对 3D 体数据，本函数返回形状为 D×H×W 的数组；对 2D 图像返回 H×W。
    """
    path = Path(path)
    lower = path.name.lower()

    if lower.endswith(".npy"):
        return np.load(path)
    if lower.endswith(".npz"):
        data = np.load(path)
        first_key = list(data.keys())[0]
        return data[first_key]

    if lower.endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")):
        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise ValueError(f"无法读取图像：{path}")
        if image.ndim == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image

    try:
        import SimpleITK as sitk
    except ImportError as exc:
        raise ImportError("读取 nii/mha/nrrd 等医学图像需要安装 SimpleITK：pip install SimpleITK") from exc

    sitk.ProcessObject_SetGlobalWarningDisplay(False)
    image = sitk.ReadImage(str(path))
    return sitk.GetArrayFromImage(image)


def to_slices(array: np.ndarray) -> List[np.ndarray]:
    """把 2D/3D/4D 数据拆成 2D 切片列表。"""
    array = np.asarray(array)
    array = np.squeeze(array)
    if array.ndim == 2:
        return [array]
    if array.ndim == 3:
        return [array[i] for i in range(array.shape[0])]
    if array.ndim == 4:
        merged = array.reshape((-1, array.shape[-2], array.shape[-1]))
        return [merged[i] for i in range(merged.shape[0])]
    raise ValueError(f"暂不支持维度为 {array.ndim} 的数据")


def resize_image(image: np.ndarray, size: int = CFG.IMAGE_SIZE, is_mask: bool = False) -> np.ndarray:
    """把图像或标签 resize 到统一大小。"""
    interpolation = cv2.INTER_NEAREST if is_mask else cv2.INTER_LINEAR
    return cv2.resize(image, (size, size), interpolation=interpolation)


def z_score(image: np.ndarray) -> np.ndarray:
    """对单张图像执行 Z-score 归一化。"""
    image = image.astype(np.float32)
    valid = image[np.isfinite(image)]
    if valid.size == 0:
        return np.zeros_like(image, dtype=np.float32)
    mean = float(valid.mean())
    std = float(valid.std())
    if std < 1e-6:
        std = 1.0
    return (image - mean) / std


def remap_labels(mask: np.ndarray, label_map: Dict[int, int]) -> np.ndarray:
    """把原始标签值映射为连续类别编号。"""
    mask = np.asarray(mask)
    unique_values = set(np.unique(mask).astype(int).tolist())
    if unique_values.issubset({0, 1, 2, 3}):
        return mask.astype(np.int64)
    out = np.zeros_like(mask, dtype=np.int64)
    for old, new in label_map.items():
        out[mask == old] = new
    return out


def acdc_to_binary(mask: np.ndarray) -> np.ndarray:
    """把 ACDC 多类别标签转为定位器二值前景。"""
    return np.isin(mask.astype(np.int64), CFG.ACDC_FOREGROUND_LABELS).astype(np.float32)


def dice_per_class(pred: np.ndarray, target: np.ndarray, num_classes: int = CFG.SEGMENTOR_CLASSES) -> List[float]:
    """分别计算每个类别的 Dice 系数。"""
    scores = []
    for cls in range(num_classes):
        pred_c = pred == cls
        target_c = target == cls
        denom = pred_c.sum() + target_c.sum()
        if denom == 0:
            scores.append(1.0)
        else:
            scores.append(float(2.0 * np.logical_and(pred_c, target_c).sum() / denom))
    return scores


def binary_hd95(pred: np.ndarray, target: np.ndarray) -> float:
    """计算单个二值类别的 HD95。"""
    pred = pred.astype(bool)
    target = target.astype(bool)
    if not pred.any() and not target.any():
        return 0.0
    if not pred.any() or not target.any():
        return float("inf")

    pred_border = pred ^ binary_erosion(pred)
    target_border = target ^ binary_erosion(target)
    if not pred_border.any() or not target_border.any():
        return 0.0 if np.array_equal(pred, target) else float("inf")

    dt_target = distance_transform_edt(~target_border)
    dt_pred = distance_transform_edt(~pred_border)
    distances = np.concatenate([dt_target[pred_border], dt_pred[target_border]])
    return float(np.percentile(distances, 95))


def hd95_per_class(pred: np.ndarray, target: np.ndarray, num_classes: int = CFG.SEGMENTOR_CLASSES) -> List[float]:
    """分别计算每个类别的 HD95。"""
    return [binary_hd95(pred == cls, target == cls) for cls in range(num_classes)]


def colorize_mask(mask: np.ndarray) -> np.ndarray:
    """把 0/1/2/3 标签图转换为 RGB 彩色图。"""
    h, w = mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for cls, color in CFG.MYOPS_CLASS_COLORS.items():
        rgb[mask == cls] = color
    return rgb


def overlay_mask(image: np.ndarray, mask: np.ndarray, alpha: float = 0.35) -> np.ndarray:
    """在灰度图上叠加彩色分割 mask。"""
    image = image.astype(np.float32)
    image = image - image.min()
    if image.max() > 0:
        image = image / image.max()
    base = np.stack([image, image, image], axis=-1)
    color = colorize_mask(mask).astype(np.float32) / 255.0
    return np.clip((1 - alpha) * base + alpha * color, 0, 1)


def save_comparison_figure(bssfp: np.ndarray, gt: np.ndarray, baseline: np.ndarray, cascade: np.ndarray, save_path: Path) -> None:
    """保存四联图：原始 bSSFP、真实标签、基线预测、级联预测。"""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    titles = ["原始 bSSFP", "真实标签", "基线预测", "级联模型预测"]
    images = [bssfp, colorize_mask(gt), colorize_mask(baseline), colorize_mask(cascade)]

    plt.figure(figsize=(16, 4))
    for i, (title, img) in enumerate(zip(titles, images), start=1):
        plt.subplot(1, 4, i)
        if i == 1:
            plt.imshow(img, cmap="gray")
        else:
            plt.imshow(img)
        plt.title(title, fontproperties="SimHei")
        plt.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


def save_metrics_csv(rows: List[Dict], path: Path) -> pd.DataFrame:
    """保存评估表格。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return df
