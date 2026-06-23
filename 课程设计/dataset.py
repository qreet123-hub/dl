"""数据集定义。

ACDCDataset 用于训练定位器：输入 bSSFP 单通道图像，输出左心室+心肌二值 mask。
MyoPSDataset 用于训练精细分割器：输入 LGE/T2/bSSFP 三通道图像，输出四分类 mask。

代码兼容 MyoPS 2020 常见目录：train25 存放 C0/DE/T2，train25_myops_gd 存放 gd 标签。
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, random_split

from config import CFG
from utils import (
    acdc_to_binary,
    contains_any,
    list_supported_files,
    read_image,
    remap_labels,
    resize_image,
    to_slices,
    z_score,
)


def _augment_pair(images: List[np.ndarray], mask: np.ndarray) -> Tuple[List[np.ndarray], np.ndarray]:
    """对多通道图像和标签做同步几何增强。"""
    h, w = mask.shape
    angle = random.uniform(-10, 10)
    scale = random.uniform(0.9, 1.1)
    center = (w / 2, h / 2)
    matrix = cv2.getRotationMatrix2D(center, angle, scale)

    aug_images = [cv2.warpAffine(img, matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101) for img in images]
    aug_mask = cv2.warpAffine(mask, matrix, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    if random.random() < 0.5:
        aug_images = [np.fliplr(img).copy() for img in aug_images]
        aug_mask = np.fliplr(aug_mask).copy()
    return aug_images, aug_mask


class DatasetView(Dataset):
    """为子集提供独立的增强和元信息返回开关。"""

    def __init__(self, base_dataset: Dataset, indices: Sequence[int], training: bool, return_meta: Optional[bool] = None):
        self.base_dataset = base_dataset
        self.indices = list(indices)
        self.training = training
        self.return_meta = return_meta

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        old_training = getattr(self.base_dataset, "training", None)
        old_return_meta = getattr(self.base_dataset, "return_meta", None)
        if old_training is not None:
            self.base_dataset.training = self.training
        if self.return_meta is not None and old_return_meta is not None:
            self.base_dataset.return_meta = self.return_meta
        item = self.base_dataset[self.indices[idx]]
        if old_training is not None:
            self.base_dataset.training = old_training
        if old_return_meta is not None:
            self.base_dataset.return_meta = old_return_meta
        return item


def split_dataset(dataset: Dataset, val_ratio: float = CFG.VAL_RATIO, test_ratio: float = 0.0, return_meta: Optional[bool] = None):
    """按比例随机划分训练、验证、测试集。"""
    n = len(dataset)
    test_n = int(n * test_ratio)
    val_n = int(n * val_ratio)
    train_n = n - val_n - test_n
    if train_n <= 0:
        raise ValueError("数据量过少或划分比例不合理，训练集为空。")
    generator = torch.Generator().manual_seed(CFG.RANDOM_SEED)
    lengths = [train_n, val_n, test_n] if test_ratio > 0 else [train_n, val_n]
    subsets = random_split(dataset, lengths, generator=generator)
    return tuple(DatasetView(dataset, subset.indices, training=(i == 0), return_meta=return_meta) for i, subset in enumerate(subsets))


class ACDCDataset(Dataset):
    """ACDC 定位器数据集。"""

    def __init__(self, root: Path = CFG.ACDC_ROOT, training: bool = True):
        self.root = Path(root)
        self.training = training
        self.samples = self._build_samples()
        if len(self.samples) == 0:
            raise RuntimeError(f"未在 {self.root} 中找到可用 ACDC 图像/标签配对，请检查路径和文件名关键词。")

    def _build_samples(self) -> List[Dict]:
        files = list_supported_files(self.root)
        image_files = [p for p in files if not contains_any(p.name, CFG.MASK_KEYS)]
        mask_files = [p for p in files if contains_any(p.name, CFG.MASK_KEYS)]
        samples: List[Dict] = []

        for image_path in image_files:
            if not contains_any(image_path.name, CFG.BSSFP_KEYS):
                if any(contains_any(p.name, CFG.BSSFP_KEYS) for p in image_files):
                    continue
            stem_tokens = set(image_path.stem.lower().replace("_", "-").split("-"))
            candidates = []
            for mask_path in mask_files:
                mask_tokens = set(mask_path.stem.lower().replace("_", "-").split("-"))
                common = len(stem_tokens & mask_tokens)
                if image_path.parent == mask_path.parent:
                    common += 2
                candidates.append((common, mask_path))
            if not candidates:
                continue
            mask_path = sorted(candidates, key=lambda x: x[0], reverse=True)[0][1]
            image_slices = to_slices(read_image(image_path))
            mask_slices = to_slices(read_image(mask_path))
            count = min(len(image_slices), len(mask_slices))
            for idx in range(count):
                if np.asarray(mask_slices[idx]).sum() == 0:
                    continue
                samples.append({"image": image_slices[idx], "mask": mask_slices[idx], "id": f"{image_path.stem}_{idx}"})
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        image = resize_image(sample["image"], is_mask=False)
        mask = resize_image(sample["mask"], is_mask=True)
        mask = acdc_to_binary(mask)

        if self.training:
            [image], mask = _augment_pair([image], mask)

        image = z_score(image)
        image_tensor = torch.from_numpy(image[None, ...].astype(np.float32))
        mask_tensor = torch.from_numpy(mask[None, ...].astype(np.float32))
        return image_tensor, mask_tensor


class MyoPSDataset(Dataset):
    """MyoPS 2020 三模态精细分割数据集。"""

    def __init__(self, root: Path = CFG.MYOPS_ROOT, training: bool = True, return_meta: bool = False):
        self.root = Path(root)
        self.training = training
        self.return_meta = return_meta
        self.samples = self._build_samples()
        if len(self.samples) == 0:
            raise RuntimeError(f"未在 {self.root} 中找到可用 MyoPS 三模态/标签配对，请检查路径和文件名关键词。")

    def _pick(self, files: Sequence[Path], keys: Sequence[str], forbid_mask: bool = True) -> Optional[Path]:
        candidates = []
        for path in files:
            name = path.name.lower()
            if forbid_mask and contains_any(name, CFG.MASK_KEYS):
                continue
            if contains_any(name, keys):
                candidates.append(path)
        return sorted(candidates)[0] if candidates else None

    def _pick_mask(self, files: Sequence[Path]) -> Optional[Path]:
        candidates = [p for p in files if contains_any(p.name, CFG.MASK_KEYS)]
        return sorted(candidates)[0] if candidates else None

    def _case_key(self, path: Path) -> str:
        """从文件名中提取病例键，例如 myops_training_101。"""
        name = path.name.lower()
        for suffix in [".nii.gz", ".nii", ".mha", ".mhd", ".nrrd", ".npy", ".npz", ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]:
            if name.endswith(suffix):
                name = name[: -len(suffix)]
                break
        for token in ["_c0", "_de", "_lge", "_t2", "_gd", "_mask", "_label", "_seg", "_gt", "_manual"]:
            name = name.replace(token, "")
        return name

    def _build_samples(self) -> List[Dict]:
        files = list_supported_files(self.root)
        case_groups: Dict[str, List[Path]] = {}
        for path in files:
            case_groups.setdefault(self._case_key(path), []).append(path)
        samples: List[Dict] = []

        for case_key, group_files in sorted(case_groups.items()):
            lge = self._pick(group_files, CFG.LGE_KEYS)
            t2 = self._pick(group_files, CFG.T2_KEYS)
            bssfp = self._pick(group_files, CFG.BSSFP_KEYS)
            mask = self._pick_mask(group_files)
            if not all([lge, t2, bssfp, mask]):
                continue

            lge_slices = to_slices(read_image(lge))
            t2_slices = to_slices(read_image(t2))
            bssfp_slices = to_slices(read_image(bssfp))
            mask_slices = to_slices(read_image(mask))
            count = min(len(lge_slices), len(t2_slices), len(bssfp_slices), len(mask_slices))
            for idx in range(count):
                mapped_mask = remap_labels(mask_slices[idx], CFG.MYOPS_LABEL_MAP)
                if mapped_mask.sum() == 0:
                    continue
                samples.append(
                    {
                        "lge": lge_slices[idx],
                        "t2": t2_slices[idx],
                        "bssfp": bssfp_slices[idx],
                        "mask": mapped_mask,
                        "id": f"{case_key}_{idx}",
                    }
                )
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        lge = resize_image(sample["lge"], is_mask=False)
        t2 = resize_image(sample["t2"], is_mask=False)
        bssfp = resize_image(sample["bssfp"], is_mask=False)
        mask = resize_image(sample["mask"], is_mask=True).astype(np.int64)

        if self.training:
            images, mask = _augment_pair([lge, t2, bssfp], mask)
            lge, t2, bssfp = images

        image = np.stack([z_score(lge), z_score(t2), z_score(bssfp)], axis=0).astype(np.float32)
        image_tensor = torch.from_numpy(image)
        mask_tensor = torch.from_numpy(mask.astype(np.int64))

        if self.return_meta:
            return image_tensor, mask_tensor, sample["id"]
        return image_tensor, mask_tensor
