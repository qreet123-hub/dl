"""模型定义和损失函数。"""

from __future__ import annotations

import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

from config import CFG


class UNetLocator(nn.Module):
    """第一阶段：单通道 bSSFP 输入的心肌区域定位器。"""

    def __init__(self):
        super().__init__()
        self.model = smp.Unet(
            encoder_name=CFG.ENCODER_NAME,
            encoder_weights=CFG.ENCODER_WEIGHTS,
            in_channels=CFG.LOCATOR_IN_CHANNELS,
            classes=CFG.LOCATOR_CLASSES,
            activation=None,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class UNetSegmentor(nn.Module):
    """第二阶段：三模态图像加定位 mask 的四分类精细分割器。"""

    def __init__(self, in_channels: int = CFG.SEGMENTOR_IN_CHANNELS):
        super().__init__()
        self.model = smp.Unet(
            encoder_name=CFG.ENCODER_NAME,
            encoder_weights=CFG.ENCODER_WEIGHTS,
            in_channels=in_channels,
            classes=CFG.SEGMENTOR_CLASSES,
            activation=None,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class CascadeModel(nn.Module):
    """完整级联模型：定位器生成先验 mask，再与 MyoPS 三模态图像拼接后输入分割器。"""

    def __init__(self, locator: UNetLocator, segmentor: UNetSegmentor):
        super().__init__()
        self.locator = locator
        self.segmentor = segmentor

    def forward(self, myops_3ch: torch.Tensor) -> torch.Tensor:
        bssfp = myops_3ch[:, 2:3, :, :]
        locator_logits = self.locator(bssfp)
        locator_mask = torch.sigmoid(locator_logits)
        x = torch.cat([myops_3ch, locator_mask], dim=1)
        return self.segmentor(x)


class BinaryDiceLoss(nn.Module):
    """二分类 DiceLoss，用于训练定位器。"""

    def __init__(self, smooth: float = 1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        probs = probs.contiguous().view(probs.shape[0], -1)
        targets = targets.contiguous().view(targets.shape[0], -1).float()
        intersection = (probs * targets).sum(dim=1)
        dice = (2 * intersection + self.smooth) / (probs.sum(dim=1) + targets.sum(dim=1) + self.smooth)
        return 1 - dice.mean()


class WeightedMulticlassDiceLoss(nn.Module):
    """多分类加权 DiceLoss；提高水肿和疤痕类别权重。"""

    def __init__(self, weights=None, smooth: float = 1e-6):
        super().__init__()
        if weights is None:
            weights = torch.tensor(CFG.DICE_LOSS_WEIGHTS, dtype=torch.float32)
        self.register_buffer("weights", weights.float())
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(logits, dim=1)
        targets_one_hot = torch.nn.functional.one_hot(targets.long(), num_classes=logits.shape[1])
        targets_one_hot = targets_one_hot.permute(0, 3, 1, 2).float()

        dims = (0, 2, 3)
        intersection = (probs * targets_one_hot).sum(dims)
        cardinality = probs.sum(dims) + targets_one_hot.sum(dims)
        dice_loss = 1 - (2 * intersection + self.smooth) / (cardinality + self.smooth)
        weights = self.weights.to(logits.device)
        return (dice_loss * weights).sum() / weights.sum()


class DiceCrossEntropyLoss(nn.Module):
    """Dice 与加权交叉熵混合损失，用于缓解水肿、疤痕等小目标学习不足。"""

    def __init__(self):
        super().__init__()
        self.dice = WeightedMulticlassDiceLoss(torch.tensor(CFG.DICE_LOSS_WEIGHTS, dtype=torch.float32))
        ce_weights = torch.tensor(CFG.CE_LOSS_WEIGHTS, dtype=torch.float32)
        self.register_buffer("ce_weights", ce_weights)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        dice_loss = self.dice(logits, targets)
        ce_loss = torch.nn.functional.cross_entropy(logits, targets.long(), weight=self.ce_weights.to(logits.device))
        return dice_loss + CFG.CE_LOSS_FACTOR * ce_loss


def build_locator() -> UNetLocator:
    return UNetLocator()


def build_segmentor(with_locator_mask: bool = True) -> UNetSegmentor:
    in_channels = CFG.SEGMENTOR_IN_CHANNELS if with_locator_mask else CFG.BASELINE_IN_CHANNELS
    return UNetSegmentor(in_channels=in_channels)
