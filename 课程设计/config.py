"""项目全局配置文件。

本项目实现两阶段医学图像分割流程：
1. 使用 ACDC bSSFP 单帧图像训练心肌区域定位器；
2. 使用 MyoPS 三模态图像和定位 mask 训练精细分割器。

如果你的数据文件命名与默认自动匹配规则不同，通常只需要修改本文件中的关键词配置。
"""

from pathlib import Path
import torch


class CFG:
    """集中管理路径、训练超参数、类别定义和文件匹配关键词。"""

    PROJECT_ROOT = Path(__file__).resolve().parent

    ACDC_ROOT = Path(r"C:\Users\C\Desktop\data\ACDC")
    MYOPS_ROOT = Path(r"C:\Users\C\Desktop\data\MyoPS 2020 Dataset")

    OUTPUT_DIR = PROJECT_ROOT / "outputs"
    CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
    RESULT_DIR = PROJECT_ROOT / "results"
    FIGURE_DIR = RESULT_DIR / "figures"
    CURVE_DIR = RESULT_DIR / "curves"
    TABLE_DIR = RESULT_DIR / "tables"
    PRED_DIR = RESULT_DIR / "predictions"

    IMAGE_SIZE = 256
    BATCH_SIZE = 4
    NUM_WORKERS = 0
    EPOCHS_LOCATOR = 30
    EPOCHS_SEGMENTOR = 80
    LEARNING_RATE = 1e-4
    VAL_RATIO = 0.2
    TEST_RATIO = 0.2
    RANDOM_SEED = 42

    DICE_LOSS_WEIGHTS = [0.1, 1.0, 4.0, 4.0]
    CE_LOSS_WEIGHTS = [0.1, 1.0, 5.0, 5.0]
    CE_LOSS_FACTOR = 0.5
    USE_LESION_WEIGHTED_SAMPLER = True
    EDEMA_SAMPLE_WEIGHT = 6.0
    SCAR_SAMPLE_WEIGHT = 4.0
    NORMAL_SAMPLE_WEIGHT = 1.5

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    ENCODER_NAME = "resnet34"
    ENCODER_WEIGHTS = None

    LOCATOR_IN_CHANNELS = 1
    LOCATOR_CLASSES = 1
    SEGMENTOR_IN_CHANNELS = 4
    BASELINE_IN_CHANNELS = 3
    SEGMENTOR_CLASSES = 4

    LOCATOR_CKPT = CHECKPOINT_DIR / "best_locator.pth"
    SEGMENTOR_CKPT = CHECKPOINT_DIR / "best_cascade_segmentor.pth"
    BASELINE_CKPT = CHECKPOINT_DIR / "best_baseline_unet.pth"

    MYOPS_CLASS_NAMES = ["背景", "正常心肌", "水肿", "疤痕"]
    MYOPS_CLASS_COLORS = {
        0: (0, 0, 0),
        1: (0, 80, 255),
        2: (255, 230, 0),
        3: (255, 0, 0),
    }

    MYOPS_LABEL_MAP = {
        0: 0,
        200: 1,
        500: 1,
        600: 1,
        1220: 2,
        2221: 3,
    }

    ACDC_FOREGROUND_LABELS = [2, 3]

    LGE_KEYS = ["lge", "de", "ce"]
    T2_KEYS = ["t2"]
    BSSFP_KEYS = ["bssfp", "cine", "ssfp", "c0"]
    MASK_KEYS = ["mask", "label", "seg", "gt", "manual", "gd"]

    SUPPORTED_EXTENSIONS = [".nii", ".nii.gz", ".mha", ".mhd", ".nrrd", ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".npy", ".npz"]


def ensure_dirs():
    """创建项目运行过程中需要的输出目录。"""
    for path in [CFG.OUTPUT_DIR, CFG.CHECKPOINT_DIR, CFG.RESULT_DIR, CFG.FIGURE_DIR, CFG.CURVE_DIR, CFG.TABLE_DIR, CFG.PRED_DIR]:
        path.mkdir(parents=True, exist_ok=True)
