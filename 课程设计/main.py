"""训练定位器、训练基线和级联分割器、推理、评估。"""

from __future__ import annotations

from config import CFG, ensure_dirs
from evaluate import evaluate_models
from inference import run_inference
from train_locator import train_locator
from train_segmentor import train_segmentors
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def main():
    ensure_dirs()
    print("=" * 80)
    print("两阶段级联心脏核磁分割项目")
    print(f"ACDC 数据路径：{CFG.ACDC_ROOT}")
    print(f"MyoPS 数据路径：{CFG.MYOPS_ROOT}")
    print(f"运行设备：{CFG.DEVICE}")
    print("=" * 80)

    if not CFG.LOCATOR_CKPT.exists():
        train_locator()
    else:
        print(f"检测到已有定位器权重，跳过第一阶段训练：{CFG.LOCATOR_CKPT}")

    if not CFG.BASELINE_CKPT.exists() or not CFG.SEGMENTOR_CKPT.exists():
        train_segmentors()
    else:
        print("检测到已有基线和级联分割器权重，跳过第二阶段训练。")

    run_inference(max_cases=20)
    evaluate_models(save_csv=True)
    print("全部流程完成。")


if __name__ == "__main__":
    main()
