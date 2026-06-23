"""将最新评估结果追加到 EXPERIMENT_LOG.md。

用法：
    python record_experiment.py --name Exp-002 --note "重新训练后结果"
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd

from config import CFG


LOG_PATH = CFG.PROJECT_ROOT / "EXPERIMENT_LOG.md"
METRICS_PATH = CFG.TABLE_DIR / "metrics_comparison.csv"


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    """将评估 CSV 转为 Markdown 表格，避免额外依赖 tabulate。"""
    headers = list(df.columns)
    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for _, row in df.iterrows():
        values = []
        for col in headers:
            value = row[col]
            if isinstance(value, float):
                values.append(f"{value:.4f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def append_record(exp_name: str, note: str) -> None:
    if not METRICS_PATH.exists():
        raise FileNotFoundError(f"未找到评估结果文件：{METRICS_PATH}，请先运行 evaluate.py 或 main.py")

    df = pd.read_csv(METRICS_PATH)
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    content = f"""

---

## 自动追加实验记录：{exp_name}

记录时间：{now}

### 实验备注

{note}

### 当前关键参数

| 参数 | 数值 |
|---|---|
| ACDC 路径 | `{CFG.ACDC_ROOT}` |
| MyoPS 路径 | `{CFG.MYOPS_ROOT}` |
| 图像尺寸 | `{CFG.IMAGE_SIZE}` |
| Batch Size | `{CFG.BATCH_SIZE}` |
| 定位器 Epoch | `{CFG.EPOCHS_LOCATOR}` |
| 分割器 Epoch | `{CFG.EPOCHS_SEGMENTOR}` |
| 学习率 | `{CFG.LEARNING_RATE}` |
| 编码器 | `{CFG.ENCODER_NAME}` |
| 编码器权重 | `{CFG.ENCODER_WEIGHTS}` |
| 设备 | `{CFG.DEVICE}` |

### 最新测试集评估结果

{dataframe_to_markdown(df)}

### 简要分析

- 请根据本次曲线、可视化和表格补充分析。
- 建议重点观察水肿和疤痕 Dice 是否提升。
- 同时检查 Cascade 背景 Dice 和 HD95 是否恶化。

### 后续计划

- 待补充。
"""

    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(content)
    print(f"已追加实验记录到：{LOG_PATH}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="Exp-New", help="实验名称，例如 Exp-002")
    parser.add_argument("--note", default="本次实验记录。", help="实验备注")
    args = parser.parse_args()
    append_record(args.name, args.note)


if __name__ == "__main__":
    main()
