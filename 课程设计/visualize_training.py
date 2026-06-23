"""训练曲线与评估指标可视化脚本。

本脚本使用 OpenCV 直接绘图，不依赖 matplotlib，适合在 Windows/Anaconda 环境中稳定生成 PNG 图片。

可生成：
1. 定位器训练曲线：locator_training_curves.png
2. Baseline 训练曲线：baseline_training_curves.png
3. Cascade 训练曲线：cascade_training_curves.png
4. Dice 对比柱状图：dice_comparison_bar.png
5. HD95 对比柱状图：hd95_comparison_bar.png
6. 汇总图：summary_dashboard.png

说明：
- 训练曲线依赖 results/curves/*.csv。
- 如果旧模型是在添加曲线记录之前训练的，则没有 history CSV；重新训练后会自动生成。
- 最终 Dice/HD95 柱状图依赖 results/tables/metrics_comparison.csv。
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import cv2
import numpy as np
import pandas as pd

from config import CFG, ensure_dirs


Color = Tuple[int, int, int]
WHITE: Color = (255, 255, 255)
BLACK: Color = (20, 20, 20)
GRAY: Color = (210, 210, 210)
BLUE: Color = (220, 120, 40)
ORANGE: Color = (40, 150, 245)
GREEN: Color = (70, 170, 70)
RED: Color = (60, 60, 230)
YELLOW: Color = (0, 210, 240)
PURPLE: Color = (180, 80, 180)


def make_canvas(width: int, height: int) -> np.ndarray:
    """创建白色画布。"""
    return np.full((height, width, 3), 255, dtype=np.uint8)


def put_text(img: np.ndarray, text: str, org: Tuple[int, int], scale: float = 0.55, color: Color = BLACK, thickness: int = 1) -> None:
    """绘制英文文本。"""
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def normalize(values: Sequence[float], ymin: float | None = None, ymax: float | None = None) -> Tuple[np.ndarray, float, float]:
    """归一化数值到 0-1。"""
    arr = np.asarray(values, dtype=np.float32)
    if ymin is None:
        ymin = float(np.nanmin(arr)) if arr.size else 0.0
    if ymax is None:
        ymax = float(np.nanmax(arr)) if arr.size else 1.0
    if abs(ymax - ymin) < 1e-8:
        ymax = ymin + 1.0
    return (arr - ymin) / (ymax - ymin), ymin, ymax


def draw_line_chart(
    title: str,
    x: Sequence[float],
    series: List[Tuple[str, Sequence[float], Color]],
    width: int = 900,
    height: int = 520,
    y_min: float | None = None,
    y_max: float | None = None,
) -> np.ndarray:
    """绘制多曲线折线图。"""
    img = make_canvas(width, height)
    left, right, top, bottom = 80, 30, 60, 80
    plot_w = width - left - right
    plot_h = height - top - bottom

    put_text(img, title, (left, 35), scale=0.8, thickness=2)
    cv2.rectangle(img, (left, top), (left + plot_w, top + plot_h), BLACK, 1)

    all_values = []
    for _, values, _ in series:
        all_values.extend(list(values))
    if y_min is None:
        y_min = float(np.nanmin(all_values)) if all_values else 0.0
    if y_max is None:
        y_max = float(np.nanmax(all_values)) if all_values else 1.0
    if abs(y_max - y_min) < 1e-8:
        y_max = y_min + 1.0

    x_arr = np.asarray(x, dtype=np.float32)
    x_norm, x_low, x_high = normalize(x_arr)

    for i in range(6):
        y = top + int(plot_h * i / 5)
        cv2.line(img, (left, y), (left + plot_w, y), GRAY, 1)
        value = y_max - (y_max - y_min) * i / 5
        put_text(img, f"{value:.3f}", (8, y + 5), scale=0.45)

    for i in range(6):
        x_pos = left + int(plot_w * i / 5)
        cv2.line(img, (x_pos, top), (x_pos, top + plot_h), GRAY, 1)
        value = x_low + (x_high - x_low) * i / 5
        put_text(img, f"{value:.0f}", (x_pos - 12, top + plot_h + 28), scale=0.45)

    legend_x = left + 10
    legend_y = top + 25
    for idx, (label, values, color) in enumerate(series):
        vals = np.asarray(values, dtype=np.float32)
        y_norm, _, _ = normalize(vals, y_min, y_max)
        points = []
        for xn, yn in zip(x_norm, y_norm):
            px = left + int(xn * plot_w)
            py = top + plot_h - int(yn * plot_h)
            points.append((px, py))
        for p1, p2 in zip(points[:-1], points[1:]):
            cv2.line(img, p1, p2, color, 2)
        for point in points:
            cv2.circle(img, point, 2, color, -1)

        y0 = legend_y + idx * 24
        cv2.line(img, (legend_x, y0), (legend_x + 25, y0), color, 3)
        put_text(img, label, (legend_x + 35, y0 + 5), scale=0.5)

    put_text(img, "Epoch", (left + plot_w // 2 - 25, height - 25), scale=0.55)
    put_text(img, "Value", (10, top - 15), scale=0.55)
    return img


def draw_bar_chart(
    title: str,
    categories: Sequence[str],
    model_names: Sequence[str],
    values: np.ndarray,
    width: int = 900,
    height: int = 560,
    y_max: float | None = None,
) -> np.ndarray:
    """绘制分组柱状图。"""
    img = make_canvas(width, height)
    left, right, top, bottom = 80, 40, 60, 100
    plot_w = width - left - right
    plot_h = height - top - bottom
    colors = [BLUE, ORANGE, GREEN, PURPLE]

    put_text(img, title, (left, 35), scale=0.8, thickness=2)
    cv2.rectangle(img, (left, top), (left + plot_w, top + plot_h), BLACK, 1)

    if y_max is None:
        y_max = float(np.nanmax(values)) * 1.15 if values.size else 1.0
    if y_max <= 0:
        y_max = 1.0

    for i in range(6):
        y = top + int(plot_h * i / 5)
        cv2.line(img, (left, y), (left + plot_w, y), GRAY, 1)
        label = y_max - y_max * i / 5
        put_text(img, f"{label:.2f}", (18, y + 5), scale=0.45)

    n_cat = len(categories)
    n_model = len(model_names)
    group_w = plot_w / max(n_cat, 1)
    bar_w = int(group_w * 0.65 / max(n_model, 1))

    for ci, cat in enumerate(categories):
        group_center = left + int(group_w * (ci + 0.5))
        start_x = group_center - (bar_w * n_model) // 2
        for mi, model in enumerate(model_names):
            val = float(values[ci, mi])
            bar_h = int((val / y_max) * plot_h)
            x1 = start_x + mi * bar_w
            x2 = x1 + bar_w - 4
            y1 = top + plot_h - bar_h
            y2 = top + plot_h
            cv2.rectangle(img, (x1, y1), (x2, y2), colors[mi % len(colors)], -1)
            cv2.rectangle(img, (x1, y1), (x2, y2), BLACK, 1)
            put_text(img, f"{val:.3f}", (x1 - 5, max(y1 - 8, top + 12)), scale=0.38)
        put_text(img, cat, (group_center - 35, top + plot_h + 30), scale=0.48)

    legend_x = left + 10
    legend_y = top + 25
    for mi, model in enumerate(model_names):
        y0 = legend_y + mi * 25
        cv2.rectangle(img, (legend_x, y0 - 10), (legend_x + 18, y0 + 8), colors[mi % len(colors)], -1)
        put_text(img, model, (legend_x + 28, y0 + 5), scale=0.5)

    return img


def save_image(path: Path, img: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img)
    print(f"Saved: {path}")


def plot_locator_history() -> None:
    path = CFG.CURVE_DIR / "locator_history.csv"
    if not path.exists():
        print(f"Skip locator curves, missing: {path}")
        return
    df = pd.read_csv(path)
    x = df["epoch"].tolist()
    img1 = draw_line_chart("Locator Loss", x, [("train_loss", df["train_loss"], BLUE), ("val_loss", df["val_loss"], ORANGE)])
    img2 = draw_line_chart("Locator Val Dice", x, [("val_dice", df["val_dice"], GREEN)], y_min=0.0, y_max=1.0)
    summary = np.vstack([img1, img2])
    save_image(CFG.CURVE_DIR / "locator_training_curves.png", summary)


def plot_segmentor_history(display_name: str, csv_name: str) -> None:
    path = CFG.CURVE_DIR / csv_name
    if not path.exists():
        print(f"Skip {display_name} curves, missing: {path}")
        return
    df = pd.read_csv(path)
    x = df["epoch"].tolist()
    loss_img = draw_line_chart(f"{display_name} Loss", x, [("train_loss", df["train_loss"], BLUE), ("val_loss", df["val_loss"], ORANGE)])
    dice_img = draw_line_chart(
        f"{display_name} Class Dice",
        x,
        [
            ("background", df["val_dice_bg"], BLUE),
            ("myocardium", df["val_dice_myo"], GREEN),
            ("edema", df["val_dice_edema"], YELLOW),
            ("scar", df["val_dice_scar"], RED),
        ],
        y_min=0.0,
        y_max=1.0,
    )
    lesion_img = draw_line_chart(
        f"{display_name} Lesion Dice",
        x,
        [
            ("edema", df["val_dice_edema"], YELLOW),
            ("scar", df["val_dice_scar"], RED),
            ("lesion_mean", df["val_dice_lesion_mean"], PURPLE),
        ],
        y_min=0.0,
        y_max=1.0,
    )
    summary = np.vstack([loss_img, dice_img, lesion_img])
    save_image(CFG.CURVE_DIR / f"{display_name.lower()}_training_curves.png", summary)


def read_metric_matrix(metric_col: str) -> Tuple[List[str], List[str], np.ndarray] | None:
    path = CFG.TABLE_DIR / "metrics_comparison.csv"
    if not path.exists():
        print(f"Skip metrics chart, missing: {path}")
        return None
    df = pd.read_csv(path)
    pivot = df.pivot(index="类别", columns="模型", values=metric_col).reindex(CFG.MYOPS_CLASS_NAMES)
    categories = ["BG", "Myo", "Edema", "Scar"]
    models = list(pivot.columns)
    values = pivot.to_numpy(dtype=np.float32)
    return categories, models, values


def plot_metrics_bars() -> None:
    dice_data = read_metric_matrix("Dice均值")
    if dice_data is not None:
        categories, models, values = dice_data
        img = draw_bar_chart("Test Dice Comparison", categories, models, values, y_max=1.05)
        save_image(CFG.CURVE_DIR / "dice_comparison_bar.png", img)

    hd95_data = read_metric_matrix("HD95均值")
    if hd95_data is not None:
        categories, models, values = hd95_data
        img = draw_bar_chart("Test HD95 Comparison", categories, models, values)
        save_image(CFG.CURVE_DIR / "hd95_comparison_bar.png", img)


def plot_summary_dashboard() -> None:
    dice_path = CFG.CURVE_DIR / "dice_comparison_bar.png"
    hd95_path = CFG.CURVE_DIR / "hd95_comparison_bar.png"
    imgs = []
    for path in [dice_path, hd95_path]:
        if path.exists():
            img = cv2.imread(str(path))
            if img is not None:
                imgs.append(img)
    if not imgs:
        print("Skip summary dashboard, no metric charts found.")
        return
    min_w = min(img.shape[1] for img in imgs)
    resized = [cv2.resize(img, (min_w, int(img.shape[0] * min_w / img.shape[1]))) for img in imgs]
    dashboard = np.vstack(resized)
    save_image(CFG.CURVE_DIR / "summary_dashboard.png", dashboard)


def main() -> None:
    ensure_dirs()
    plot_locator_history()
    plot_segmentor_history("Baseline", "baseline_history.csv")
    plot_segmentor_history("Cascade", "cascade_history.csv")
    plot_metrics_bars()
    plot_summary_dashboard()
    print(f"Visualization output dir: {CFG.CURVE_DIR}")


if __name__ == "__main__":
    main()
