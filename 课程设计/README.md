# 两阶段级联心脏核磁分割

本项目使用 Python、PyTorch 和 `segmentation_models_pytorch` 实现医学图像分割两阶段级联方案：

1. **U-Net-Locator**：使用 ACDC 的 bSSFP 单帧图像训练心肌区域粗定位器，输出二值定位 mask。
2. **U-Net-Segmentor**：使用 MyoPS 的 LGE、T2、bSSFP 三模态图像，再拼接第一阶段定位 mask，进行背景、正常心肌、水肿、疤痕四分类分割。
3. **对比实验**：同时训练一个只使用 MyoPS 三通道输入的标准 U-Net 作为 Baseline，对比级联模型在水肿和疤痕上的 Dice/HD95。

## 目录结构

```text
.
├── config.py              # 全局配置
├── dataset.py             # ACDC 和 MyoPS 数据加载器
├── models.py              # 定位器、分割器、级联模型、损失函数
├── train_locator.py       # 训练第一阶段定位器
├── train_segmentor.py     # 训练 Baseline 和级联分割器
├── inference.py           # 推理和可视化
├── evaluate.py            # Dice、HD95 指标评估
├── main.py                # 一键运行完整流程
├── utils.py               # 通用工具函数
└── requirements.txt       # 依赖列表
└── visualize_training.py  # 训练曲线可视化
```


## 数据路径

默认数据路径已经写入 `config.py`：

```python
ACDC_ROOT = Path(r"C:\Users\C\Desktop\data\ACDC")
MYOPS_ROOT = Path(r"C:\Users\C\Desktop\data\MyoPS 2020 Dataset")
```

程序会递归扫描数据目录，并根据文件名关键词自动匹配图像和标签。若文件命名特殊，需要在 `config.py` 中修改：

- `LGE_KEYS`
- `T2_KEYS`
- `BSSFP_KEYS`
- `MASK_KEYS`

## 运行方式

### 运行入口

```bash
python main.py
```

该命令会依次执行：

1. 训练定位器；
2. 训练三通道 Baseline U-Net；
3. 训练四通道级联 U-Net；
4. 在测试集上推理并保存可视化；
5. 计算 Dice 和 HD95 并保存表格。
6. 完成训练相关可视化并保存

### 分步运行

```bash
python train_locator.py
python train_segmentor.py
python inference.py
python evaluate.py
```

## 输出结果

训练权重保存在：

```text
outputs/checkpoints/
```

推理结果和可视化保存在：

```text
results/predictions/
results/figures/
results/tables/metrics_comparison.csv
```
训练曲线和指标对比可视化保存在

```text
D:\a\results\curves\locator_training_curves.png
D:\a\results\curves\baseline_training_curves.png
D:\a\results\curves\cascade_training_curves.png
```

四联图内容为：

1. 原始 bSSFP 图；
2. 真实标签；
3. Baseline 预测；
4. 级联模型预测。

颜色约定：

- 背景：黑色
- 正常心肌：蓝色
- 水肿：黄色
- 疤痕：红色

## 注意事项

- 第一次运行 `segmentation_models_pytorch` 的 `resnet34` 编码器需要下载 ImageNet 预训练权重。
- 若无法正常下载，可在 `config.py` 中把 `ENCODER_WEIGHTS = "imagenet"` 改为 `ENCODER_WEIGHTS = None`。
- ACDC 标签默认使用 `2=心肌、3=左心室` 作为定位器前景；如标签定义不同，请修改 `ACDC_FOREGROUND_LABELS`。
- MyoPS 默认支持常见标签值 `200=正常心肌、1220=水肿、2221=疤痕`，也支持已经整理好的 `0/1/2/3` 标签。
