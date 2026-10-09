# 第 1 步：FP32 预处理精度对比

## 目的

使用相同的验证照片、标注和 FP32 模型权重，比较两种 640×640
预处理方法对签名检测的影响：

- `standard`：1280×720 相机输入经过 Ultralytics `LetterBox(auto=False)`；
- `rtl`：1280×720 相机输入取偶数行、偶数列，再上下各补 140 行灰色。

现有验证照片是 1920×1080，因此两组实验都会先用 OpenCV
`INTER_LINEAR` 生成共同的 1280×720 相机输入替代图。
本实验衡量的是这条模拟输入链路，不直接等同于未来真实摄像头的表现。

## 文件

- `prepare_val_sets.py`：生成两套验证图片和同一坐标变换后的标注。
- `evaluate.py`：用相同权重验证两套图片，比较指标和逐图漏检。
- `results/datasets/`：自动生成的数据，不提交 Git。
- `results/summary.json`：总体指标与差值。
- `results/per_image.csv`：使用置信度阈值 0.25、匹配 IoU 阈值 0.5 时，每张图的 TP、FP、FN。

## 运行

在 `yolo26n` 目录执行：

先把自己的数据集放在仓库根目录的 `data/signature/`（`data/` 不提交 Git），或在命令中把 `--dataset-root` 改为自己的目录。

```powershell
conda activate pytorch
python .\software\quantization\01_fp32_preprocess_accuracy\prepare_val_sets.py --dataset-root ".\data\signature"
python .\software\quantization\01_fp32_preprocess_accuracy\evaluate.py
```

默认使用 CPU。要使用第一块 GPU，可在第二条命令末尾加 `--device 0`。

脚本源代码没有写死训练数据集的绝对路径。其他人克隆仓库后，
只需在第一条命令中传入自己的 `--dataset-root`。
模型权重默认按软件工作区内的相对位置查找，也可用 `--weights` 指定。

## 如何判断

先看 `results/summary.json` 中的 precision、recall、map50、
map50_95 和 rtl_minus_standard；再看 `results/per_image.csv`，
重点检查 rtl_extra_fn 大于 0 的照片。

总体 mAP 验证使用 Ultralytics 的低置信度阈值 0.001，以覆盖完整的
精度召回曲线；逐图 TP、FP、FN 则另外在实际检测阈值 0.25 下统计。

只有比较完成后，才决定是否继续用当前 RTL 取样方式，
以及后续 INT8 校准数据应采用哪一种预处理。

## 本次运行结果（2026-10-08）

使用 `software/models/signature_yolo26n/best_stripped.pt`、Ultralytics 8.4.41、
CUDA GPU、`half=False` 验证了 35 张图。两组共用同一份变换后的标注。

| 指标 | standard | rtl |
|---|---:|---:|
| Precision | 1.000000 | 1.000000 |
| Recall | 0.959677 | 0.964005 |
| mAP50 | 0.990909 | 0.991667 |
| mAP50-95 | 0.957901 | 0.958109 |

另以 `conf=0.25`、匹配 `IoU>=0.5` 统计逐图检测：两组均为 34 TP、
1 FN，漏检均发生于 `Frame_48.png`。standard 为 4 FP，rtl 为 3 FP；
差异来自 `Frame_244.png`。在这 35 张验证图上，当前 RTL 取样没有造成新增漏检。
数据量较小，这些轻微指标差异不足以证明 RTL 方案通常优于线性缩放。
