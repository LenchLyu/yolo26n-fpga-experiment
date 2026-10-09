# 第 2 步：准备与 RTL 输入一致的 INT8 校准图片

## 当前步骤的边界

本目录只生成 **校准输入图片**，暂不生成 INT8 模型、量化参数或 FPGA 权重文件。
量化方式必须与之后的自定义 PE/NPU 数值格式一起确定；
TensorRT 等软件后端导出的 INT8 文件不能直接当作 FPGA PE 的权重格式。

## 数据来源

只读取原始数据集的 `images/train/`。本地当前有 143 张训练图片。
`images/val/` 的 35 张图片保留用于后续 FP32 与 INT8 精度对比。
校准阶段观察的是输入和中间激活值的分布，因此此步骤不复制目标框标注。

## 每张图片的处理

1. 读取训练照片；如果照片不是 16:9，则报错。
2. 用 OpenCV `INTER_LINEAR` 缩放成 1280×720，模拟当前没有录制下来的相机输入。
3. 取偶数行和偶数列，得到 640×360。
4. 上下各补 140 行 RGB `(114,114,114)`，得到 640×640。
5. 保存无损 PNG，再读回核对尺寸、像素和灰边。

保存的是普通 RGB 视觉图片。OpenCV 读入和内存处理时采用 BGR 排列；
写入 PNG 后颜色语义不变。真正送入模型时，还需要确认所选量化工具
是否把图片转换成 RGB、CHW，并且只做一次 `/255` 归一化。

## 运行

在仓库根目录中，使用安装了 Ultralytics/OpenCV 的 `pytorch` 环境。先把自己的数据集放在 `data/signature/`（`data/` 不提交 Git），或修改 `--dataset-root`：

```powershell
conda activate pytorch
python .\software\quantization\02_calibration_data\prepare_calibration.py --dataset-root ".\data\signature"
```

生成结果位于 `results/images/calib/` 和 `results/manifest.json`。
脚本拒绝覆盖已有 `results/`，以防混入上一轮实验的数据。
这些生成文件已被 `software/.gitignore` 排除。

## 后续

确认模型的算子、张量尺寸和 FPGA 数值格式后，选定 PTQ 工具，
让其读取这批校准输入并记录各层激活值范围。随后在独立的 35 张验证图上
比较 INT8 和当前 FP32+RTL 预处理基线。未来有真实 1280×720 相机样本时，
应优先补充这些样本，以减少“1920×1080 训练照片模拟相机”的差距。
