# YOLO26n 卷积 W8A8 PTQ 与 PE 数值约定

本阶段处理已完成 RTL 取样与灰边预处理的 640×640 图像。**候选模型仅将 ONNX 图中的 Conv 量化为 W8A8；注意力、SiLU、上采样、拼接和检测后处理继续使用浮点。** 它是验证 PE 卷积数值格式的混合精度软件模型，不是整网 INT8，也不是可直接装入 FPGA 的 NPU。

## 文件与数据

| 文件 | 作用 |
| --- | --- |
| `audit_export.py` | 复制推理权重、导出固定 `(1,3,640,640)` 的 FP32 ONNX、列出每个推理节点的输入输出形状及卷积类型。 |
| `evaluate.py` | 先比较原 `.pt` 与 FP32 ONNX；通过前置门槛后，再比较 QDQ 候选模型。记录 mAP 与 `conf=0.25`、匹配 IoU≥0.5 的逐图 TP/FP/FN。 |
| `calibrate.py` | 读取 143 张独立校准 PNG，用 MinMax 收集卷积输入输出激活范围，并用 ONNX Runtime 生成仅 Conv 的对称 S8 QDQ 模型。 |
| `pe_numeric_check.py` | 从 QDQ 图提取各卷积实际使用的尺度、零点、INT32 偏置，验证最坏累加上界并做三类卷积的整数算例。 |
| `results/` | 生成的 ONNX、审计、范围、量化参数、逐图结果、模型和数据哈希；由 `.gitignore` 忽略。 |

默认模型为 `software/models/signature_yolo26n/best_stripped.pt`；校准集是 `software/quantization/02_calibration_data/results/images/calib/` 下的 143 张 PNG；验证数据由 `software/quantization/01_fp32_preprocess_accuracy/results/datasets/rtl/data.yaml` 指向 35 张 PNG。模型、校准集和验证集都由前序阶段生成；脚本会检查数量、文件哈希和校准/验证图像不重合。可用 `--weights`、`--calib-dir`、`--val-data`、`--onnx`、`--out` 更改路径；每个脚本的 `--help` 列出其参数。

## 重跑顺序

在 `yolo26n/` 根目录、已安装 PyTorch 和 `software/requirements.txt` 的环境中执行：

```powershell
conda activate pytorch
python .\software\quantization\03_int8_ptq\audit_export.py
python .\software\quantization\03_int8_ptq\evaluate.py --stage preflight
python .\software\quantization\03_int8_ptq\calibrate.py
python .\software\quantization\03_int8_ptq\pe_numeric_check.py
python .\software\quantization\03_int8_ptq\evaluate.py --stage candidate
```

步骤含义与停止条件：

1. **算子审计与导出。**原权重先复制到 `results/model_source/`；Ultralytics 导出 FP32 ONNX，固定批量 1、输入 640×640、opset 17，保留 `(1,300,6)` 的端到端检测输出。原权重不会被修改。`operator_manifest.json` 记录全部节点连接、形状、算子类型、版本和文件 SHA-256。
2. **前置等价性。**对同一 35 张 RTL 验证图，使用相同的 640×640、`rect=False`、CPU、FP32 和 Ultralytics 检测输出解析。mAP50-95 差异必须≤0.005，`conf=0.25` 时无新增漏检。未通过则停止量化；报告写入 `preflight.json`。
3. **校准与量化。**每张 PNG 已是 640×640 的 RTL 预处理结果；这里只执行一次 BGR→RGB、HWC→CHW、`float32 / 255`，没有第二次缩放或补边。MinMax 范围在 `activation_ranges.json/.csv`；QDQ 模型在 `conv_w8a8_qdq.onnx`，配置与哈希在 `quantization_manifest.json`。
4. **PE 数值核查。**`pe_numeric_contract.json` 给出每层输入/输出尺度、每输出通道的权重尺度、INT32 偏置、重定标系数、每输出元素的乘加项数和保守溢出上界。权重整数保存在 QDQ ONNX initializer 中，由其 SHA-256 绑定。脚本取普通 3×3、1×1、深度卷积各一个真实权重切片，核对整数乘加、舍入和饱和规则。
5. **保留集验收。**候选相对 FP32+RTL 基线 mAP50-95 下降须≤0.03，新增漏检图片最多 1 张。未通过则保留候选及诊断，不把这些参数定为 PE 最终实现依据。`candidate_eval.json` 保存逐图统计和文件哈希。

重新训练、换图或修改 ONNX 后须从第 1 步重新运行；后续脚本会拒绝与前置报告哈希不一致的输入。`results/` 中可出现 ONNX Runtime 的校准增强图及 Ultralytics 验证中间文件，它们不是部署产物。

## 本次结果

推理图共有 102 个 Conv：39 个普通卷积、55 个 1×1 卷积和 8 个深度卷积；其余可见算子包括 Sigmoid/Mul、MatMul/Softmax、Resize、Concat 与检测输出计算。FP32 ONNX 与原 `.pt` 的 35 图 mAP50-95 均为 **0.958109**，没有新增漏检，前置门槛通过。

候选模型含 102 个 Conv、194 个 QuantizeLinear、398 个 DequantizeLinear。候选 mAP50-95 为 **0.930091**，比 FP32+RTL 低 **0.028018**，在 0.03 门槛内；`conf=0.25` 时仍仅漏检 `Frame_48.png`，无新增漏检。143 张校准图均被读取，35 张验证图未参与校准。最大卷积乘加项数为 2304；按所有输入取 S8 全范围估算，最大单通道 INT32 累加绝对值上界为 4,800,969，低于 2,147,483,647。具体数字以本机 `results/` 内报告为准。

## PE 第一版数值约定

- 激活：NCHW、有符号 INT8，每张量一个正尺度，零点 0。
- 权重：OIHW、有符号 INT8，每输出通道一个正尺度，零点 0。导出的 Conv 权重已融合 BatchNorm。
- 偏置：每输出通道 INT32，尺度 `Sx × Sw[c]`；INT32 累加器先加载该偏置，再逐项加 `xq × wq`。
- 输出：`clip(round_ties_to_even(acc × Sx × Sw[c] / Sy), -128, 127)`，输出零点 0。`round_ties_to_even` 表示恰好在两整数中间时取偶数；饱和发生在舍入后。
- 最坏界：`128 × sum(abs(wq[c])) + abs(bias_int32[c])`。此界保守覆盖任意 S8 输入；每层均已检查不超过 INT32。

QDQ ONNX 用于功能和精度评估；这里的整数算例用于固定未来 PE 的数值规则。它**尚未**证明整张特征图与未来 RTL 按位一致，也尚未完成 PE 的流水线、缓存、访存或整网硬件实现。35 张验证图很少，门槛通过只代表该保留集。

## 注意力与后续硬件化

当前 ONNX 推理图中另有 4 个 MatMul 和 2 个 Softmax，它们属于注意力路径且仍为浮点。FPGA 可以用浮点运算 IP 实现这部分，代价是需要评估器件资源、时序与吞吐；现有 QDQ 模型只是软件候选，不会自动变成这种硬件电路。全部推理要放在 PL 时，必须为 MatMul、Softmax 和其他剩余算子安排 PL 实现。可根据资源预算选择浮点 IP 或定点近似；先用一张代表图片跑通检测框，再考虑扩大精度回归。Softmax 是非线性归一化，不能仅把 ONNX 的 MatMul 改成 INT8 就称为“注意力全 INT8”。

项目最终要求全部推理计算（含检测框后处理）发生在 PL，PS/电脑只负责数据装载、启动和显示。由于当前无开发板，这一要求将通过完整 RTL/HLS 设计的仿真检测框、目标器件综合和实现报告验证；本目录的 ONNX 精度结果只能充当功能参考，不能充当 PL 完成证明。
