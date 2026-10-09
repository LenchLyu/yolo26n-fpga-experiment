# 第 1、2 阶段怎么执行：图审计与初版 DDR 工作区

这一步只处理固定 YOLO26n 的**软件导出图**，为 PL 模块确定算子、张量和存储需求。此时先不在 Vivado 里添加 PE，也不修改已验证的图像预处理 RTL。

## 运行前确认

- 使用现有 Conda `pytorch` 环境，安装 `software/requirements.txt` 中的 ONNX、ONNX Runtime、NumPy 与 OpenCV。
- 已有 `software/quantization/03_int8_ptq/results/model_source/best_stripped.onnx`。若缺失，先从 `yolo26n/` 根目录运行 `python .\software\quantization\03_int8_ptq\audit_export.py`。
- 已有 143 张 640×640 校准 PNG；脚本只用其中 3 张确认检测头张量形状，不读 35 张验证图。

## 顺序执行

先在 PowerShell 中切换到你本机的 `yolo26n/` 仓库根目录，然后执行：

```powershell
conda activate pytorch
python .\software\pl_inference\audit_graph.py
python .\software\pl_inference\resolve_shapes.py
python .\software\pl_inference\plan_memory.py
```

脚本默认用自身位置寻找输入和输出，不用在代码里改个人磁盘路径。需要更换输入时，分别运行 `python .\software\pl_inference\audit_graph.py --help` 等查看 `--onnx`、`--calib-dir`、`--out` 参数。第三个脚本必须在前两个成功后运行。

### 命令 1：`audit_graph.py`

读取固定输入的 **FP32 ONNX 拓扑图**，不把 QDQ 节点当作硬件层。每一种运行时算子都先指定一个 PL 运算类别；遇到未知类型立即报错。输出：

- `results/operator_coverage.csv`：算子种类、数量、预定 PL 单元。
- `results/first_pass_graph.json`：452 个节点的连接、已知张量形状/类型及最后使用位置。
- `results/unresolved_tensors.csv`：静态形状推断尚未确定的张量。

预期提示：**452 nodes / 28 operator types / 26 unresolved tensors**。`Constant` 是离线常量，其余运行时运算必须在 PL。

### 命令 2：`resolve_shapes.py`

为了给检测头分配内存，把第一步的 26 个中间值临时暴露为 ONNX 输出，并在 3 张独立校准图上运行 ONNX Runtime。要求这 3 次的形状和类型完全一致。输出 `results/observed_shapes.json`。这只是对固定输入的**实测形状**；今后若换模型或尺寸，须重新运行。

预期提示：**26 tensors on 3 calibration images**。不应通过猜测填入检测头的符号维度。

### 命令 3：`plan_memory.py`

用生产者/最后消费者位置规划可复用的 DDR 暂存区。不同张量若存活时间不重叠，可重复使用同一段地址；脚本会检查同时存活的缓冲区绝不重叠。输出：

- `results/memory_intervals_fp32.csv`：每个运行时张量的生命周期、大小、相对偏移。
- `results/memory_plan_fp32.json`：完整初版分配与峰值统计。

本次结果：**423 个运行时张量，64 字节对齐，暂存区高水位 24,576,000 字节（约 23.44 MiB），同时存活峰值 19,660,800 字节（约 18.75 MiB）**。这份规划按 FP32 ONNX 每个中间输出都实际存储的保守方式计算，**不含权重**，也不是最终量化/定点布局；其中偏移是相对地址，不是 Zynq DDR 的物理地址。它说明需要 DDR 工作区，不代表要在 PL 片上放 24 MB。

## 看结果与下一步

```powershell
Import-Csv .\software\pl_inference\results\operator_coverage.csv | Format-Table -AutoSize
Import-Csv .\software\pl_inference\results\memory_intervals_fp32.csv | Select-Object -First 12 | Format-Table -AutoSize
```

这三个脚本解决了“有哪些算子、检测头形状是什么、缓冲区大致多少”三个问题。**写 PE 前还需确定**：非卷积算子的 PL 数值格式、哪些 Reshape/Split 等可只用地址视图而不复制、权重与工作区的实际 DDR 基址、AXI 端口宽度和突发读写方案。然后生成最终硬件指令和二进制参数，才开始编写或综合 PE。

所有 `results/` 生成物被 Git 忽略；脚本及这份说明可以提交仓库。完整后续阶段见 [PLAN.md](PLAN.md)。
