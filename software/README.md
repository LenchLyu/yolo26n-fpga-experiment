# 软件工作区

这里集中放置部署侧的软件文件。以下命令从克隆后的 `yolo26n/` 仓库根目录运行；脚本通过命令行参数接收输入路径，避免依赖某台电脑的绝对路径。

| 目录 | 当前用途 |
| --- | --- |
| [models/](models/README.md) | 保存训练所得推理权重的副本，以及模型来源和校验信息 |
| [quantization/](quantization/README.md) | 预处理对照、后续校准、INT8 导出与精度比较 |
| [detection/](detection/README.md) | 用训练好的模型进行软件检测 |
| [simulation/](simulation/README.md) | 后续放算子、PE、NPU 的 Python 参考计算及 RTL 数值对照 |
| [pl_inference/](pl_inference/README.md) | 固定 YOLO26n 的全 PL 推理范围、算子覆盖与无开发板验证路线 |

当前功能、输入输出约定和进入下一阶段的条件见 [REQUIREMENTS.md](REQUIREMENTS.md)；环境和依赖安装说明见 [ENVIRONMENT.md](ENVIRONMENT.md)。`requirements.txt` 是供 `pip` 读取的依赖清单，人工阅读以这两个 Markdown 文件为准。

## 快速检查

从 `yolo26n/` 根目录激活现有环境，运行当前预处理对照：

```powershell
conda activate pytorch
python .\software\quantization\00_preprocess_check\check_preprocess.py --image .\project_1\testdata\axis_resize_input_1280x720.png --sim-hex .\project_1\testdata\sim_output_rgb.hex
```

软件检测通过 `--source` 指定图片。例如，训练数据位于相邻目录时：

```powershell
python .\software\detection\predict.py --source '..\模型训练与部署实战\datasets\signature\images\val\Frame_108.jpg'
```

训练工程仍是原始权重与训练记录的来源；本目录的 `best_stripped.pt` 是部署实验用副本，不是 INT8 模型。预处理第 0 步的脚本和结果见 [quantization/00_preprocess_check/README.md](quantization/00_preprocess_check/README.md)。
