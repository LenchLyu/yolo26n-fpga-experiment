# 软件环境与依赖

当前已验证的本机环境是 Conda `pytorch`：Python 3.11、PyTorch 2.0.0、Ultralytics 8.4.41、ONNX 1.21.0 和 ONNX Runtime 1.25.0，另使用 NumPy、OpenCV 和 Pillow。不同电脑可以选择适合自己 CPU/GPU 的 PyTorch 安装方式。

在 PowerShell 中先进入你自己的 `yolo26n/` 仓库根目录，再激活环境：

```powershell
conda activate pytorch
```

`software/requirements.txt` 是可供 `pip` 使用的机器可读依赖清单；此文件解释其用途。首次配置新环境时，在安装合适的 PyTorch 后运行：

```powershell
python -m pip install -r .\software\requirements.txt
```

不要求把 Conda 环境目录或 Python 包复制进仓库。预处理脚本需要 `ultralytics`、`opencv-python`、`numpy` 和 `Pillow`；检测脚本需要 `ultralytics` 与 PyTorch；卷积 PTQ 脚本还需要 `onnx` 和 `onnxruntime`，以 CPU 执行候选模型的功能及精度评估。Vivado 仿真在硬件工程中运行，不由这些 Python 依赖替代。

验证环境是否可用：

```powershell
python -c "import ultralytics, cv2, numpy, PIL, torch, onnx, onnxruntime; print(ultralytics.__version__, torch.__version__, onnx.__version__, onnxruntime.__version__)"
```
