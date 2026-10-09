# 模型副本

`signature_yolo26n/best_stripped.pt` 复制自训练工程的 `runs/signature_yolo26n/weights/best_stripped.pt`。训练工程仍是原始记录；此副本用于 FPGA 工作区的软件检测和后续导出实验。

- 任务：单类别签名检测。
- 本工程软件参考输入：固定 640×640。
- 文件类型：PyTorch 检查点，**不是 INT8 模型**。
- SHA-256: `D861A8863AFFE791463A2C78C28EB8E8731140994E760EE9E44B30D79B68B04B`.

后续 ONNX/INT8 导出应使用独立文件名，并先记录校准数据来源、导出配置和精度结果，再生成硬件参数。
