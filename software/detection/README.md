# 软件检测

`predict.py` 默认读取 `../models/signature_yolo26n/best_stripped.pt`，使用 `imgsz=640, rect=False` 进行固定 640×640 输入的签名检测。默认用 CPU；有可用 GPU 时可指定 `--device 0`。

从 `yolo26n/` 仓库根目录运行，例如训练数据位于相邻目录时：

```powershell
python .\software\detection\predict.py --source '..\模型训练与部署实战\datasets\signature\images\val\Frame_108.jpg'
```

脚本打印类别、置信度和原图坐标，带框图片默认写到 `software/detection/outputs/signature_result.jpg`。可用 `--weights`、`--output`、`--conf`、`--device` 覆盖默认值；输入图片也可以是你自己选择的本地文件。

本目录的检测结果用于验证模型文件能加载及软件推理能运行。量化前的精度比较需要完整验证集，不能以单张图的检测结果代替。
