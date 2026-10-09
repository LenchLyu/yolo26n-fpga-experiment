# YOLO26n FPGA 部署工作区

> 这是我个人的一次小尝试，代码很粗糙，甚至可以说非常垃圾；上传主要是留个学习记录，传着玩玩。项目尚未完成，请不要把它当作成熟的 FPGA NPU 实现。

本目录用于把签名检测模型逐步部署到 Zynq UltraScale+ MPSoC。目前已有 PS DDR → AXI VDMA → 自定义图像预处理 RTL 的链路；卷积计算和 NPU/PE 仍在规划中。

| 目录 | 内容 |
| --- | --- |
| [software/](software/README.md) | 模型副本、量化、软件检测和软件侧仿真 |
| `project_1/` | Vivado Block Design、RTL、SystemVerilog testbench 和图像流测试数据 |
| `vitis_workspace/` | Vitis 工作区 |

软件工作的入口是 [software/README.md](software/README.md)；目前的功能边界和下一阶段验收条件见 [software/REQUIREMENTS.md](software/REQUIREMENTS.md)。原训练工程单独保存；本地推理权重位于 `software/models/`，默认不提交到 Git。

目前已完成 1280×720 到 640×640 的图像预处理 RTL 和真实图片行为仿真，以及软件侧 YOLO26n 卷积量化、ONNX 算子审计、张量形状与 FP32 内存规划。完整模型的 PL 推理、检测框输出和上板验证尚未完成。下一阶段工作见 [software/pl_inference/PLAN.md](software/pl_inference/PLAN.md)。

Vivado 工程使用 2024.2 创建，`project_1.xpr` 中保留了原电脑的工程路径，换目录打开时可能需要重新定位源文件。`image_preprocess_core.v` 是早期未使用的练习草稿，当前通过仿真的图像缩放补边逻辑在 `xis_resize_letterbox_1280x720_to_640x640.v` 等模块中。
