# 软件侧仿真

这里用于放未来 NPU 算子的 Python 参考实现，包括张量布局转换、量化卷积、偏置与重定标，以及与 RTL 输出逐层比较的脚本。输入生成和数值检查代码放在本目录；较大的张量或追踪结果放到 `outputs/`。

现有 AXI 图像流 testbench 位于 `project_1/project_1.srcs/sim_1/new/`，像素检查脚本是 `project_1/scripts/output_check.py`。预处理对照脚本位于 `software/quantization/00_preprocess_check/`。目前还没有 PE 或 NPU 仿真器。
