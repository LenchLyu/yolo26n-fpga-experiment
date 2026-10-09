# Step 0: preprocessing comparison

中文说明见 [GUIDE.zh-CN.md](GUIDE.zh-CN.md)。

Run from the `yolo26n/` root:

```powershell
python .\software\quantization\00_preprocess_check\check_preprocess.py --image .\project_1\testdata\axis_resize_input_1280x720.png --sim-hex .\project_1\testdata\sim_output_rgb.hex
```

The script compares three images: the XSim RGB output, the image expected from the current RTL, and the fixed 640×640 Ultralytics `LetterBox` reference. The current RTL prediction uses padding 114 by default. For the historical black-padding version, pass `--rtl-pad 0` and use its matching old simulation output.

The current XSim output and RTL expectation match in all 409,600 pixels. Compared with the fixed Ultralytics reference, the remaining differences are in the 640×360 image area: 44,332 pixels differ on the synthetic test input (10.8232% of the full frame). Padding now matches exactly. A single real validation frame, first scaled to a 1280×720 camera-input proxy, differs in 117,389 pixels (28.6594% of the full frame), with mean absolute active-area channel difference 0.9459/255. Pixel differences do not determine detection accuracy.

The script saves PNG previews, a magenta difference map, and `report.json` under its local `results/` directory. Its source images remain in `project_1/testdata/` or in the separate training dataset.
