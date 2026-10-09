from pathlib import Path

import numpy as np
from PIL import Image


# 本脚本位于 project_1/scripts/ 下，所以 parents[1] 是 project_1 根目录。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
TESTDATA_DIR = PROJECT_ROOT / "testdata"

INPUT_IMAGE = TESTDATA_DIR / "axis_resize_input_1280x720.png"
OUTPUT_HEX = TESTDATA_DIR / "input_rgb.hex"

SRC_WIDTH = 1280
SRC_HEIGHT = 720


def main():
    if not INPUT_IMAGE.is_file():
        raise FileNotFoundError(f"找不到输入图片：{INPUT_IMAGE}")

    TESTDATA_DIR.mkdir(parents=True, exist_ok=True)

    with Image.open(INPUT_IMAGE) as image:
        if image.size != (SRC_WIDTH, SRC_HEIGHT):
            raise ValueError(
                f"图片尺寸应为 {SRC_WIDTH}x{SRC_HEIGHT}，"
                f"实际为 {image.width}x{image.height}"
            )

        # 统一转换成 RGB888，忽略 PNG 可能包含的透明度通道。
        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)

    # 每个像素按 RRGGBB 打包为 24 位数值。
    pixels = (
        (rgb[:, :, 0].astype(np.uint32) << 16)
        | (rgb[:, :, 1].astype(np.uint32) << 8)
        | rgb[:, :, 2].astype(np.uint32)
    )

    # 一行一个像素，格式如 12AB34，供 Verilog $readmemh 读取。
    with OUTPUT_HEX.open("w", encoding="ascii", newline="\n") as file:
        np.savetxt(file, pixels.reshape(-1, 1), fmt="%06X")

    expected_count = SRC_WIDTH * SRC_HEIGHT
    print(f"输入图片：{INPUT_IMAGE}")
    print(f"输出数据：{OUTPUT_HEX}")
    print(f"图片尺寸：{SRC_WIDTH}x{SRC_HEIGHT}")
    print(f"像素数量：{expected_count}")
    print("像素格式：RGB888，按 RRGGBB 写入，每行一个像素")


if __name__ == "__main__":
    main()