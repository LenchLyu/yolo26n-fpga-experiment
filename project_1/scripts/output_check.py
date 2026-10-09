from pathlib import Path

import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TESTDATA_DIR = PROJECT_ROOT / "testdata"

INPUT_IMAGE = TESTDATA_DIR / "axis_resize_input_1280x720.png"
SIM_OUTPUT_HEX = TESTDATA_DIR / "sim_output_rgb.hex"

SIM_OUTPUT_PNG = TESTDATA_DIR / "sim_output.png"
EXPECTED_OUTPUT_PNG = TESTDATA_DIR / "expected_output.png"
DIFF_OUTPUT_PNG = TESTDATA_DIR / "difference.png"

SRC_WIDTH = 1280
SRC_HEIGHT = 720

OUTPUT_WIDTH = 640
OUTPUT_HEIGHT = 640

RESIZED_WIDTH = 640
RESIZED_HEIGHT = 360
PAD_TOP = 140


def read_hex_pixels(hex_path):
    """读取一行一个 24 位 RGB 像素的 hex 文件。"""

    def values_from_file(file):
        for line_number, line in enumerate(file, start=1):
            # 允许空行和 // 注释。
            token = line.partition("//")[0].strip()
            if not token:
                continue

            # 允许十六进制数据中带下划线或 0x 前缀。
            token = token.replace("_", "")
            if token.lower().startswith("0x"):
                token = token[2:]

            try:
                value = int(token, 16)
            except ValueError as exc:
                raise ValueError(
                    f"{hex_path} 第 {line_number} 行不是有效的十六进制像素："
                    f"{line.strip()!r}"
                ) from exc

            if value < 0 or value > 0xFFFFFF:
                raise ValueError(
                    f"{hex_path} 第 {line_number} 行超出 24 位 RGB 范围：{token}"
                )

            yield value

    if not hex_path.is_file():
        raise FileNotFoundError(
            f"找不到仿真输出文件：{hex_path}\n"
            "请确认 testbench 已运行，并把输出写到了 testdata/sim_output_rgb.hex。"
        )

    with hex_path.open("r", encoding="ascii") as file:
        return np.fromiter(values_from_file(file), dtype=np.uint32)


def words_to_rgb(words):
    """把 24 位 RRGGBB 数值数组转换成 RGB 图像数组。"""
    rgb = np.empty((words.size, 3), dtype=np.uint8)
    rgb[:, 0] = (words >> 16) & 0xFF
    rgb[:, 1] = (words >> 8) & 0xFF
    rgb[:, 2] = words & 0xFF
    return rgb.reshape((OUTPUT_HEIGHT, OUTPUT_WIDTH, 3))


def make_expected_image():
    """按当前 RTL 的隔行隔列取样和上下补黑边逻辑生成预期图。"""
    if not INPUT_IMAGE.is_file():
        raise FileNotFoundError(f"找不到输入图片：{INPUT_IMAGE}")

    with Image.open(INPUT_IMAGE) as image:
        if image.size != (SRC_WIDTH, SRC_HEIGHT):
            raise ValueError(
                f"输入图片尺寸应为 {SRC_WIDTH}x{SRC_HEIGHT}，"
                f"实际为 {image.width}x{image.height}"
            )
        src = np.asarray(image.convert("RGB"), dtype=np.uint8)

    # 取偶数行、偶数列：720x1280 -> 360x640。
    resized = src[::2, ::2, :]

    if resized.shape != (RESIZED_HEIGHT, RESIZED_WIDTH, 3):
        raise ValueError(f"缩放结果尺寸异常：{resized.shape}")

    expected = np.full((OUTPUT_HEIGHT, OUTPUT_WIDTH, 3),114, dtype=np.uint8)

    # 放置到画布中间：第 140 到 499 行，共 360 行。
    expected[PAD_TOP:PAD_TOP + RESIZED_HEIGHT, :, :] = resized
    return expected


def main():
    expected_pixel_count = OUTPUT_WIDTH * OUTPUT_HEIGHT
    words = read_hex_pixels(SIM_OUTPUT_HEX)

    if words.size != expected_pixel_count:
        raise ValueError(
            f"仿真输出像素数量不正确：期望 {expected_pixel_count}，"
            f"实际 {words.size}。\n"
            "请检查 testbench 是否完整运行，以及每次输出握手是否只记录一次像素。"
        )

    actual = words_to_rgb(words)
    expected = make_expected_image()

    # 保存两张图片，便于直接打开查看。
    Image.fromarray(actual).save(SIM_OUTPUT_PNG)
    Image.fromarray(expected).save(EXPECTED_OUTPUT_PNG)

    mismatch = np.any(actual != expected, axis=2)
    mismatch_count = int(np.count_nonzero(mismatch))

    print(f"仿真输出图片：{SIM_OUTPUT_PNG}")
    print(f"预期结果图片：{EXPECTED_OUTPUT_PNG}")
    print(f"输出像素数量：{words.size}")

    if mismatch_count == 0:
        print("PASS：仿真输出与预期结果逐像素完全一致。")
        return

    # 生成差异图：黑色代表一致，洋红色代表该像素至少有一个通道不同。
    diff = np.zeros_like(actual)
    diff[mismatch] = (255, 0, 255)
    Image.fromarray(diff).save(DIFF_OUTPUT_PNG)

    y, x = np.argwhere(mismatch)[0]
    actual_rgb = actual[y, x].tolist()
    expected_rgb = expected[y, x].tolist()

    print(f"FAIL：有 {mismatch_count} 个像素与预期不一致。")
    print(f"第一个差异坐标：x={x}, y={y}")
    print(f"仿真 RGB：{actual_rgb}")
    print(f"预期 RGB：{expected_rgb}")
    print(f"差异图：{DIFF_OUTPUT_PNG}")


if __name__ == "__main__":
    main()
