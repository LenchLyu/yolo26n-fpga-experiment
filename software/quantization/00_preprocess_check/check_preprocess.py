"""Compare the current 1280x720 RTL preprocessing with Ultralytics LetterBox.

Example (from any working directory):
    python check_preprocess.py --image path/to/1280x720.png
    python check_preprocess.py --image path/to/1280x720.png --sim-hex path/to/sim_output_rgb.hex

The script writes its results next to itself by default, under results/<image stem>/.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from ultralytics.data.augment import LetterBox


CAPTURE_W = 1280
CAPTURE_H = 720
OUTPUT_W = 640
OUTPUT_H = 640
ACTIVE_H = 360
PAD_TOP = 140


def read_bgr(path: Path) -> np.ndarray:
    # imdecode works with non-ASCII paths on Windows.
    raw = np.frombuffer(path.read_bytes(), dtype=np.uint8)
    image = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not decode image: {path}")
    return image


def save_rgb(path: Path, rgb: np.ndarray) -> None:
    Image.fromarray(rgb).save(path)


def rtl_expected_rgb(capture_bgr: np.ndarray, pad_value: int) -> np.ndarray:
    captured_rgb = cv2.cvtColor(capture_bgr, cv2.COLOR_BGR2RGB)
    result = np.full((OUTPUT_H, OUTPUT_W, 3), pad_value, dtype=np.uint8)
    result[PAD_TOP : PAD_TOP + ACTIVE_H] = captured_rgb[::2, ::2]
    return result


def read_sim_hex(path: Path) -> np.ndarray:
    with path.open("r", encoding="ascii") as stream:
        words = np.fromiter((int(line.strip(), 16) for line in stream if line.strip()), dtype=np.uint32)
    if words.size != OUTPUT_W * OUTPUT_H:
        raise ValueError(f"Expected {OUTPUT_W * OUTPUT_H} pixels in {path}, found {words.size}")
    rgb = np.empty((words.size, 3), dtype=np.uint8)
    rgb[:, 0] = (words >> 16) & 255
    rgb[:, 1] = (words >> 8) & 255
    rgb[:, 2] = words & 255
    return rgb.reshape(OUTPUT_H, OUTPUT_W, 3)


def compare(a: np.ndarray, b: np.ndarray) -> dict[str, float | int]:
    delta = np.abs(a.astype(np.int16) - b.astype(np.int16))
    return {
        "different_pixels": int(np.count_nonzero(np.any(delta != 0, axis=2))),
        "total_pixels": int(a.shape[0] * a.shape[1]),
        "different_pixel_percent": round(float(np.mean(np.any(delta != 0, axis=2)) * 100), 4),
        "mean_absolute_channel_error_0_to_255": round(float(delta.mean()), 4),
        "maximum_channel_error": int(delta.max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True, help="A 1280x720 input image, or a 16:9 source to resize to 1280x720")
    parser.add_argument("--sim-hex", type=Path, help="Optional XSim output with one RRGGBB pixel per line")
    parser.add_argument("--out", type=Path, help="Output directory (default: results/<image stem> next to this script)")
    parser.add_argument("--rtl-pad", type=int, default=114, choices=range(256), metavar="0..255", help="RTL RGB padding byte (default: 114; use 0 for the old black-padding RTL)")
    args = parser.parse_args()

    source_bgr = read_bgr(args.image)
    original_h, original_w = source_bgr.shape[:2]
    if original_w * CAPTURE_H != original_h * CAPTURE_W:
        raise ValueError(f"Source aspect ratio must be 16:9, got {original_w}x{original_h}")
    was_resized = (original_w, original_h) != (CAPTURE_W, CAPTURE_H)
    capture_bgr = cv2.resize(source_bgr, (CAPTURE_W, CAPTURE_H), interpolation=cv2.INTER_LINEAR) if was_resized else source_bgr

    out = args.out or Path(__file__).resolve().parent / "results" / args.image.stem
    out.mkdir(parents=True, exist_ok=True)

    # The fixed-square reference is suitable for a static 640x640 accelerator input.
    yolo_fixed_bgr = LetterBox(new_shape=(OUTPUT_H, OUTPUT_W), auto=False)(image=capture_bgr.copy())
    # Ultralytics predict() defaults to rect=True for a same-shape .pt batch.
    yolo_rect_bgr = LetterBox(new_shape=(OUTPUT_H, OUTPUT_W), auto=True, stride=32)(image=capture_bgr.copy())

    capture_rgb = cv2.cvtColor(capture_bgr, cv2.COLOR_BGR2RGB)
    rtl_rgb = rtl_expected_rgb(capture_bgr, args.rtl_pad)
    yolo_fixed_rgb = cv2.cvtColor(yolo_fixed_bgr, cv2.COLOR_BGR2RGB)
    yolo_rect_rgb = cv2.cvtColor(yolo_rect_bgr, cv2.COLOR_BGR2RGB)

    save_rgb(out / "capture_1280x720_rgb.png", capture_rgb)
    save_rgb(out / "rtl_expected_640x640_rgb.png", rtl_rgb)
    save_rgb(out / "yolo_fixed_640x640_rgb.png", yolo_fixed_rgb)
    save_rgb(out / "yolo_predict_rect_rgb.png", yolo_rect_rgb)
    mismatch = np.any(rtl_rgb != yolo_fixed_rgb, axis=2)
    difference = np.zeros_like(rtl_rgb)
    difference[mismatch] = (255, 0, 255)
    save_rgb(out / "difference_map.png", difference)

    report: dict[str, object] = {
        "source": str(args.image.resolve()),
        "source_shape_wh": [original_w, original_h],
        "capture_shape_wh": [CAPTURE_W, CAPTURE_H],
        "capture_resized_from_source": was_resized,
        "rtl": f"RGB888; even rows/columns; 640x360; RGB padding {args.rtl_pad} for 140 rows top/bottom",
        "yolo_fixed": "Ultralytics LetterBox auto=False; cv2.INTER_LINEAR; padding=114",
        "yolo_fixed_shape_wh": [int(yolo_fixed_rgb.shape[1]), int(yolo_fixed_rgb.shape[0])],
        "yolo_predict_rect_shape_wh": [int(yolo_rect_rgb.shape[1]), int(yolo_rect_rgb.shape[0])],
        "rtl_vs_yolo_fixed": compare(rtl_rgb, yolo_fixed_rgb),
        "padding_region": {
            "pixel_count": OUTPUT_W * (OUTPUT_H - ACTIVE_H),
            "rtl_rgb": [args.rtl_pad] * 3,
            "yolo_rgb": [114, 114, 114],
        },
        "active_region_rtl_vs_yolo": compare(
            rtl_rgb[PAD_TOP : PAD_TOP + ACTIVE_H],
            yolo_fixed_rgb[PAD_TOP : PAD_TOP + ACTIVE_H],
        ),
        "model_tensor_after_yolo_preprocess": "RGB, CHW, floating-point pixel values divided by 255",
    }

    if args.sim_hex:
        sim_rgb = read_sim_hex(args.sim_hex)
        save_rgb(out / "sim_output_640x640_rgb.png", sim_rgb)
        report["sim_hex"] = str(args.sim_hex.resolve())
        report["sim_vs_rtl_expected"] = compare(sim_rgb, rtl_rgb)
        report["sim_vs_yolo_fixed"] = compare(sim_rgb, yolo_fixed_rgb)

    (out / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"Saved results to: {out.resolve()}")


if __name__ == "__main__":
    main()
