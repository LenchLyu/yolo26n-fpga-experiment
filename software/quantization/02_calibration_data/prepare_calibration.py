"""Prepare RTL-equivalent model input images from the training split.

This script only prepares representative inputs. It does not quantize a model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np


HERE = Path(__file__).resolve().parent
CAPTURE_WIDTH = 1280
CAPTURE_HEIGHT = 720
OUTPUT_SIZE = 640
IMAGE_HEIGHT = 360
PAD_TOP = 140
PAD_VALUE = 114
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def read_image(path: Path) -> np.ndarray:
    """Read an image through pathlib so Unicode Windows paths work."""
    encoded = np.frombuffer(path.read_bytes(), dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Cannot decode image: {path}")
    return image


def save_png(path: Path, image: np.ndarray) -> None:
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise RuntimeError(f"Cannot encode image: {path}")
    path.write_bytes(encoded.tobytes())


def make_rtl_input(source_bgr: np.ndarray) -> np.ndarray:
    """Match the current 1280x720 -> 640x640 RTL pixel operation."""
    source_height, source_width = source_bgr.shape[:2]
    if abs(source_width / source_height - CAPTURE_WIDTH / CAPTURE_HEIGHT) > 0.002:
        raise ValueError(f"Expected 16:9 input, got {source_width}x{source_height}")

    # Training images are 1920x1080; this is the shared camera-input proxy.
    capture = cv2.resize(
        source_bgr,
        (CAPTURE_WIDTH, CAPTURE_HEIGHT),
        interpolation=cv2.INTER_LINEAR,
    )
    output = np.full(
        (OUTPUT_SIZE, OUTPUT_SIZE, 3), PAD_VALUE, dtype=np.uint8
    )
    output[PAD_TOP:PAD_TOP + IMAGE_HEIGHT, :, :] = capture[::2, ::2, :]
    return output


def verify_saved_image(path: Path, expected: np.ndarray) -> None:
    actual = read_image(path)
    if actual.shape != (OUTPUT_SIZE, OUTPUT_SIZE, 3) or actual.dtype != np.uint8:
        raise AssertionError(f"Wrong shape or dtype: {path}")
    if not np.array_equal(actual, expected):
        raise AssertionError(f"Saved PNG differs from generated pixels: {path}")
    if not np.all(actual[:PAD_TOP] == PAD_VALUE):
        raise AssertionError(f"Wrong top padding: {path}")
    if not np.all(actual[PAD_TOP + IMAGE_HEIGHT:] == PAD_VALUE):
        raise AssertionError(f"Wrong bottom padding: {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="Dataset root containing images/train",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=HERE / "results",
        help="Output root; must not already exist",
    )
    args = parser.parse_args()

    dataset_root = args.dataset_root.expanduser().resolve()
    source_dir = dataset_root / "images" / "train"
    output_root = args.out.expanduser().resolve()
    image_dir = output_root / "images" / "calib"

    if not source_dir.is_dir():
        raise FileNotFoundError(f"Missing training images: {source_dir}")
    if output_root.exists():
        raise FileExistsError(
            f"Output already exists: {output_root}\n"
            "Choose a new --out directory or remove the old generated results."
        )

    sources = sorted(
        path for path in source_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not sources:
        raise RuntimeError(f"No images found in {source_dir}")
    if len({path.stem for path in sources}) != len(sources):
        raise RuntimeError("Two source images have the same filename stem")

    image_dir.mkdir(parents=True)
    entries = []
    for index, source in enumerate(sources, start=1):
        original = read_image(source)
        result = make_rtl_input(original)
        destination = image_dir / f"{source.stem}.png"
        save_png(destination, result)
        verify_saved_image(destination, result)
        entries.append({
            "source": source.name,
            "source_shape_hw": list(original.shape[:2]),
            "output": destination.name,
            "output_sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
        })
        print(f"[{index}/{len(sources)}] {source.name} -> {destination.name}")

    manifest = {
        "purpose": "Representative input images for future INT8 calibration; no quantization performed",
        "source_dataset": str(dataset_root),
        "source_split": "train",
        "image_count": len(entries),
        "capture_proxy": "OpenCV INTER_LINEAR resize to 1280x720",
        "rtl_preprocess": "even rows and columns, then 140 rows of RGB(114,114,114) at top and bottom",
        "stored_format": "640x640 PNG, 8-bit per channel; image viewers decode RGB, OpenCV decodes BGR",
        "model_input_contract_to_verify": "RGB CHW, float32, pixel values divided by 255 exactly once",
        "files": entries,
    }
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Done. Prepared and checked {len(entries)} calibration images.")
    print(f"Images: {image_dir}")
    print(f"Manifest: {output_root / 'manifest.json'}")


if __name__ == "__main__":
    main()
