from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from ultralytics.data.augment import LetterBox


HERE = Path(__file__).resolve().parent
CAPTURE_W, CAPTURE_H = 1280, 720
OUT_W, OUT_H = 640, 640
IMAGE_H = 360
PAD_TOP = 140
PAD_VALUE = 114
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def read_image(path: Path) -> np.ndarray:
    """Read an image even when its Windows path contains Chinese characters."""
    encoded = np.frombuffer(path.read_bytes(), dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Cannot read image: {path}")
    return image


def write_png(path: Path, image: np.ndarray) -> None:
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise RuntimeError(f"Cannot encode PNG: {path}")
    path.write_bytes(encoded.tobytes())


def convert_labels(source: Path) -> str:
    """Transform normalized YOLO labels to the 640x640 letterboxed frame."""
    if not source.is_file():
        raise FileNotFoundError(f"Missing label: {source}")

    output_lines = []
    for line_number, raw_line in enumerate(source.read_text(encoding="utf-8").splitlines(), 1):
        if not raw_line.strip():
            continue
        fields = raw_line.split()
        if len(fields) != 5:
            raise ValueError(f"{source}:{line_number}: expected 5 fields")

        class_id = int(fields[0])
        x_center, y_center, width, height = map(float, fields[1:])
        if class_id != 0:
            raise ValueError(f"{source}:{line_number}: expected signature class 0")

        # Both resizes preserve the 16:9 image area, so normalized x and width stay the same.
        new_y_center = (PAD_TOP + y_center * IMAGE_H) / OUT_H
        new_height = height * IMAGE_H / OUT_H
        output_lines.append(
            f"{class_id} {x_center:.8f} {new_y_center:.8f} "
            f"{width:.8f} {new_height:.8f}"
        )

    return "\n".join(output_lines) + ("\n" if output_lines else "")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-root", type=Path, required=True,
        help="Source dataset root containing images/val and labels/val",
    )
    parser.add_argument(
        "--out", type=Path, default=HERE / "results" / "datasets",
        help="Directory for the two generated validation datasets",
    )
    args = parser.parse_args()

    dataset_root = args.dataset_root.expanduser().resolve()
    output_root = args.out.expanduser().resolve()
    source_images = dataset_root / "images" / "val"
    source_labels = dataset_root / "labels" / "val"

    if not source_images.is_dir() or not source_labels.is_dir():
        raise FileNotFoundError("Expected images/val and labels/val under --dataset-root")

    images = sorted(
        path for path in source_images.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not images:
        raise RuntimeError(f"No validation images found in {source_images}")
    stems = [path.stem for path in images]
    if len(stems) != len(set(stems)):
        raise RuntimeError("Two source images have the same filename stem")

    # Fail instead of mixing a prior experiment's generated images into this one.
    if output_root.exists():
        raise FileExistsError(
            f"Output already exists: {output_root}\n"
            "Use another --out directory or remove the previous generated results."
        )

    for variant in ("standard", "rtl"):
        (output_root / variant / "images" / "val").mkdir(parents=True, exist_ok=False)
        (output_root / variant / "labels" / "val").mkdir(parents=True, exist_ok=False)
        # The installed dataset parser requires train and val; this experiment only calls val().
        (output_root / variant / "data.yaml").write_text(
            "train: images/val\nval: images/val\nnames:\n  0: signature\n",
            encoding="utf-8",
        )

    letterbox = LetterBox(new_shape=(OUT_H, OUT_W), auto=False)

    for index, source_path in enumerate(images, 1):
        original = read_image(source_path)
        height, width = original.shape[:2]
        if abs(width / height - CAPTURE_W / CAPTURE_H) > 0.002:
            raise ValueError(f"Expected a 16:9 image, got {width}x{height}: {source_path}")

        # Both branches start from the same simulated 1280x720 camera frame.
        capture = cv2.resize(original, (CAPTURE_W, CAPTURE_H), interpolation=cv2.INTER_LINEAR)
        standard = letterbox(image=capture.copy())
        if standard.shape != (OUT_H, OUT_W, 3):
            raise RuntimeError(f"Unexpected standard image shape: {standard.shape}")

        rtl = np.full((OUT_H, OUT_W, 3), PAD_VALUE, dtype=np.uint8)
        rtl[PAD_TOP:PAD_TOP + IMAGE_H, :, :] = capture[::2, ::2, :]
        label_text = convert_labels(source_labels / f"{source_path.stem}.txt")

        for variant, output_image in (("standard", standard), ("rtl", rtl)):
            write_png(
                output_root / variant / "images" / "val" / f"{source_path.stem}.png",
                output_image,
            )
            (output_root / variant / "labels" / "val" / f"{source_path.stem}.txt").write_text(
                label_text, encoding="utf-8"
            )
        print(f"[{index}/{len(images)}] {source_path.name}")

    manifest = {
        "source_dataset": str(dataset_root),
        "image_count": len(images),
        "capture_proxy": "resize source to 1280x720 with cv2.INTER_LINEAR",
        "standard": "Ultralytics LetterBox 640x640, auto=False",
        "rtl": "capture[::2, ::2] plus 140 gray rows above and below",
        "pad_rgb": [PAD_VALUE, PAD_VALUE, PAD_VALUE],
    }
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Done. Generated {len(images)} images in each dataset.")
    print(f"Output: {output_root}")


if __name__ == "__main__":
    main()
