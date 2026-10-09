from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from ultralytics import YOLO


HERE = Path(__file__).resolve().parent
SOFTWARE_ROOT = HERE.parents[1]
DEFAULT_WEIGHTS = SOFTWARE_ROOT / "models" / "signature_yolo26n" / "best_stripped.pt"
METRIC_NAMES = ("precision", "recall", "map50", "map50_95")
PRED_CONF = 0.25
MATCH_IOU = 0.50


def box_iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    left = max(a[0], b[0])
    top = max(a[1], b[1])
    right = min(a[2], b[2])
    bottom = min(a[3], b[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def load_ground_truth(label_path: Path) -> list[tuple[float, float, float, float]]:
    boxes = []
    for raw_line in label_path.read_text(encoding="utf-8").splitlines():
        if not raw_line.strip():
            continue
        class_id, xc, yc, width, height = raw_line.split()
        if int(class_id) != 0:
            raise ValueError(f"Unexpected class in {label_path}: {class_id}")
        xc, yc, width, height = map(float, (xc, yc, width, height))
        boxes.append((
            (xc - width / 2) * 640,
            (yc - height / 2) * 640,
            (xc + width / 2) * 640,
            (yc + height / 2) * 640,
        ))
    return boxes


def count_detections(
    predictions: list[tuple[tuple[float, float, float, float], float]],
    ground_truth: list[tuple[float, float, float, float]],
) -> dict[str, int]:
    """Greedily match predictions to labels at IoU >= 0.5."""
    matched_labels = set()
    tp = 0
    for predicted_box, _confidence in sorted(predictions, key=lambda p: p[1], reverse=True):
        candidates = (
            (box_iou(predicted_box, true_box), index)
            for index, true_box in enumerate(ground_truth)
            if index not in matched_labels
        )
        best_iou, best_index = max(candidates, default=(0.0, -1))
        if best_iou >= MATCH_IOU:
            matched_labels.add(best_index)
            tp += 1
    return {
        "tp": tp,
        "fp": len(predictions) - tp,
        "fn": len(ground_truth) - tp,
    }


def practical_image_metrics(model: YOLO, image_dir: Path, device: str, batch: int) -> dict:
    """Count per-image detections at the operational confidence threshold."""
    label_dir = image_dir.parent.parent / "labels" / "val"
    per_image = {}
    results = model.predict(
        source=str(image_dir),
        imgsz=640,
        rect=False,
        device=device,
        batch=batch,
        half=False,
        conf=PRED_CONF,
        save=False,
        stream=True,
        verbose=False,
    )
    for result in results:
        image_name = Path(result.path).name
        predicted = []
        for xyxy, confidence, class_id in zip(
            result.boxes.xyxy.cpu().tolist(),
            result.boxes.conf.cpu().tolist(),
            result.boxes.cls.cpu().tolist(),
        ):
            if int(class_id) != 0:
                continue
            predicted.append((tuple(xyxy), float(confidence)))
        true_boxes = load_ground_truth(label_dir / f"{Path(image_name).stem}.txt")
        per_image[image_name] = count_detections(predicted, true_boxes)
    return per_image


def validate_one(
    name: str,
    data_root: Path,
    weights: Path,
    device: str,
    batch: int,
    run_root: Path,
) -> tuple[dict, dict]:
    yaml_path = data_root / name / "data.yaml"
    if not yaml_path.is_file():
        raise FileNotFoundError(f"Missing dataset YAML: {yaml_path}")

    print(f"\n=== Validating {name} ===")
    # Reload identical FP32 weights for each branch.
    model = YOLO(str(weights))
    metrics = model.val(
        data=str(yaml_path),
        imgsz=640,
        rect=False,
        device=device,
        batch=batch,
        workers=0,
        half=False,
        conf=0.001,
        plots=False,
        save_json=False,
        project=str(run_root),
        name=name,
        exist_ok=True,
        verbose=False,
    )

    overall = {
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
        "map50": float(metrics.box.map50),
        "map50_95": float(metrics.box.map),
    }
    image_dir = data_root / name / "images" / "val"
    per_image = practical_image_metrics(model, image_dir, device, batch)

    for key in METRIC_NAMES:
        print(f"{key}: {overall[key]:.6f}")
    print(f"Images with conf={PRED_CONF:.2f} detection counts: {len(per_image)}")
    return overall, per_image


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=HERE / "results" / "datasets")
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--device", default="cpu", help="'cpu' or GPU index such as '0'")
    parser.add_argument("--batch", type=int, default=4)
    args = parser.parse_args()

    data_root = args.data_root.expanduser().resolve()
    weights = args.weights.expanduser().resolve()
    result_root = HERE / "results"

    if not weights.is_file():
        raise FileNotFoundError(f"Missing model weights: {weights}")
    if not data_root.is_dir():
        raise FileNotFoundError(
            f"Missing generated datasets: {data_root}\nRun prepare_val_sets.py first."
        )
    if args.batch < 1:
        raise ValueError("--batch must be at least 1")

    run_root = result_root / "validation_runs"
    run_root.mkdir(parents=True, exist_ok=True)

    standard, standard_images = validate_one(
        "standard", data_root, weights, args.device, args.batch, run_root
    )
    rtl, rtl_images = validate_one(
        "rtl", data_root, weights, args.device, args.batch, run_root
    )

    expected_names = {
        path.name for path in (data_root / "standard" / "images" / "val").glob("*.png")
    }
    if set(standard_images) != expected_names:
        raise RuntimeError("Standard per-image results do not match the generated image list")
    if set(rtl_images) != expected_names:
        raise RuntimeError("RTL per-image results do not match the generated image list")

    rows = []
    more_misses = []
    for image_name in sorted(expected_names):
        a = standard_images[image_name]
        b = rtl_images[image_name]
        extra_fn = b["fn"] - a["fn"]
        rows.append({
            "image": image_name,
            "standard_tp": a["tp"],
            "standard_fp": a["fp"],
            "standard_fn": a["fn"],
            "rtl_tp": b["tp"],
            "rtl_fp": b["fp"],
            "rtl_fn": b["fn"],
            "rtl_extra_fn": extra_fn,
        })
        if extra_fn > 0:
            more_misses.append(image_name)

    summary = {
        "model": str(weights),
        "image_count": len(expected_names),
        "settings": {
            "imgsz": 640,
            "rect": False,
            "half": False,
            "conf": 0.001,
            "per_image_conf": PRED_CONF,
            "per_image_match_iou": MATCH_IOU,
            "device": args.device,
            "batch": args.batch,
        },
        "standard": standard,
        "rtl": rtl,
        "rtl_minus_standard": {key: rtl[key] - standard[key] for key in METRIC_NAMES},
        "images_with_more_rtl_false_negatives": more_misses,
    }
    summary_path = result_root / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    csv_path = result_root / "per_image.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print("\n=== RTL minus standard ===")
    for key in METRIC_NAMES:
        print(f"{key}: {summary['rtl_minus_standard'][key]:+.6f}")
    print(f"Images with more RTL false negatives: {len(more_misses)}")
    print(f"Summary: {summary_path}")
    print(f"Per-image CSV: {csv_path}")


if __name__ == "__main__":
    main()
