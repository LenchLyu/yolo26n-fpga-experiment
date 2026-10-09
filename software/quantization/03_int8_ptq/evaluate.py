"""Compare the original checkpoint, FP32 ONNX, and convolution-only QDQ ONNX."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from ultralytics import YOLO


HERE = Path(__file__).resolve().parent
SOFTWARE = HERE.parents[1]
DEFAULT_WEIGHTS = SOFTWARE / "models" / "signature_yolo26n" / "best_stripped.pt"
DEFAULT_DATA = HERE.parent / "01_fp32_preprocess_accuracy" / "results" / "datasets" / "rtl" / "data.yaml"
DEFAULT_CALIB = HERE.parent / "02_calibration_data" / "results" / "images" / "calib"
IMAGE_CONF = 0.25
MATCH_IOU = 0.50


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def box_iou(a: list[float], b: list[float]) -> float:
    left, top = max(a[0], b[0]), max(a[1], b[1])
    right, bottom = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - intersection
    return intersection / union if union else 0.0


def ground_truth(path: Path) -> list[list[float]]:
    boxes = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        class_id, xc, yc, width, height = raw.split()
        if int(class_id) != 0:
            raise ValueError(f"Unexpected class in {path}")
        xc, yc, width, height = map(float, (xc, yc, width, height))
        boxes.append([
            (xc - width / 2) * 640, (yc - height / 2) * 640,
            (xc + width / 2) * 640, (yc + height / 2) * 640,
        ])
    return boxes


def practical_counts(model: YOLO, image_dir: Path) -> dict:
    label_dir = image_dir.parent.parent / "labels" / "val"
    output = {}
    for result in model.predict(
        source=str(image_dir), imgsz=640, rect=False, batch=1,
        device="cpu", half=False, conf=IMAGE_CONF, save=False,
        stream=True, verbose=False,
    ):
        name = Path(result.path).name
        true_boxes = ground_truth(label_dir / f"{Path(name).stem}.txt")
        predicted = sorted(
            [
                (float(conf), list(xyxy))
                for xyxy, conf, cls in zip(
                    result.boxes.xyxy.cpu().tolist(),
                    result.boxes.conf.cpu().tolist(),
                    result.boxes.cls.cpu().tolist(),
                )
                if int(cls) == 0
            ],
            reverse=True,
        )
        matched = set()
        tp = 0
        for _, box in predicted:
            candidates = (
                (box_iou(box, target), index)
                for index, target in enumerate(true_boxes)
                if index not in matched
            )
            best_iou, best_index = max(candidates, default=(0.0, -1))
            if best_iou >= MATCH_IOU:
                matched.add(best_index)
                tp += 1
        output[name] = {
            "tp": tp,
            "fp": len(predicted) - tp,
            "fn": len(true_boxes) - tp,
        }
    return output


def assess(model_path: Path, data_yaml: Path, image_dir: Path, run_dir: Path) -> dict:
    model = YOLO(str(model_path), task="detect")
    metrics = model.val(
        data=str(data_yaml), imgsz=640, rect=False, batch=1,
        device="cpu", workers=0, half=False, conf=0.001,
        plots=False, save_json=False, project=str(run_dir),
        name=model_path.stem, exist_ok=True, verbose=False,
    )
    per_image = practical_counts(model, image_dir)
    expected = {path.name for path in image_dir.glob("*.png")}
    if len(expected) != 35 or set(per_image) != expected:
        raise RuntimeError("Expected exactly the same 35 validation images in every assessment")
    return {
        "model": str(model_path),
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
        "map50": float(metrics.box.map50),
        "map50_95": float(metrics.box.map),
        "per_image": per_image,
        "missed_images": sorted(name for name, row in per_image.items() if row["fn"] > 0),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("preflight", "candidate"), required=True)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--val-data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--calib-dir", type=Path, default=DEFAULT_CALIB)
    parser.add_argument("--out", type=Path, default=HERE / "results")
    args = parser.parse_args()
    output = args.out.expanduser().resolve()
    data_yaml = args.val_data.expanduser().resolve()
    weights = args.weights.expanduser().resolve()
    calibration = args.calib_dir.expanduser().resolve()
    for path in (data_yaml, weights):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not calibration.is_dir():
        raise FileNotFoundError(calibration)
    image_dir = data_yaml.parent / "images" / "val"
    if not image_dir.is_dir():
        raise FileNotFoundError(image_dir)
    calib_paths = sorted(calibration.glob("*.png"))
    val_paths = sorted(image_dir.glob("*.png"))
    calib_hashes = {p.name: sha256(p) for p in calib_paths}
    val_hashes = {p.name: sha256(p) for p in val_paths}
    if (len(calib_paths) != 143 or len(val_paths) != 35
            or set(calib_hashes) & set(val_hashes)
            or set(calib_hashes.values()) & set(val_hashes.values())):
        raise RuntimeError("Calibration and validation split sizes/overlap are incorrect")
    data_fingerprints = {
        "calibration_images": calib_hashes,
        "validation_images": val_hashes,
        "validation_labels": {p.name: sha256(p) for p in sorted(
            (image_dir.parent.parent / "labels" / "val").glob("*.txt"))},
        "data_yaml_sha256": sha256(data_yaml),
    }

    run_dir = output / "validation_runs"
    run_dir.mkdir(parents=True, exist_ok=True)
    fp32_onnx = output / "model_source" / "best_stripped.onnx"
    candidate_onnx = output / "conv_w8a8_qdq.onnx"

    if args.stage == "preflight":
        if not fp32_onnx.is_file():
            raise FileNotFoundError(fp32_onnx)
        original = assess(weights, data_yaml, image_dir, run_dir / "pt")
        exported = assess(fp32_onnx, data_yaml, image_dir, run_dir / "fp32_onnx")
        new_misses = sorted(set(exported["missed_images"]) - set(original["missed_images"]))
        difference = exported["map50_95"] - original["map50_95"]
        passed = abs(difference) <= 0.005 and not new_misses
        report = {
            "stage": "preflight", "passed": passed,
            "map50_95_delta": difference, "new_missed_images": new_misses,
            "files": {"original_weights_sha256": sha256(weights),
                      "fp32_onnx_sha256": sha256(fp32_onnx)},
            "data_fingerprints": data_fingerprints,
            "settings": {"imgsz": 640, "rect": False, "batch": 1, "half": False,
                         "mAP_conf": 0.001, "per_image_conf": IMAGE_CONF,
                         "per_image_match_iou": MATCH_IOU},
            "original_pt": original, "fp32_onnx": exported,
        }
        path = output / "preflight.json"
    else:
        preflight_path = output / "preflight.json"
        if not preflight_path.is_file():
            raise FileNotFoundError("Run --stage preflight first")
        preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
        if not preflight["passed"]:
            raise RuntimeError("FP32 ONNX preflight failed; do not evaluate a candidate")
        if preflight.get("files") != {"original_weights_sha256": sha256(weights),
                                       "fp32_onnx_sha256": sha256(fp32_onnx)}:
            raise RuntimeError("Model files changed after FP32 ONNX preflight")
        if preflight.get("data_fingerprints") != data_fingerprints:
            raise RuntimeError("Calibration or validation data changed after preflight")
        if not candidate_onnx.is_file():
            raise FileNotFoundError(candidate_onnx)
        quant_manifest = json.loads((output / "quantization_manifest.json").read_text(encoding="utf-8"))
        if (quant_manifest["source_onnx_sha256"] != sha256(fp32_onnx)
                or quant_manifest["candidate_sha256"] != sha256(candidate_onnx)):
            raise RuntimeError("Candidate or source ONNX changed after quantization")
        candidate = assess(candidate_onnx, data_yaml, image_dir, run_dir / "candidate")
        original = preflight["original_pt"]
        new_misses = sorted(set(candidate["missed_images"]) - set(original["missed_images"]))
        difference = candidate["map50_95"] - original["map50_95"]
        passed = difference >= -0.03 and len(new_misses) <= 1
        report = {
            "stage": "candidate", "passed": passed,
            "map50_95_delta": difference, "new_missed_images": new_misses,
            "files": {**preflight["files"], "candidate_onnx_sha256": sha256(candidate_onnx)},
            "data_fingerprints": data_fingerprints,
            "settings": preflight["settings"],
            "original_pt": original, "candidate_qdq": candidate,
        }
        path = output / "candidate_eval.json"

    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print("Report:", path)
    print("mAP50-95 delta:", report["map50_95_delta"])
    print("New missed images:", report["new_missed_images"])
    print("Gate passed:", report["passed"])
    if not report["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
