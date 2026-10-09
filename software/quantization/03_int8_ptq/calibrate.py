"""Collect Conv activation ranges and create a signed W8A8 QDQ candidate."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import onnx
import onnxruntime as ort
from onnxruntime.quantization import (
    CalibrationDataReader,
    CalibrationMethod,
    QuantFormat,
    QuantType,
    create_calibrator,
    quantize_static,
)


HERE = Path(__file__).resolve().parent
DEFAULT_CALIB = HERE.parent / "02_calibration_data" / "results" / "images" / "calib"
DEFAULT_DATA = HERE.parent / "01_fp32_preprocess_accuracy" / "results" / "datasets" / "rtl" / "data.yaml"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class ImageReader(CalibrationDataReader):
    def __init__(self, directory: Path, input_name: str):
        self.paths = sorted(directory.glob("*.png"))
        if len(self.paths) != 143:
            raise RuntimeError(f"Expected 143 calibration images; found {len(self.paths)}")
        self.input_name = input_name
        self.index = 0

    def get_next(self) -> dict[str, np.ndarray] | None:
        if self.index == len(self.paths):
            return None
        path = self.paths[self.index]
        self.index += 1
        encoded = np.frombuffer(path.read_bytes(), dtype=np.uint8)
        image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if image is None or image.shape != (640, 640, 3) or image.dtype != np.uint8:
            raise ValueError(f"Expected a 640x640 uint8 PNG: {path}")
        if not np.all(image[:140] == 114) or not np.all(image[500:] == 114):
            raise ValueError(f"Gray padding differs from RTL: {path}")
        # Ultralytics model input: BGR image -> RGB CHW -> one /255 normalization.
        tensor = np.ascontiguousarray(image[:, :, ::-1].transpose(2, 0, 1))[None]
        tensor = tensor.astype(np.float32) / 255.0
        if not np.isfinite(tensor).all():
            raise ValueError(f"Nonfinite model input: {path}")
        return {self.input_name: tensor}

    def rewind(self) -> None:
        self.index = 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", type=Path, default=HERE / "results" / "model_source" / "best_stripped.onnx")
    parser.add_argument("--calib-dir", type=Path, default=DEFAULT_CALIB)
    parser.add_argument("--val-data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--out", type=Path, default=HERE / "results")
    args = parser.parse_args()
    model_path = args.onnx.expanduser().resolve()
    calib_dir = args.calib_dir.expanduser().resolve()
    output = args.out.expanduser().resolve()
    data_yaml = args.val_data.expanduser().resolve()
    preflight_path = output / "preflight.json"
    if not model_path.is_file() or not calib_dir.is_dir():
        raise FileNotFoundError("FP32 ONNX or calibration directory is missing")
    if not preflight_path.is_file():
        raise RuntimeError("Passing FP32 ONNX preflight is required before calibration")
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    if not preflight["passed"]:
        raise RuntimeError("Passing FP32 ONNX preflight is required before calibration")
    if preflight.get("files", {}).get("fp32_onnx_sha256") != sha256(model_path):
        raise RuntimeError("FP32 ONNX changed after preflight")
    if not data_yaml.is_file() or preflight["data_fingerprints"]["data_yaml_sha256"] != sha256(data_yaml):
        raise RuntimeError("Validation data YAML changed after preflight")
    val_dir = data_yaml.parent / "images" / "val"
    val_hashes = {p.name: sha256(p) for p in sorted(val_dir.glob("*.png"))}
    if val_hashes != preflight["data_fingerprints"]["validation_images"]:
        raise RuntimeError("Validation images changed after preflight")

    model = onnx.load(str(model_path))
    onnx.checker.check_model(model)
    if len(model.graph.input) != 1:
        raise RuntimeError("Expected exactly one ONNX input")
    input_name = model.graph.input[0].name
    reader = ImageReader(calib_dir, input_name)
    calib_hashes = {p.name: sha256(p) for p in reader.paths}
    if calib_hashes != preflight["data_fingerprints"]["calibration_images"]:
        raise RuntimeError("Calibration images changed after preflight")
    if set(calib_hashes.values()) & set(val_hashes.values()):
        raise RuntimeError("Validation image content appears in the calibration set")
    augmented = output / "calibration_augmented.onnx"
    calibrator = create_calibrator(
        model_path,
        op_types_to_calibrate=["Conv"],
        augmented_model_path=str(augmented),
        calibrate_method=CalibrationMethod.MinMax,
        providers=["CPUExecutionProvider"],
    )
    print(f"Collecting MinMax ranges from {len(reader.paths)} calibration images...")
    calibrator.collect_data(reader)
    if reader.index != 143:
        raise RuntimeError(f"Only {reader.index} calibration images were consumed")
    tensor_data = calibrator.compute_data()
    ranges = {}
    for name, stats in tensor_data.items():
        low, high = stats.range_value
        low_float, high_float = float(np.asarray(low).min()), float(np.asarray(high).max())
        if not np.isfinite([low_float, high_float]).all():
            raise RuntimeError(f"Nonfinite activation range: {name}")
        ranges[name] = {"min": low_float, "max": high_float}
    if not ranges:
        raise RuntimeError("No Conv activation ranges were collected")
    ranges_path = output / "activation_ranges.json"
    ranges_path.write_text(json.dumps({
        "method": "MinMax", "op_types": ["Conv"], "image_count": reader.index,
        "onnx_sha256": sha256(model_path),
        "calibration_files": [{"name": p.name, "sha256": sha256(p)} for p in reader.paths],
        "tensors": ranges,
    }, indent=2), encoding="utf-8")
    with (output / "activation_ranges.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        writer.writerow(("tensor", "min", "max"))
        writer.writerows((name, row["min"], row["max"]) for name, row in sorted(ranges.items()))

    reader.rewind()
    candidate_path = output / "conv_w8a8_qdq.onnx"
    quantize_static(
        model_input=model_path,
        model_output=candidate_path,
        calibration_data_reader=reader,
        quant_format=QuantFormat.QDQ,
        op_types_to_quantize=["Conv"],
        per_channel=True,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
        calibrate_method=CalibrationMethod.MinMax,
        calibration_providers=["CPUExecutionProvider"],
        extra_options={"ActivationSymmetric": True, "WeightSymmetric": True},
    )
    if reader.index != 143:
        raise RuntimeError(f"Quantizer consumed {reader.index} images rather than 143")
    candidate = onnx.load(str(candidate_path))
    onnx.checker.check_model(candidate)
    counts = {}
    for node in candidate.graph.node:
        counts[node.op_type] = counts.get(node.op_type, 0) + 1
    if counts.get("QuantizeLinear", 0) == 0 or counts.get("DequantizeLinear", 0) == 0:
        raise RuntimeError("Candidate has no QDQ nodes")
    if counts.get("Conv") != sum(node.op_type == "Conv" for node in model.graph.node):
        raise RuntimeError("Conv node count changed unexpectedly")
    (output / "quantization_manifest.json").write_text(json.dumps({
        "candidate": str(candidate_path), "candidate_sha256": sha256(candidate_path),
        "source_onnx_sha256": sha256(model_path), "calibration_count": 143,
        "activation_type": "signed int8 symmetric per tensor, zero point 0",
        "weight_type": "signed int8 symmetric per output channel, zero point 0",
        "quantized_ops": ["Conv"],
        "float_ops": "all non-Conv operators, including attention, SiLU, and detection postprocess",
        "operator_counts": counts,
        "versions": {"onnx": onnx.__version__, "onnxruntime": ort.__version__},
    }, indent=2), encoding="utf-8")
    print("Collected activation tensor ranges:", len(ranges))
    print("Candidate operator counts:", counts)
    print("Candidate:", candidate_path)


if __name__ == "__main__":
    main()
