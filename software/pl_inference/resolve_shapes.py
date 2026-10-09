"""Observe detector-head tensor shapes on calibration inputs (never validation images)."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import onnx
import onnxruntime as ort
from onnx import shape_inference


HERE = Path(__file__).resolve().parent
DEFAULT_CALIB = HERE.parent / "quantization" / "02_calibration_data" / "results" / "images" / "calib"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def image_tensor(path: Path) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None or bgr.shape != (640, 640, 3) or bgr.dtype != np.uint8:
        raise ValueError(f"Expected 640x640 RGB calibration PNG: {path}")
    rgb_chw = np.ascontiguousarray(bgr[:, :, ::-1].transpose(2, 0, 1))[None]
    return rgb_chw.astype(np.float32) / 255.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, default=HERE / "results" / "first_pass_graph.json")
    parser.add_argument("--calib-dir", type=Path, default=DEFAULT_CALIB)
    parser.add_argument("--out", type=Path, default=HERE / "results")
    parser.add_argument("--sample-count", type=int, default=3)
    args = parser.parse_args()
    audit = json.loads(args.audit.expanduser().resolve().read_text(encoding="utf-8"))
    source = Path(audit["source_onnx"])
    if sha256(source) != audit["source_onnx_sha256"]:
        raise RuntimeError("ONNX differs from the operator audit")
    names = [item["tensor"] for item in audit["tensors"]
             if item["fp32_graph_bytes"] is None]
    if not names:
        print("All tensor shapes are already known.")
        return
    calib_dir = args.calib_dir.expanduser().resolve()
    paths = sorted(calib_dir.glob("*.png"))
    if args.sample_count < 2 or len(paths) != 143:
        raise ValueError("Use at least two of the 143 calibration PNGs")
    paths = paths[:args.sample_count]

    model = shape_inference.infer_shapes(onnx.load(str(source)))
    infos = {entry.name: entry for entry in (*model.graph.value_info, *model.graph.output)}
    for name in names:
        if name not in infos:
            raise RuntimeError(f"Cannot add tensor as ONNX output: {name}")
        model.graph.output.append(copy.deepcopy(infos[name]))
    onnx.checker.check_model(model)
    session_options = ort.SessionOptions()
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    session = ort.InferenceSession(model.SerializeToString(), sess_options=session_options,
                                   providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    if len(session.get_inputs()) != 1:
        raise RuntimeError("Expected one model input")
    baseline = None
    observed = []
    for path in paths:
        outputs = session.run(names, {input_name: image_tensor(path)})
        current = {name: {"shape": list(array.shape), "dtype": str(array.dtype),
                          "observed_bytes": int(array.nbytes)}
                   for name, array in zip(names, outputs)}
        if baseline is not None and current != baseline:
            raise RuntimeError(f"Tensor shape/type depends on image content: {path}")
        baseline = current
        observed.append({"name": path.name, "sha256": sha256(path)})

    output = args.out.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    result = {
        "source_onnx_sha256": audit["source_onnx_sha256"],
        "method": "Observed with ONNX Runtime on fixed-size calibration images; not a proof for arbitrary inputs",
        "sample_images": observed,
        "resolved_tensor_count": len(names),
        "tensors": baseline,
    }
    destination = output / "observed_shapes.json"
    destination.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Observed matching shapes for {len(names)} tensors on {len(paths)} calibration images.")
    print("Report:", destination)


if __name__ == "__main__":
    main()
