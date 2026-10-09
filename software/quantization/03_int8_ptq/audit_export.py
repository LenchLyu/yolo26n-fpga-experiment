"""Export the frozen YOLO26n checkpoint and audit its inference ONNX graph."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import shutil
from pathlib import Path

import onnx
import onnxruntime as ort
import torch
import ultralytics
from onnx import numpy_helper, shape_inference
from ultralytics import YOLO


HERE = Path(__file__).resolve().parent
SOFTWARE = HERE.parents[1]
DEFAULT_WEIGHTS = SOFTWARE / "models" / "signature_yolo26n" / "best_stripped.pt"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_shapes(model: onnx.ModelProto) -> dict[str, list[int | str]]:
    found = {}
    for info in list(model.graph.input) + list(model.graph.output) + list(model.graph.value_info):
        dims = []
        for dim in info.type.tensor_type.shape.dim:
            dims.append(dim.dim_value if dim.HasField("dim_value") else dim.dim_param or "?")
        found[info.name] = dims
    for initializer in model.graph.initializer:
        found[initializer.name] = list(initializer.dims)
    return found


def audit_graph(model: onnx.ModelProto) -> dict:
    inferred = shape_inference.infer_shapes(model)
    shapes = tensor_shapes(inferred)
    initializers = {item.name: item for item in inferred.graph.initializer}
    nodes = []
    conv_styles = collections.Counter()

    for index, node in enumerate(inferred.graph.node):
        item = {
            "index": index,
            "name": node.name,
            "op": node.op_type,
            "inputs": [{"name": name, "shape": shapes.get(name)} for name in node.input],
            "outputs": [{"name": name, "shape": shapes.get(name)} for name in node.output],
        }
        if node.op_type == "Conv":
            attributes = {attr.name: onnx.helper.get_attribute_value(attr) for attr in node.attribute}
            group = int(attributes.get("group", 1))
            kernel = list(initializers[node.input[1]].dims) if node.input[1] in initializers else []
            style = "depthwise" if group > 1 and kernel and kernel[1] == 1 else "regular"
            if kernel and kernel[-2:] == [1, 1]:
                style = "pointwise_1x1"
            conv_styles[style] += 1
            item.update({
                "group": group,
                "weight_shape": kernel,
                "kernel_style": style,
                "mac_terms_per_output": int(kernel[1] * kernel[2] * kernel[3]) if kernel else None,
            })
        nodes.append(item)

    return {
        "inputs": [{"name": item.name, "shape": shapes.get(item.name)} for item in inferred.graph.input],
        "outputs": [{"name": item.name, "shape": shapes.get(item.name)} for item in inferred.graph.output],
        "operator_counts": dict(collections.Counter(node.op_type for node in inferred.graph.node)),
        "convolution_styles": dict(conv_styles),
        "float_operator_families_for_this_stage": [
            "Sigmoid/Mul (SiLU)", "MatMul/Softmax (attention)",
            "Resize", "Concat", "pooling", "detection decode/TopK",
        ],
        "nodes": nodes,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--out", type=Path, default=HERE / "results")
    args = parser.parse_args()
    weights = args.weights.expanduser().resolve()
    output = args.out.expanduser().resolve()
    if not weights.is_file():
        raise FileNotFoundError(weights)
    model_dir = output / "model_source"
    model_dir.mkdir(parents=True, exist_ok=True)
    copy = model_dir / "best_stripped.pt"
    if copy.exists():
        if sha256(copy) != sha256(weights):
            raise RuntimeError(f"Existing checkpoint copy differs: {copy}")
    else:
        shutil.copy2(weights, copy)
    onnx_path = model_dir / "best_stripped.onnx"
    if not onnx_path.exists():
        exported = Path(YOLO(str(copy)).export(
            format="onnx", imgsz=640, batch=1, dynamic=False,
            half=False, nms=False, simplify=False, opset=17, device="cpu",
        )).resolve()
        if exported != onnx_path:
            raise RuntimeError(f"Unexpected export path: {exported}")

    model = onnx.load(str(onnx_path))
    onnx.checker.check_model(model)
    report = audit_graph(model)
    if len(report["inputs"]) != 1 or report["inputs"][0]["shape"] != [1, 3, 640, 640]:
        raise RuntimeError(f"Unexpected ONNX input: {report['inputs']}")
    if len(report["outputs"]) != 1 or report["outputs"][0]["shape"] != [1, 300, 6]:
        raise RuntimeError(f"Unexpected end-to-end detection output: {report['outputs']}")
    if not report["operator_counts"].get("Conv"):
        raise RuntimeError("Exported ONNX graph has no convolution operators")
    report["files"] = {
        "original_weights": str(weights),
        "original_weights_sha256": sha256(weights),
        "checkpoint_copy_sha256": sha256(copy),
        "onnx_path": str(onnx_path),
        "onnx_sha256": sha256(onnx_path),
    }
    report["versions"] = {
        "torch": torch.__version__, "ultralytics": ultralytics.__version__,
        "onnx": onnx.__version__, "onnxruntime": ort.__version__,
    }
    report_path = output / "operator_manifest.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print("ONNX:", onnx_path)
    print("Operator counts:", report["operator_counts"])
    print("Convolution styles:", report["convolution_styles"])
    print("Manifest:", report_path)


if __name__ == "__main__":
    main()
