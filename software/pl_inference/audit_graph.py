"""Make the first, conservative PL operator and tensor inventory from FP32 ONNX.

This is an audit, not an executable accelerator instruction stream: a tensor
without a known shape must be resolved before final DDR addresses are assigned.
"""

from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, shape_inference


HERE = Path(__file__).resolve().parent
DEFAULT_MODEL = HERE.parent / "quantization" / "03_int8_ptq" / "results" / "model_source" / "best_stripped.onnx"

# Every run-time operation must have a PL implementation. Compile-time constants
# may be embedded in weights/instructions but cannot depend on the input image.
PL_UNITS = {
    "Conv": "mac",
    "MatMul": "mac",
    "Sigmoid": "nonlinear",
    "Softmax": "nonlinear",
    "Mul": "elementwise",
    "Add": "elementwise",
    "Sub": "elementwise",
    "Div": "elementwise",
    "Mod": "elementwise",
    "MaxPool": "pool",
    "ReduceMax": "reduction",
    "TopK": "reduction",
    "Resize": "resample",
    "Concat": "tensor_layout",
    "Split": "tensor_layout",
    "Reshape": "tensor_layout",
    "Transpose": "tensor_layout",
    "Slice": "tensor_layout",
    "Unsqueeze": "tensor_layout",
    "Expand": "tensor_layout",
    "Tile": "tensor_layout",
    "Flatten": "tensor_layout",
    "Shape": "shape_index",
    "Gather": "shape_index",
    "GatherElements": "shape_index",
    "Cast": "shape_index",
    "ConstantOfShape": "shape_index",
    "Constant": "compile_time",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def type_inventory(model: onnx.ModelProto) -> dict[str, dict]:
    inventory = {}
    for entry in (*model.graph.input, *model.graph.output, *model.graph.value_info):
        tensor_type = entry.type.tensor_type
        if not tensor_type.HasField("shape"):
            continue
        shape = []
        for dim in tensor_type.shape.dim:
            shape.append(int(dim.dim_value) if dim.HasField("dim_value")
                         else dim.dim_param or "?")
        try:
            dtype = str(np.dtype(helper.tensor_dtype_to_np_dtype(tensor_type.elem_type)))
        except (TypeError, ValueError, KeyError):
            dtype = "unknown"
        inventory[entry.name] = {"shape": shape, "dtype": dtype}
    for entry in model.graph.initializer:
        inventory[entry.name] = {
            "shape": list(entry.dims),
            "dtype": str(np.dtype(helper.tensor_dtype_to_np_dtype(entry.data_type))),
        }
    return inventory


def tensor_size(metadata: dict | None) -> int | None:
    if not metadata or metadata["dtype"] == "unknown":
        return None
    shape = metadata["shape"]
    if any(not isinstance(dim, int) or dim <= 0 for dim in shape):
        return None
    return int(np.prod(shape, dtype=np.int64)) * np.dtype(metadata["dtype"]).itemsize


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", type=Path, default=DEFAULT_MODEL,
                        help="FP32 inference ONNX from 03_int8_ptq/audit_export.py")
    parser.add_argument("--out", type=Path, default=HERE / "results")
    args = parser.parse_args()
    source = args.onnx.expanduser().resolve()
    output = args.out.expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    output.mkdir(parents=True, exist_ok=True)
    model = onnx.load(str(source))
    onnx.checker.check_model(model)
    inferred = shape_inference.infer_shapes(model)
    types = type_inventory(inferred)
    counts = collections.Counter(node.op_type for node in inferred.graph.node)
    unknown_ops = sorted(set(counts) - set(PL_UNITS))
    if unknown_ops:
        raise RuntimeError(f"Unmapped ONNX operators: {unknown_ops}")

    consumers: dict[str, list[int]] = collections.defaultdict(list)
    for index, node in enumerate(inferred.graph.node):
        for name in node.input:
            if name:
                consumers[name].append(index)
    terminal_index = len(inferred.graph.node)
    for entry in inferred.graph.output:
        consumers[entry.name].append(terminal_index)

    rows = []
    unresolved = []
    tensors = []
    for index, node in enumerate(inferred.graph.node):
        unit = PL_UNITS[node.op_type]
        item = {
            "index": index,
            "name": node.name,
            "op": node.op_type,
            "execution": "compile_time" if node.op_type == "Constant" else "PL",
            "unit": unit,
            "inputs": list(node.input),
            "outputs": list(node.output),
        }
        rows.append(item)
        for name in node.output:
            if not name:
                continue
            metadata = types.get(name)
            byte_count = tensor_size(metadata)
            last_use = max(consumers[name], default=index)
            tensor = {
                "tensor": name, "producer_index": index,
                "last_consumer_index": last_use,
                "shape": metadata["shape"] if metadata else None,
                "dtype": metadata["dtype"] if metadata else "unknown",
                "fp32_graph_bytes": byte_count,
                "unit": unit,
            }
            tensors.append(tensor)
            if byte_count is None:
                unresolved.append(tensor)

    with (output / "operator_coverage.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        writer.writerow(("op", "node_count", "first_pass_pl_unit"))
        writer.writerows((op, count, PL_UNITS[op]) for op, count in sorted(counts.items()))
    with (output / "unresolved_tensors.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        writer.writerow(("tensor", "producer_index", "shape", "dtype"))
        writer.writerows((item["tensor"], item["producer_index"],
                          json.dumps(item["shape"]), item["dtype"]) for item in unresolved)
    report = {
        "source_onnx": str(source), "source_onnx_sha256": sha256(source),
        "purpose": "first-pass coverage and tensor liveness, not final DDR addresses",
        "fixed_input": [{"name": x.name, **types[x.name]} for x in inferred.graph.input],
        "fixed_output": [{"name": x.name, **types[x.name]} for x in inferred.graph.output],
        "operator_counts": dict(counts),
        "pl_operator_types": sorted(op for op in counts if op != "Constant"),
        "compile_time_operator_types": ["Constant"],
        "unresolved_tensor_count": len(unresolved),
        "nodes": rows,
        "tensors": tensors,
    }
    (output / "first_pass_graph.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Mapped {len(rows)} ONNX nodes and {len(counts)} operator types.")
    print(f"Unresolved tensor sizes: {len(unresolved)}. Resolve these before assigning DDR addresses.")
    print("Coverage:", output / "operator_coverage.csv")
    print("Unresolved:", output / "unresolved_tensors.csv")
    print("Graph:", output / "first_pass_graph.json")


if __name__ == "__main__":
    main()
