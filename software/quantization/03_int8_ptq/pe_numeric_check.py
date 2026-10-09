"""Extract the Conv QDQ parameters and verify the first PE integer contract."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper


HERE = Path(__file__).resolve().parent
I32_MAX = np.iinfo(np.int32).max
I32_MIN = np.iinfo(np.int32).min


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_dq(name: str, producers: dict) -> onnx.NodeProto:
    node = producers.get(name)
    if node is None or node.op_type != "DequantizeLinear":
        raise ValueError(f"Conv input has no DequantizeLinear producer: {name}")
    if len(node.input) != 3:
        raise ValueError(f"Expected explicit QDQ scale and zero point: {name}")
    return node


def initial(name: str, arrays: dict[str, np.ndarray]) -> np.ndarray:
    if name not in arrays:
        raise ValueError(f"Missing QDQ initializer: {name}")
    return arrays[name]


def positive_scale(value: np.ndarray, expected_shape: tuple, label: str) -> np.ndarray:
    if value.shape != expected_shape or not np.issubdtype(value.dtype, np.floating):
        raise ValueError(f"Unexpected {label} scale shape/type: {value.shape}, {value.dtype}")
    if not np.all(np.isfinite(value)) or np.any(value <= 0):
        raise ValueError(f"Invalid {label} scale")
    return value.astype(np.float64)


def zero_point(value: np.ndarray, expected_shape: tuple, dtype: np.dtype, label: str) -> None:
    if value.shape != expected_shape or value.dtype != dtype or np.any(value != 0):
        raise ValueError(f"Expected signed symmetric zero point 0 for {label}; "
                         f"got shape={value.shape}, dtype={value.dtype}, "
                         f"min={value.min()}, max={value.max()}")


def requantize(accumulator: int, multiplier: float) -> int:
    """ONNX QuantizeLinear convention: nearest, ties to even, then S8 saturation."""
    return int(np.clip(np.rint(float(accumulator) * multiplier), -128, 127))


def arithmetic_examples(layers: list[dict], weights: dict[str, np.ndarray]) -> list[dict]:
    checks = []
    for style in ("regular", "pointwise_1x1", "depthwise"):
        layer = next((row for row in layers if row["kernel_style"] == style), None)
        if layer is None:
            raise ValueError(f"No {style} example in candidate ONNX")
        kernel = weights[layer["weight_initializer"]][0].astype(np.int64).reshape(-1)
        activation = ((np.arange(kernel.size, dtype=np.int64) * 37 + 11) % 256) - 128
        bias = int(layer["bias_int32"][0])
        scalar = bias + sum(int(a) * int(w) for a, w in zip(activation, kernel))
        vectorized = int(np.dot(activation, kernel)) + bias
        if scalar != vectorized:
            raise AssertionError(f"MAC disagreement for {style}")
        multiplier = (layer["input_scale"] * layer["weight_scales"][0]
                      / layer["output_scale"])
        pe_output = requantize(scalar, multiplier)
        reference = max(-128, min(127, round(vectorized * multiplier)))
        if pe_output != reference:
            raise AssertionError(f"Requantization disagreement for {style}")
        checks.append({
            "kernel_style": style, "conv": layer["name"], "output_channel": 0,
            "input_pattern": "((index * 37 + 11) % 256) - 128",
            "mac_terms": int(kernel.size), "bias_int32": bias,
            "accumulator_int32": scalar, "requant_multiplier": multiplier,
            "output_int8": pe_output, "passed": True,
        })
    rounding = {str(value): requantize(value, 0.5) for value in (-259, -257, -5, -3, -1, 1, 3, 5, 255, 257)}
    expected = {"-259": -128, "-257": -128, "-5": -2, "-3": -2, "-1": 0,
                "1": 0, "3": 2, "5": 2, "255": 127, "257": 127}
    if rounding != expected:
        raise AssertionError(f"Ties-to-even or saturation is incorrect: {rounding}")
    checks.append({"kernel_style": "rounding_and_saturation", "multiplier": 0.5,
                   "cases": rounding, "passed": True})
    return checks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", type=Path, default=HERE / "results" / "conv_w8a8_qdq.onnx")
    parser.add_argument("--out", type=Path, default=HERE / "results")
    args = parser.parse_args()
    candidate_path = args.onnx.expanduser().resolve()
    output = args.out.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    model = onnx.load(str(candidate_path))
    onnx.checker.check_model(model)
    arrays = {item.name: numpy_helper.to_array(item) for item in model.graph.initializer}
    producers = {name: node for node in model.graph.node for name in node.output}
    consumers: dict[str, list[onnx.NodeProto]] = {}
    for node in model.graph.node:
        for name in node.input:
            consumers.setdefault(name, []).append(node)

    layers = []
    weights = {}
    for conv in (node for node in model.graph.node if node.op_type == "Conv"):
        if len(conv.input) != 3:
            raise ValueError(f"Expected a fused Conv with bias: {conv.name}")
        x_dq, w_dq, b_dq = (require_dq(name, producers) for name in conv.input)
        x_scale = positive_scale(initial(x_dq.input[1], arrays), (), "activation")
        zero_point(initial(x_dq.input[2], arrays), (), np.dtype("int8"), "activation")
        w = initial(w_dq.input[0], arrays)
        if w.dtype != np.int8 or w.ndim != 4:
            raise ValueError(f"Expected S8 Conv weight: {conv.name}")
        channels = w.shape[0]
        w_scales = positive_scale(initial(w_dq.input[1], arrays), (channels,), "weight")
        zero_point(initial(w_dq.input[2], arrays), (channels,), np.dtype("int8"), "weight")
        b = initial(b_dq.input[0], arrays)
        if b.dtype != np.int32 or b.shape != (channels,):
            raise ValueError(f"Expected INT32 bias: {conv.name}")
        b_scales = positive_scale(initial(b_dq.input[1], arrays), (channels,), "bias")
        bias_zp = initial(b_dq.input[2], arrays)
        if bias_zp.shape not in ((), (channels,)):
            raise ValueError(f"Unexpected bias zero point shape: {conv.name}")
        zero_point(bias_zp, bias_zp.shape, np.dtype("int32"), "bias")
        if not np.allclose(b_scales, float(x_scale) * w_scales, rtol=1e-5, atol=0):
            raise ValueError(f"Bias scale differs from input_scale * weight_scale: {conv.name}")
        successors = [node for node in consumers.get(conv.output[0], []) if node.op_type == "QuantizeLinear"]
        if len(successors) != 1:
            raise ValueError(f"Expected one Conv output QuantizeLinear: {conv.name}")
        out_q = successors[0]
        out_scale = positive_scale(initial(out_q.input[1], arrays), (), "output")
        zero_point(initial(out_q.input[2], arrays), (), np.dtype("int8"), "output")
        group = next((int(helper.get_attribute_value(attr)) for attr in conv.attribute
                      if attr.name == "group"), 1)
        style = "depthwise" if group > 1 and w.shape[1] == 1 else (
            "pointwise_1x1" if w.shape[2:] == (1, 1) else "regular")
        mac_terms = int(np.prod(w.shape[1:]))
        bound_by_channel = 128 * np.abs(w.astype(np.int64)).reshape(channels, -1).sum(axis=1)
        bound_by_channel += np.abs(b.astype(np.int64))
        largest = int(bound_by_channel.max())
        if largest > I32_MAX or int(b.min()) < I32_MIN or int(b.max()) > I32_MAX:
            raise OverflowError(f"INT32 conservative bound exceeded: {conv.name}, {largest}")
        multipliers = float(x_scale) * w_scales / float(out_scale)
        layer = {
            "name": conv.name, "kernel_style": style, "groups": group,
            "weight_shape_oihw": list(w.shape), "mac_terms_per_output": mac_terms,
            "input_scale": float(x_scale), "input_zero_point": 0,
            "weight_initializer": w_dq.input[0],
            "weight_scales": w_scales.tolist(), "weight_zero_points": [0] * channels,
            "bias_int32": b.astype(np.int64).tolist(), "bias_scales": b_scales.tolist(),
            "output_scale": float(out_scale), "output_zero_point": 0,
            "requant_multipliers": multipliers.tolist(),
            "max_abs_bias_int32": int(np.abs(b.astype(np.int64)).max()),
            "conservative_abs_accumulator_bound": largest,
            "int32_bound_passed": True,
        }
        layers.append(layer)
        weights[w_dq.input[0]] = w

    if len(layers) != 102:
        raise RuntimeError(f"Expected 102 quantized Conv layers, got {len(layers)}")
    examples = arithmetic_examples(layers, weights)
    report = {
        "candidate_onnx_sha256": sha256(candidate_path),
        "scope": "Conv W8A8 only; other operators remain floating point",
        "arithmetic": {
            "activation": "signed INT8 per tensor, symmetric, zero point 0",
            "weight": "signed INT8 per output channel, symmetric, zero point 0",
            "bias": "signed INT32, scale = input_scale * weight_scale[channel]",
            "accumulator": "signed INT32, initial value bias_int32[channel]",
            "mac": "acc += activation_int8 * weight_int8",
            "requantization": "clip(round_ties_to_even(acc * input_scale * weight_scale[channel] / output_scale), -128, 127)",
            "overflow_check": "abs(acc) <= 128 * sum(abs(weight_int8[channel])) + abs(bias_int32[channel])",
            "weight_layout": "OIHW", "activation_layout": "NCHW",
        },
        "conv_count": len(layers),
        "max_mac_terms_per_output": max(layer["mac_terms_per_output"] for layer in layers),
        "max_conservative_abs_accumulator_bound": max(layer["conservative_abs_accumulator_bound"] for layer in layers),
        "all_int32_bounds_passed": True,
        "sample_arithmetic_checks": examples,
        "layers": layers,
    }
    report_path = output / "pe_numeric_contract.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Checked {len(layers)} QDQ Conv layers; max conservative INT32 bound: "
          f"{report['max_conservative_abs_accumulator_bound']:,}")
    print("Arithmetic checks:", [row["kernel_style"] for row in examples])
    print("Report:", report_path)


if __name__ == "__main__":
    main()
