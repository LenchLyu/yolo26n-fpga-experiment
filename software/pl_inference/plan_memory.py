"""Allocate a conservative DDR scratch arena for the fixed FP32 ONNX graph.

The allocator does not alias view operations. This is an initial address plan,
not the final mixed-precision accelerator storage format or AXI base address.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
ALIGN = 64


def align(value: int) -> int:
    return (value + ALIGN - 1) // ALIGN * ALIGN


def item_bytes(shape: list[int], dtype: str) -> int:
    if any(not isinstance(dim, int) or dim <= 0 for dim in shape):
        raise ValueError(f"Unresolved tensor shape: {shape}")
    return int(np.prod(shape, dtype=np.int64)) * np.dtype(dtype).itemsize


class Arena:
    def __init__(self) -> None:
        self.end = 0
        self.holes: list[tuple[int, int]] = []

    def take(self, size: int) -> int:
        need = align(size)
        for index, (offset, length) in enumerate(self.holes):
            if length >= need:
                if length == need:
                    self.holes.pop(index)
                else:
                    self.holes[index] = (offset + need, length - need)
                return offset
        offset = self.end
        self.end += need
        return offset

    def release(self, offset: int, size: int) -> None:
        self.holes.append((offset, align(size)))
        self.holes.sort()
        merged: list[tuple[int, int]] = []
        for start, length in self.holes:
            if merged and merged[-1][0] + merged[-1][1] == start:
                old_start, old_length = merged[-1]
                merged[-1] = (old_start, old_length + length)
            else:
                merged.append((start, length))
        self.holes = merged


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, default=HERE / "results" / "first_pass_graph.json")
    parser.add_argument("--observed", type=Path, default=HERE / "results" / "observed_shapes.json")
    parser.add_argument("--out", type=Path, default=HERE / "results")
    args = parser.parse_args()
    audit = json.loads(args.audit.expanduser().resolve().read_text(encoding="utf-8"))
    observed = json.loads(args.observed.expanduser().resolve().read_text(encoding="utf-8"))
    if audit["source_onnx_sha256"] != observed["source_onnx_sha256"]:
        raise RuntimeError("Graph audit and observed shapes are from different ONNX files")
    if observed["resolved_tensor_count"] != audit["unresolved_tensor_count"]:
        raise RuntimeError("Some unresolved tensor shapes were not observed")

    tensors = {}
    for item in audit["tensors"]:
        item = dict(item)
        if item["fp32_graph_bytes"] is None:
            actual = observed["tensors"].get(item["tensor"])
            if actual is None or actual["dtype"] != item["dtype"]:
                raise RuntimeError(f"Missing or conflicting observed shape: {item['tensor']}")
            item["shape"] = actual["shape"]
            item["fp32_graph_bytes"] = item_bytes(item["shape"], item["dtype"])
        tensors[item["tensor"]] = item

    inputs = {item["name"]: item for item in audit["fixed_input"]}
    if set(inputs) != {"images"}:
        raise RuntimeError(f"Unexpected model input: {sorted(inputs)}")
    consumers = {}
    for node in audit["nodes"]:
        for name in node["inputs"]:
            consumers.setdefault(name, []).append(node["index"])
    for name, item in inputs.items():
        tensors[name] = {
            "tensor": name, "producer_index": -1,
            "last_consumer_index": max(consumers[name]),
            "shape": item["shape"], "dtype": item["dtype"],
            "fp32_graph_bytes": item_bytes(item["shape"], item["dtype"]),
            "unit": "input",
        }

    arena = Arena()
    active = {}
    allocation = []
    peak_live = 0
    for index in range(-1, len(audit["nodes"])):
        current = ([tensors["images"]] if index == -1 else [
            tensors[name] for name in audit["nodes"][index]["outputs"]
            if name and tensors[name]["unit"] != "compile_time"
        ])
        for item in current:
            name = item["tensor"]
            size = item["fp32_graph_bytes"]
            if size is None:
                raise RuntimeError(f"Unresolved byte count: {name}")
            offset = arena.take(size)
            active[name] = (offset, size)
            allocation.append({**item, "offset_bytes": offset,
                               "allocated_bytes": align(size)})
        peak_live = max(peak_live, sum(align(size) for _, size in active.values()))
        for name, (offset, size) in list(active.items()):
            if tensors[name]["last_consumer_index"] == index:
                arena.release(offset, size)
                del active[name]
    expected_outputs = {item["name"] for item in audit["fixed_output"]}
    if set(active) != expected_outputs:
        raise RuntimeError(f"Unexpected tensors live at graph completion: {sorted(active)}")
    for position, left in enumerate(allocation):
        for right in allocation[position + 1:]:
            lifetimes_overlap = max(left["producer_index"], right["producer_index"]) <= min(
                left["last_consumer_index"], right["last_consumer_index"])
            addresses_overlap = max(left["offset_bytes"], right["offset_bytes"]) < min(
                left["offset_bytes"] + left["allocated_bytes"],
                right["offset_bytes"] + right["allocated_bytes"])
            if lifetimes_overlap and addresses_overlap:
                raise AssertionError(f"Live buffers overlap: {left['tensor']} and {right['tensor']}")

    output = args.out.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    with (output / "memory_intervals_fp32.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        writer.writerow(("tensor", "producer_index", "last_consumer_index", "shape",
                         "dtype", "logical_bytes", "offset_bytes", "allocated_bytes"))
        writer.writerows((item["tensor"], item["producer_index"],
                          item["last_consumer_index"], json.dumps(item["shape"]),
                          item["dtype"], item["fp32_graph_bytes"],
                          item["offset_bytes"], item["allocated_bytes"])
                         for item in allocation)
    report = {
        "source_onnx_sha256": audit["source_onnx_sha256"],
        "scope": "conservative FP32 ONNX scratch plan; all runtime outputs materialized",
        "relative_offsets_only": True,
        "alignment_bytes": ALIGN,
        "scratch_arena_bytes": arena.end,
        "peak_simultaneously_live_bytes": peak_live,
        "runtime_tensor_count": len(allocation),
        "compile_time_constants_excluded": True,
        "weight_storage_excluded": True,
        "all_tensor_sizes_resolved": True,
        "tensors": allocation,
    }
    destination = output / "memory_plan_fp32.json"
    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Initial FP32 scratch arena: {arena.end:,} bytes; peak live: {peak_live:,} bytes.")
    print(f"Allocated {len(allocation)} runtime tensors at {ALIGN}-byte alignment.")
    print("Plan:", destination)
    print("Intervals:", output / "memory_intervals_fp32.csv")


if __name__ == "__main__":
    main()
