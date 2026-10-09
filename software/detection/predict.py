"""Run the copied signature model with a fixed 640x640 input."""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


SOFTWARE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WEIGHTS = SOFTWARE_ROOT / "models" / "signature_yolo26n" / "best_stripped.pt"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "outputs" / "signature_result.jpg"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Input image")
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS, help="PyTorch checkpoint")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Annotated output image")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--device", default="cpu", help="PyTorch device, e.g. cpu or 0")
    args = parser.parse_args()

    if not args.source.is_file():
        parser.error(f"Input image not found: {args.source}")
    if not args.weights.is_file():
        parser.error(f"Weights not found: {args.weights}")
    if not 0 <= args.conf <= 1:
        parser.error("--conf must be between 0 and 1")

    result = YOLO(str(args.weights)).predict(
        source=str(args.source),
        imgsz=640,
        rect=False,
        conf=args.conf,
        device=args.device,
        verbose=False,
    )[0]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.save(filename=str(args.output))
    print(f"Saved annotated image: {args.output}")
    if result.boxes is None or len(result.boxes) == 0:
        print("No detections")
        return
    for index, (xyxy, confidence, category) in enumerate(
        zip(result.boxes.xyxy.cpu().tolist(), result.boxes.conf.cpu().tolist(), result.boxes.cls.cpu().tolist()),
        start=1,
    ):
        coords = [round(value, 1) for value in xyxy]
        print(f"{index}. {result.names[int(category)]}: xyxy={coords}, confidence={confidence:.3f}")


if __name__ == "__main__":
    main()
