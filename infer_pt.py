"""Run the original PyTorch model on an image and print/save the detections.

Kept symmetric with infer_onnx.py so the .pt vs .onnx comparison is fair.
"""

import argparse
import os
import time

import cv2
import torch
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="assets/test.jpg")
    parser.add_argument("--model", default="best.pt")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--device", default="0")
    parser.add_argument("--save", default="outputs/result_pt.png")
    args = parser.parse_args()

    use_gpu = args.device != "cpu" and torch.cuda.is_available()
    device = "cuda:0" if use_gpu else "cpu"

    model = YOLO(args.model).to(device)

    # ---- Pure forward timing (same dummy-input approach as ONNX). ----
    dummy = torch.zeros(1, 3, args.imgsz, args.imgsz).to(device)
    with torch.no_grad():
        for _ in range(10):
            model.model(dummy)
    if use_gpu:
        torch.cuda.synchronize()

    runs = 30
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(runs):
            if use_gpu:
                torch.cuda.synchronize()
            model.model(dummy)
            if use_gpu:
                torch.cuda.synchronize()
    pure_ms = (time.perf_counter() - t0) / runs * 1000

    # ---- End-to-end inference (pre/post processing included). ----
    t0 = time.perf_counter()
    results = model.predict(
        source=args.source,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=device,
        verbose=False,
    )
    e2e_ms = (time.perf_counter() - t0) * 1000

    result = results[0]
    annotated = result.plot()
    os.makedirs(os.path.dirname(args.save), exist_ok=True)
    cv2.imwrite(args.save, annotated)

    detections = []
    if result.boxes is not None and len(result.boxes) > 0:
        for cls, conf, xyxy in zip(
            result.boxes.cls.cpu().tolist(),
            result.boxes.conf.cpu().tolist(),
            result.boxes.xyxy.cpu().tolist(),
        ):
            cls_id = int(cls)
            name = result.names[cls_id]
            x1, y1, x2, y2 = (int(v) for v in xyxy)
            detections.append((name, float(conf), (x1, y1, x2, y2)))

    print(f"Device        : {device}")
    print(f"Pure forward  : {pure_ms:.2f} ms/image (avg over {runs} runs)")
    print(f"End-to-end    : {e2e_ms:.2f} ms")
    print(f"Detections    : {len(detections)}")
    for name, conf, (x1, y1, x2, y2) in detections:
        print(f"  {name:3s} conf={conf:.3f} box=({x1},{y1},{x2},{y2})")
    print(f"Saved         : {args.save}")


if __name__ == "__main__":
    main()
