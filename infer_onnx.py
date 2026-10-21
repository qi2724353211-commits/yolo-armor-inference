"""Run the exported ONNX model on an image and print/save the detections.

This script is intentionally kept as close as possible to the PyTorch
inference script (infer_pt.py) so the two can be compared fairly:

  * same input image and image size (640)
  * same confidence threshold and IoU threshold
  * same pre-processing (letterbox + BGR->RGB + 1/255 normalisation)
  * same timing methodology (warm-up + repeated forward passes)
"""

import argparse
import os
import time

import cv2
import numpy as np
import torch

# onnxruntime's CUDA provider depends on CUDA/cuDNN DLLs.  PyTorch already
# ships a compatible copy of those DLLs, so we put torch/lib on PATH before
# importing onnxruntime.
_torch_lib = os.path.join(os.path.dirname(torch.__file__), "lib")
if os.path.isdir(_torch_lib):
    os.environ["PATH"] = _torch_lib + os.pathsep + os.environ.get("PATH", "")

import onnxruntime as ort


CLASS_NAMES = {0: "B1", 1: "B2", 2: "B3", 3: "R1", 4: "R2", 5: "R3"}
CLASS_COLORS = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 0), (0, 255, 255), (255, 0, 255)]
NUM_CLASSES = len(CLASS_NAMES)
NUM_KEYPOINTS = 4


def letterbox(im, new_shape=(640, 640), color=(114, 114, 114)):
    shape = im.shape[:2]
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = (new_shape[1] - new_unpad[0]) / 2, (new_shape[0] - new_unpad[1]) / 2
    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return im, r, (dw, dh)


def preprocess(img_path, imgsz):
    img0 = cv2.imread(img_path)
    if img0 is None:
        raise FileNotFoundError(f"Could not read image: {img_path}")
    img, r, dwdh = letterbox(img0, (imgsz, imgsz))
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR -> RGB, HWC -> CHW
    img = np.ascontiguousarray(img, dtype=np.float32) / 255.0
    return np.expand_dims(img, axis=0), img0, r, dwdh


def postprocess(outputs, img0, r, dwdh, conf_thres, iou_thres):
    # ONNX output shape: [1, 4 + C + K*2, anchors] = [1, 18, 8400]
    preds = np.squeeze(outputs[0]).T  # -> [8400, 18]

    boxes = preds[:, :4]                 # cx, cy, w, h (letterboxed coords)
    scores = preds[:, 4:4 + NUM_CLASSES]
    kpts = preds[:, 4 + NUM_CLASSES:]

    class_ids = np.argmax(scores, axis=1)
    confidences = np.max(scores, axis=1)

    keep = confidences > conf_thres
    boxes, confidences, class_ids, kpts = boxes[keep], confidences[keep], class_ids[keep], kpts[keep]
    if len(boxes) == 0:
        return img0, []

    # NMS expects top-left + width/height.
    xywh = np.zeros_like(boxes)
    xywh[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
    xywh[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
    xywh[:, 2] = boxes[:, 2]
    xywh[:, 3] = boxes[:, 3]

    idx = cv2.dnn.NMSBoxes(xywh.tolist(), confidences.tolist(), conf_thres, iou_thres)
    if isinstance(idx, tuple):
        idx = idx[0] if len(idx) else np.array([], dtype=int)
    idx = np.asarray(idx).flatten().astype(int)

    dw, dh = dwdh
    detections = []
    for i in idx:
        cx, cy, w, h = boxes[i]
        x1 = int((cx - w / 2 - dw) / r)
        y1 = int((cy - h / 2 - dh) / r)
        x2 = int((cx + w / 2 - dw) / r)
        y2 = int((cy + h / 2 - dh) / r)
        cls_id = int(class_ids[i])
        conf = float(confidences[i])

        color = CLASS_COLORS[cls_id % len(CLASS_COLORS)]
        cv2.rectangle(img0, (x1, y1), (x2, y2), color, 2)
        label = f"{CLASS_NAMES[cls_id]} {conf:.2f}"
        cv2.putText(img0, label, (x1, max(y1 - 10, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        kpt = kpts[i].reshape(NUM_KEYPOINTS, 2)
        for kx, ky in kpt:
            kx_o, ky_o = int((kx - dw) / r), int((ky - dh) / r)
            cv2.circle(img0, (kx_o, ky_o), 3, (0, 255, 0), -1)

        detections.append((CLASS_NAMES[cls_id], conf, (x1, y1, x2, y2)))

    return img0, detections


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="assets/test.jpg")
    parser.add_argument("--model", default="best.onnx")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--save", default="outputs/result_onnx.png")
    args = parser.parse_args()

    if "CUDAExecutionProvider" in ort.get_available_providers():
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    else:
        providers = ["CPUExecutionProvider"]

    session = ort.InferenceSession(args.model, providers=providers)
    input_name = session.get_inputs()[0].name

    img_tensor, img0, r, dwdh = preprocess(args.source, args.imgsz)

    # Warm-up.
    for _ in range(10):
        session.run(None, {input_name: img_tensor})

    runs = 30
    t0 = time.perf_counter()
    for _ in range(runs):
        outputs = session.run(None, {input_name: img_tensor})
    pure_ms = (time.perf_counter() - t0) / runs * 1000

    t0 = time.perf_counter()
    result_img, detections = postprocess(outputs, img0, r, dwdh, args.conf, args.iou)
    e2e_ms = (time.perf_counter() - t0) * 1000

    os.makedirs(os.path.dirname(args.save), exist_ok=True)
    cv2.imwrite(args.save, result_img)

    print(f"Provider      : {session.get_providers()}")
    print(f"Pure forward  : {pure_ms:.2f} ms/image (avg over {runs} runs)")
    print(f"Detections    : {len(detections)}")
    for name, conf, (x1, y1, x2, y2) in detections:
        print(f"  {name:3s} conf={conf:.3f} box=({x1},{y1},{x2},{y2})")
    print(f"Saved         : {args.save}")


if __name__ == "__main__":
    main()
