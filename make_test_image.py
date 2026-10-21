"""Extract a clean test frame from a source video for .pt vs .onnx comparison.

The previous assignment used videos/ as the test data.  To keep the comparison
fair and reproducible we extract a single un-annotated frame where the armour
plate is clearly visible, then run BOTH the PyTorch model and the ONNX model on
that exact same image.
"""

import argparse
import os
import sys

import cv2
import numpy as np
import torch

# Reuse PyTorch's bundled CUDA/cuDNN runtime for the onnxruntime CUDA provider.
_torch_lib = os.path.join(os.path.dirname(torch.__file__), "lib")
if os.path.isdir(_torch_lib):
    os.environ["PATH"] = _torch_lib + os.pathsep + os.environ.get("PATH", "")

import onnxruntime as ort


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
    return im


def preprocess(img0):
    img = letterbox(img0, (640, 640))
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR -> RGB, HWC -> CHW
    img = np.ascontiguousarray(img, dtype=np.float32) / 255.0
    return np.expand_dims(img, axis=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", default="videos/video_3.mp4")
    parser.add_argument("--model", default="best.onnx")
    parser.add_argument("--out", default="assets/test.jpg")
    parser.add_argument("--step", type=int, default=1)
    args = parser.parse_args()

    if "CUDAExecutionProvider" in ort.get_available_providers():
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    else:
        providers = ["CPUExecutionProvider"]
    session = ort.InferenceSession(args.model, providers=providers)
    input_name = session.get_inputs()[0].name

    cap = cv2.VideoCapture(args.video)
    best_score = -1.0
    best_frame = None
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % args.step != 0:
            frame_idx += 1
            continue
        blob = preprocess(frame)
        out = session.run(None, {input_name: blob})[0]
        preds = np.squeeze(out).T
        conf = np.max(preds[:, 4:10], axis=1)
        score = float(conf.max())
        if score > best_score:
            best_score = score
            best_frame = frame.copy()
        frame_idx += 1

    cap.release()
    if best_frame is None:
        sys.exit("No frame could be read.")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    cv2.imwrite(args.out, best_frame)
    print(f"Best frame confidence={best_score:.4f} saved to {args.out}")


if __name__ == "__main__":
    main()
