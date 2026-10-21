import os
import torch

# 让 onnxruntime 的 CUDA provider 能复用 PyTorch 自带的 CUDA/cuDNN 运行库。
_torch_lib = os.path.join(os.path.dirname(torch.__file__), "lib")
if os.path.isdir(_torch_lib):
    os.environ["PATH"] = _torch_lib + os.pathsep + os.environ.get("PATH", "")

import onnxruntime as ort

MODEL = "best.onnx"


if not os.path.exists(MODEL):
    raise FileNotFoundError(f"找不到 {MODEL}, 请确认文件是否在当前目录下")

with open(MODEL, "rb") as f:
    magic = f.read(4)
print(f"文件头字节: {magic!r}")

providers = (
    ["CUDAExecutionProvider", "CPUExecutionProvider"]
    if "CUDAExecutionProvider" in ort.get_available_providers()
    else ["CPUExecutionProvider"]
)
session = ort.InferenceSession(MODEL, providers=providers)

print("\nInputs:")
for i in session.get_inputs():
    print(f"  name={i.name!r}, shape={i.shape}, type={i.type}")

print("\nOutputs:")
for o in session.get_outputs():
    print(f"  name={o.name!r}, shape={o.shape}, type={o.type}")
