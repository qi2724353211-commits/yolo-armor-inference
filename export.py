import os
os.environ['ULTRALYTICS_DISABLE_AUTOUPDATE'] = '1'#必须在import ultralytics之前

from ultralytics import YOLO
model = YOLO("best.pt")
model.export(format="onnx", imgsz=640, opset=12, simplify=True, dynamic=False)