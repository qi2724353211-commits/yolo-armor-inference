from ultralytics import YOLO

model = YOLO("best.pt")

print("===类别名===")
print(model.names)
print("类别数:", len(model.names))

print("\n=== 模型版本线索 ===")
print(type(model.model).__name__)
