
from ultralytics import YOLO
import torch

def main():
    model = YOLO("best.pt")

    print("模型任务：", model.task)
    print("关键点配置：", model.model.yaml.get("kpt_shape"))
    print("类别名称：", model.names)

    if not torch.cuda.is_available():
        print("CUDA 不可用，将无法进行 GPU 测试。")
        return

    print("使用显卡：", torch.cuda.get_device_name(0))

    # 创建一张临时黑色图片，测试模型前向推理
    import numpy as np

    image = np.zeros((640, 640, 3), dtype=np.uint8)
    results = model.predict(
        source=image,
        device=0,
        imgsz=640,
        verbose=False
    )

    result = results[0]
    print("GPU 推理测试成功")
    print("检测框数量：", len(result.boxes))
    print(
        "关键点输出形状：",
        tuple(result.keypoints.data.shape)
        if result.keypoints is not None
        else "无关键点输出"
    )

if __name__ == "__main__":
    main()
