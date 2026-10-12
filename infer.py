from pathlib import Path
import argparse

from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(
        description="YOLO armor plate detection and keypoint inference"
    )
    parser.add_argument(
        "--source",
        required=True,
        help="Input image or video path"
    )
    parser.add_argument(
        "--model",
        default="best.pt",
        help="Model weights path (default: best.pt)"
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Inference image size (default: 640)"
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold (default: 0.25)"
    )
    parser.add_argument(
        "--device",
        default="0",
        help="Inference device: 0 for GPU, cpu for CPU"
    )
    args = parser.parse_args()

    project_dir = Path(__file__).resolve().parent

    model_path = Path(args.model)
    if not model_path.is_absolute():
        model_path = project_dir / model_path

    source_path = Path(args.source)
    if not source_path.is_absolute():
        source_path = project_dir / source_path

    if not model_path.is_file():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    if not source_path.is_file():
        raise FileNotFoundError(f"Input file not found: {source_path}")

    output_dir = project_dir / "outputs" / "inference"
    output_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(model_path))

    print("Model task:", model.task)
    print("Keypoint shape:", model.model.yaml.get("kpt_shape"))
    print("Class names:", model.names)

    if model.task != "pose":
        raise ValueError("The supplied model is not a pose model.")

    if model.model.yaml.get("kpt_shape") != [4, 2]:
        raise ValueError("Expected four 2D keypoints per target.")

    print("Input:", source_path)
    print("Output directory:", output_dir)
    print("Starting inference...")

    results = model.predict(
        source=str(source_path),
        device=args.device,
        imgsz=args.imgsz,
        conf=args.conf,
        save=True,
        project=str(output_dir.parent),
        name=output_dir.name,
        exist_ok=True,
        stream=True,
        verbose=False,
        show=False
    )

    frame_count = 0
    total_detections = 0

    for result in results:
        frame_count += 1
        detection_count = len(result.boxes)
        total_detections += detection_count

        if frame_count == 1:
            print("Actual save directory:", result.save_dir)

        if frame_count % 50 == 0:
            print(
                f"Processed {frame_count} frames; "
                f"{detection_count} detections in the current frame."
            )

    print("\nInference finished.")
    print("Frames processed:", frame_count)
    print("Sum of detections across frames:", total_detections)
    print("Output directory:", output_dir)
    print("Check the saved video to verify detection boxes, labels,")
    print("confidence scores, and the four predicted keypoints.")


if __name__ == "__main__":
    main()
