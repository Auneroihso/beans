import argparse
from pathlib import Path
#import cv2
from ultralytics import YOLO


def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description='Run inference with trained bean detection model')
    parser.add_argument('--model', type=str, default='trained_model/my_model.pt',
                        help='Path to trained model weights')
    parser.add_argument('--source', type=str, default='data/yolo/images/test',
                        help='Source for inference (image file, directory, or webcam index)')
    parser.add_argument('--conf', type=float, default=0.25,
                        help='Confidence threshold')
    parser.add_argument('--iou', type=float, default=0.45,
                        help='IOU threshold for NMS')
    args = parser.parse_args()

    # Load the trained model
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"[ERROR] Model file not found: {model_path}")
        return 1

    print(f"[INFO] Loading model from {model_path}")
    model = YOLO(str(model_path))

    # Determine the source type
    source = Path(args.source)

    if source.is_file():
        # Process a single image file
        print(f"[INFO] Running inference on single image: {source}")
        results = model.predict(
            source=str(source),
            conf=args.conf,
            iou=args.iou,
            save=True,
            show=False
        )
    elif source.is_dir():
        # Process all images in a directory
        print(f"[INFO] Running inference on all images in directory: {source}")
        results = model.predict(
            source=str(source),
            conf=args.conf,
            iou=args.iou,
            save=True,
            show=False
        )
    elif args.source.isdigit():
        # Process webcam feed
        webcam_index = int(args.source)
        print(f"[INFO] Starting webcam (index {webcam_index})")
        results = model.predict(
            source=webcam_index,
            conf=args.conf,
            iou=args.iou,
            show=True
        )
    else:
        print(f"[ERROR] Invalid source: {args.source}")
        return 1

    print("[INFO] Inference completed successfully!")
    print("[INFO] Results saved to runs/detect/predict/")
    return 0


if __name__ == "__main__":
    exit(main())
