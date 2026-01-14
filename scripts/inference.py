import argparse
import base64
import time
from pathlib import Path
import cv2
import numpy as np
from flask import Flask, request, jsonify
from ultralytics import YOLO


def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Bean Sorting Inference with YOLOv11')
    parser.add_argument('--model', type=str,
                        default='trained_model/my_model.pt',
                        help='Path to trained model weights')
    parser.add_argument('--source', type=str, default='0',
                        help='Video source (0 for webcam, or path to video/image file)')
    parser.add_argument('--conf', type=float, default=0.25,
                        help='Confidence threshold')
    parser.add_argument('--iou', type=float, default=0.45,
                        help='IOU threshold for NMS')
    parser.add_argument('--output', type=str, default=None,
                        help='Output video file path (optional)')
    parser.add_argument('--no-display', action='store_true',
                        help='Disable display window')
    parser.add_argument('--save-txt', action='store_true',
                        help='Save detection results to text file')
    parser.add_argument('--web', action='store_true',
                        help='Start as a web server (Flask) for frontend integration')
    return parser.parse_args()


def load_model(args):
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"[ERROR] Model file not found: {model_path}")
        print("Please train the model first or specify a valid model path.")
        exit(1)
    print(f"[INFO] Loading model from {model_path}")
    model = YOLO(str(model_path))
    return model


def run_inference(model, args):
    # Determine source type
    source = args.source
    if source.isdigit():
        source = int(source)  # webcam index
        print(f"[INFO] Using webcam index {source}")
    else:
        source = str(source)
        if not Path(source).exists():
            print(f"[ERROR] Source file not found: {source}")
            return

    # Open video capture
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[ERROR] Could not open video source: {source}")
        return

    # Get video properties for output writer
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS)) if cap.get(
        cv2.CAP_PROP_FPS) > 0 else 30

    # Initialize video writer if output specified
    writer = None
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(
            args.output, fourcc, fps, (frame_width, frame_height))
        print(f"[INFO] Saving output to {args.output}")

    # Statistics
    frame_count = 0
    total_good = 0
    total_bad = 0
    start_time = time.time()

    print("[INFO] Starting inference. Press 'q' to quit, 's' to save screenshot.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # Perform inference
        results = model.predict(
            source=frame,
            conf=args.conf,
            iou=args.iou,
            verbose=False,
            stream=True  # efficient for video
        )

        # Process detections
        good_count = 0
        bad_count = 0
        for r in results:
            boxes = r.boxes.xyxy.cpu().numpy()
            classes = r.boxes.cls.cpu().numpy().astype(int)
            confidences = r.boxes.conf.cpu().numpy()

            for bbox, cls, conf in zip(boxes, classes, confidences):
                x1, y1, x2, y2 = map(int, bbox)
                # Class 0 = bad, Class 1 = good (as per data.yaml)
                if cls == 1:
                    color = (0, 255, 0)  # Green for good
                    label = f"good {conf:.2f}"
                    good_count += 1
                else:
                    color = (0, 0, 255)  # Red for bad
                    label = f"bad {conf:.2f}"
                    bad_count += 1

                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                # Draw label background
                (label_width, label_height), _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                cv2.rectangle(frame, (x1, y1 - label_height - 10),
                              (x1 + label_width, y1), color, -1)
                cv2.putText(frame, label, (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        total_good += good_count
        total_bad += bad_count

        # Display statistics on frame
        stats_text = f"Frame: {frame_count} | Good: {good_count} | Bad: {bad_count} | Total: {good_count + bad_count}"
        cv2.putText(frame, stats_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Conf: {args.conf} | IOU: {args.iou}",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Write to output video
        if writer is not None:
            writer.write(frame)

        # Display
        if not args.no_display:
            cv2.imshow('Bean Sorting AI', frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                screenshot_path = f"screenshot_{frame_count}.png"
                cv2.imwrite(screenshot_path, frame)
                print(f"[INFO] Screenshot saved to {screenshot_path}")

    # Cleanup
    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()

    # Print summary
    elapsed = time.time() - start_time
    print("\n[INFO] Inference Summary:")
    print(f"  Total frames processed: {frame_count}")
    print(f"  Total good beans detected: {total_good}")
    print(f"  Total bad beans detected: {total_bad}")
    print(f"  Total beans: {total_good + total_bad}")
    print(
        f"  Average beans per frame: {(total_good + total_bad) / frame_count:.2f}")
    print(f"  Processing time: {elapsed:.2f}s ({frame_count/elapsed:.2f} FPS)")

    # Save text results if requested
    if args.save_txt:
        txt_path = "detection_results.txt"
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(f"Source: {args.source}\n")
            f.write(f"Model: {args.model}\n")
            f.write(f"Confidence threshold: {args.conf}\n")
            f.write(f"IOU threshold: {args.iou}\n")
            f.write(f"Total frames: {frame_count}\n")
            f.write(f"Total good beans: {total_good}\n")
            f.write(f"Total bad beans: {total_bad}\n")
            f.write(f"Total beans: {total_good + total_bad}\n")
            f.write(f"Processing time: {elapsed:.2f}s\n")
            f.write(f"Average FPS: {frame_count/elapsed:.2f}\n")
        print(f"[INFO] Results saved to {txt_path}")


def run_flask_server(model, args):
    app = Flask(__name__)

    @app.route('/process', methods=['POST'])
    def process_image():
        data = request.json
        if 'image' not in data:
            return jsonify({'error': 'No image data provided'}), 400

        image_data = data['image']
        if image_data.startswith('data:image/jpeg;base64,'):
            image_data = image_data.split(',')[1]

        try:
            img = base64.b64decode(image_data)
            nparr = np.frombuffer(img, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        #except Exception as e:
        except (ValueError, TypeError) as e:
            return jsonify({'error': str(e)}), 400

        results = model.predict(
            source=frame,
            conf=args.conf,
            iou=args.iou,
            verbose=False,
            stream=True
        )

        good_count = 0
        bad_count = 0
        processed_frame = frame.copy()
        for r in results:
            boxes = r.boxes.xyxy.cpu().numpy()
            classes = r.boxes.cls.cpu().numpy().astype(int)
            confidences = r.boxes.conf.cpu().numpy()
            for bbox, cls, conf in zip(boxes, classes, confidences):
                x1, y1, x2, y2 = map(int, bbox)
                if cls == 1:
                    color = (0, 255, 0)
                    label = f"good {conf:.2f}"
                    good_count += 1
                else:
                    color = (0, 0, 255)
                    label = f"bad {conf:.2f}"
                    bad_count += 1
                cv2.rectangle(processed_frame, (x1, y1), (x2, y2), color, 2)
                (label_width, label_height), _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                cv2.rectangle(processed_frame, (x1, y1 - label_height - 10),
                              (x1 + label_width, y1), color, -1)
                cv2.putText(processed_frame, label, (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        _, buffer = cv2.imencode('.jpg', processed_frame)
        processed_image_base64 = base64.b64encode(buffer).decode('utf-8')

        return jsonify({
            'good': good_count,
            'bad': bad_count,
            'total': good_count + bad_count,
            'confidence': args.conf,
            'processedImage': f"data:image/jpeg;base64,{processed_image_base64}"
        })

    print("[INFO] Starting Flask server on http://0.0.0.0:5000")
    app.run(host='0.0.0.0', port=5000, debug=True)


def main():
    args = parse_arguments()
    model = load_model(args)

    if args.web:
        run_flask_server(model, args)
    else:
        run_inference(model, args)


if __name__ == "__main__":
    main()
