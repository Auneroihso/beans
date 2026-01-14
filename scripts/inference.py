import argparse
import base64
import time
import csv
from pathlib import Path
from datetime import datetime
import cv2
import numpy as np
from flask import Flask, request, jsonify
from ultralytics import YOLO

# Import for reporting
try:
    import matplotlib.pyplot as plt
    from fpdf import FPDF
    REPORTING_AVAILABLE = True
except ImportError:
    REPORTING_AVAILABLE = False
    print("[WARN] matplotlib or fpdf not found. Advanced reporting disabled.")

def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Bean Sorting Inference with YOLO11n')
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
    parser.add_argument('--report', action='store_true',
                        help='Generate CSV and PDF reports')
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

def generate_csv_report(history, output_path="report.csv"):
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Frame/ID", "Timestamp", "Good Beans", "Bad Beans", "Total", "Inference Time (ms)"])
        for row in history:
            writer.writerow([
                row['frame'],
                row['timestamp'],
                row['good'],
                row['bad'],
                row['good'] + row['bad'],
                f"{row['inference_ms']:.2f}"
            ])
    print(f"[INFO] CSV Report saved to {output_path}")

def generate_pdf_report(history, summary, output_path="report.pdf"):
    if not REPORTING_AVAILABLE:
        print("[ERROR] Cannot generate PDF. Install matplotlib and fpdf.")
        return

    # Generate charts
    total_good = summary['total_good']
    total_bad = summary['total_bad']

    # Pie Chart
    plt.figure(figsize=(6, 6))
    if total_good + total_bad > 0:
        plt.pie([total_good, total_bad], labels=['Good', 'Bad'], colors=['green', 'red'], autopct='%1.1f%%')
    else:
        plt.text(0.5, 0.5, 'No Detections', ha='center')
    plt.title('Good vs Bad Bean Distribution')
    plt.savefig('temp_pie_chart.png')
    plt.close()

    # Timeline Chart (if enough data)
    if len(history) > 1:
        frames = [x['frame'] for x in history]
        good_counts = [x['good'] for x in history]
        bad_counts = [x['bad'] for x in history]

        plt.figure(figsize=(10, 4))
        plt.plot(frames, good_counts, label='Good', color='green')
        plt.plot(frames, bad_counts, label='Bad', color='red')
        plt.xlabel('Frame/Image ID')
        plt.ylabel('Count')
        plt.title('Detection Timeline')
        plt.legend()
        plt.grid(True)
        plt.savefig('temp_timeline.png')
        plt.close()

    # Create PDF
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(40, 10, "Bean Sorting Quality Report")
    pdf.ln(10)

    pdf.set_font("Arial", size=12)
    pdf.cell(0, 10, f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)
    pdf.cell(0, 10, f"Source: {summary['source']}", ln=True)
    pdf.cell(0, 10, f"Model: {summary['model']}", ln=True)
    pdf.ln(10)

    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, "Summary Statistics", ln=True)
    pdf.set_font("Arial", size=12)
    pdf.cell(0, 10, f"Total Frames/Images: {summary['frames']}", ln=True)
    pdf.cell(0, 10, f"Total Good Beans Detected: {total_good}", ln=True)
    pdf.cell(0, 10, f"Total Bad Beans Detected: {total_bad}", ln=True)

    # Calculate Yield
    total = total_good + total_bad
    yield_pct = (total_good / total * 100) if total > 0 else 0
    pdf.cell(0, 10, f"Yield (Good/Total): {yield_pct:.2f}%", ln=True)
    pdf.ln(10)

    # Add Charts
    pdf.image('temp_pie_chart.png', x=10, y=None, w=100)
    if len(history) > 1:
        pdf.image('temp_timeline.png', x=10, y=None, w=180)

    pdf.output(output_path)

    # Cleanup temp files
    Path('temp_pie_chart.png').unlink(missing_ok=True)
    Path('temp_timeline.png').unlink(missing_ok=True)

    print(f"[INFO] PDF Report saved to {output_path}")

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

    # History for reporting
    history = []

    print("[INFO] Starting inference. Press 'q' to quit, 's' to save screenshot.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        frame_start = time.time()

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

        # Record history
        inference_time = (time.time() - frame_start) * 1000
        history.append({
            'frame': frame_count,
            'timestamp': datetime.now().isoformat(),
            'good': good_count,
            'bad': bad_count,
            'inference_ms': inference_time
        })

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

    # Generate Advanced Reports
    if args.report:
        generate_csv_report(history)
        if REPORTING_AVAILABLE:
            summary = {
                'source': str(args.source),
                'model': str(args.model),
                'frames': frame_count,
                'total_good': total_good,
                'total_bad': total_bad
            }
            generate_pdf_report(history, summary)


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
