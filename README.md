# Bean Counter & Quality Inspector

## 🫘 Project Overview

The **Bean Counter** is an AI-powered computer vision system designed to detect, count, and classify beans in real-time. Using a custom-trained **YOLOv11** model, it distinguishes between "good" beans (green bounding boxes) and "bad" beans (red bounding boxes), providing instant quality control statistics.

This system supports multiple input sources, including static images, video files, and live webcam feeds. It also features a **Flask-based web server** for easy integration with frontend applications.

## ✨ Key Features

- **Real-time Detection:** High-speed detection of beans using YOLOv11.
- **Quality Classification:** Automatically sorts beans into 'Good' (Class 1) and 'Bad' (Class 0) categories.
- **Counting & Statistics:** Live tracking of total, good, and bad bean counts per session.
- **Multiple Inputs:** Works with:
  - Live Webcam Feed
  - Image Directories
  - Single Image/Video Files
- **Web Interface API:** Built-in Flask server to process base64 images from web clients.
- **Visual Feedback:** Color-coded bounding boxes and on-screen statistics.

## 🛠️ Installation

1. **Clone the Repository**
   ```bash
   git clone <your-repo-url>
   cd <repo-name>
   ```

2. **Install Dependencies**
   Ensure you have Python 3.8+ installed. Then run:
   ```bash
   pip install ultralytics opencv-python flask numpy
   ```
   *(Note: `opencv-python-headless` may be used for server-only environments)*

## 🚀 Usage

The project provides two main scripts in the `scripts/` directory: `inference.py` (full-featured) and `run_trained_model.py` (simplified).

### 1. Live Webcam Inference
To run detection using your default webcam:
```bash
python scripts/inference.py --source 0
```
*Controls:* Press `q` to quit, `s` to save a screenshot.

### 2. Process an Image Directory
To detect beans in a folder of images and save the results:
```bash
python scripts/inference.py --source data/yolo/images/test --save-txt
```

### 3. Run the Web Server
Start the Flask backend for web integration:
```bash
python scripts/inference.py --web
```
The server will start at `http://0.0.0.0:5000`. You can POST images to the `/process` endpoint.

### 4. Basic Run Script
For a quick test using the simplified runner:
```bash
python scripts/run_trained_model.py --source 0
```

## ⚙️ Configuration

- **Model Weights:** The default model is loaded from `trained_model/my_model.pt`.
- **Thresholds:**
  - Confidence: Default `0.25` (Adjust with `--conf`)
  - IOU: Default `0.45` (Adjust with `--iou`)
- **Classes:**
  - `0`: Bad Beans (Red)
  - `1`: Good Beans (Green)

## 📂 Project Structure

```
.
├── data/
│   └── yolo/              # Dataset configuration and images
├── runs/                  # Detection results (images, labels)
├── scripts/
│   ├── inference.py       # Main application with CLI and Web modes
│   └── run_trained_model.py # Simplified inference runner
├── trained_model/
│   └── my_model.pt        # Trained YOLOv11 weights
├── inference_results.txt  # Output log (optional)
└── README.md              # Project documentation
```

## 📝 To-Do & Roadmap

- [x] Implement YOLO inference and counting
- [x] Add webcam and video support
- [x] Integrate Flask web server
- [ ] Improve counting accuracy for dense clusters
- [ ] Optimize detection parameters

## 📄 License

[Include your license here, e.g., MIT License]
