# Bean Sorter & Quality Inspector

## 🫘 Project Overview

The **Bean Sorter** is an AI-powered computer vision system designed to identify and classify beans for automated sorting. This project serves as **Part One** of a larger system, establishing the core intelligence needed to drive an electromechanical sorting mechanism.

Using a custom-trained **YOLO11n** model, the system distinguishes between:
- **Good Beans** (Class 1, Green): Beans that meet quality standards.
- **Bad Beans** (Class 0, Red): Beans that are defective and should be rejected.

The ultimate goal of this trained model is to interface with hardware that physically separates the beans based on these real-time classifications.

## ✨ Key Features

- **Real-time Classification:** High-speed identification of bean quality using YOLO11n.
- **Sorting Logic:** Clearly distinguishes 'Good' vs 'Bad' beans for downstream hardware triggers.
- **Visual Feedback:** Color-coded bounding boxes (Green for Good, Red for Bad) for easy monitoring.
- **Multiple Inputs:** Supports live webcam feeds, video files, and image datasets.
- **Web Interface API:** Built-in Flask server to process images, allowing integration with external control systems or frontends.

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

The project provides two main scripts in the `scripts/` directory:

### 1. Live Webcam Inference
To run real-time detection (simulation of the sorting eye):
```bash
python scripts/inference.py --source 0
```
*Controls:* Press `q` to quit, `s` to save a screenshot.

### 2. Process an Image Directory
To validate the model on a dataset:
```bash
python scripts/inference.py --source data/yolo/images/test --save-txt
```

### 3. Run the Web Server
Start the Flask backend for integration with other systems:
```bash
python scripts/inference.py --web
```
The server starts at `http://0.0.0.0:5000` and accepts POST requests at `/process`.

### 4. Basic Run Script
For a quick simplified test:
```bash
python scripts/run_trained_model.py --source 0
```

## ⚙️ Configuration

- **Model Weights:** Loaded from `trained_model/my_model.pt`.
- **Thresholds:**
  - Confidence: Default `0.25` (`--conf`)
  - IOU: Default `0.45` (`--iou`)
- **Classes:**
  - `0`: Bad Beans (Trigger Rejection)
  - `1`: Good Beans (Keep)

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
│   └── my_model.pt        # Trained YOLO11n weights
├── inference_results.txt  # Output log (optional)
└── README.md              # Project documentation
```

## 📝 Roadmap (Part Two & Beyond)

This project is currently in the software identification phase. Future development includes:

- [ ] **Hardware Integration:** Interface with Arduino/Raspberry Pi to control servos or air jets for physical sorting.
- [ ] **Signal Output:** Modify scripts to send GPIO signals when "Bad" beans are detected.
- [ ] **Speed Optimization:** Optimize inference for high-speed conveyor belts.
- [ ] **Advanced Logic:** Tracking individual beans across frames to prevent double-counting/double-sorting.

## 📄 License

[Include your license here, e.g., MIT License]
