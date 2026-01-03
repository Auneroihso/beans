# How to Run the Bean Detection Project

This guide provides step-by-step instructions on how to run the trained bean detection model with various options including live camera feed and external image folders.

## Prerequisites

Before running the model, ensure you have:
1. Python 3.7 or higher installed
2. Required Python packages installed:
   ```
   pip install ultralytics opencv-python
   ```

## Project Structure

After optimization, your project should have the following structure:
```
beans/
├── trained_model/
│   └── my_model.pt              # Trained model weights
├── data/
│   └── yolo/
│       ├── data.yaml            # Dataset configuration
│       └── images/
│           ├── test/            # Test images
│           ├── train/           # Training images
│           └── val/             # Validation images
├── scripts/
│   ├── run_trained_model.py     # Simplified inference script
│   └── inference.py             # Full-featured inference script
└── runs/
    └── detect/
        └── predict*/            # Results from inference
```

## Running the Model

### 1. Basic Inference on a Single Image

To run inference on a single image from the test dataset:

```bash
python scripts/run_trained_model.py --source data/yolo/images/test/IMG_20251203_070539.jpg
```

### 2. Inference on All Images in a Directory

To run inference on all images in the test directory:

```bash
python scripts/run_trained_model.py --source data/yolo/images/test
```

### 3. Using a Live Camera Feed

To use your webcam for real-time bean detection:

```bash
python scripts/run_trained_model.py --source 0
```

Note: The number `0` represents the default webcam. If you have multiple cameras, you can use `1`, `2`, etc.

### 4. Using an External Image Folder

To run inference on images from an external folder:

1. Create a folder with your images (e.g., `C:\MyImages\beans\`)
2. Run the model on that folder:
   ```bash
   python scripts/run_trained_model.py --source "C:\MyImages\beans"
   ```

### 5. Adjusting Detection Parameters

You can adjust the confidence and IOU thresholds for detection:

```bash
# Higher confidence threshold (more selective)
python scripts/run_trained_model.py --source data/yolo/images/test --conf 0.5

# Different IOU threshold for non-maximum suppression
python scripts/run_trained_model.py --source data/yolo/images/test --iou 0.6
```

## Advanced Usage with inference.py

The `inference.py` script provides more options:

### 1. Live Camera with Display Control

```bash
# Run with webcam and disable display window
python scripts/inference.py --source 0 --no-display

# Save results to text file
python scripts/inference.py --source data/yolo/images/test --save-txt
```

### 2. Save Results to Video File

```bash
# Save webcam feed to video file
python scripts/inference.py --source 0 --output bean_detection.mp4
```

### 3. Start Web Server for Frontend Integration

```bash
# Start Flask server for web integration
python scripts/inference.py --web
```

## Understanding the Results

After running inference, results are saved to `runs/detect/predict*/` where `*` is a number that increments with each run. The output includes:

1. Annotated images with bounding boxes around detected beans
2. Each bean is labeled as either "good" (green) or "bad" (red) with confidence scores
3. Statistics on the number of good and bad beans detected

## Troubleshooting

### Common Issues:

1. **"Model file not found" error**: Ensure `trained_model/my_model.pt` exists
2. **"Source not found" error**: Check that your image path or camera index is correct
3. **No detections**: Try lowering the confidence threshold with `--conf 0.1`

### Performance Tips:

1. For faster inference, use a lower confidence threshold
2. For more accurate detections, use a higher confidence threshold
3. If using a webcam, ensure good lighting conditions

## Customization

You can modify the scripts to:
1. Change class labels in `data/yolo/data.yaml`
2. Adjust preprocessing steps in the Python scripts
3. Add post-processing for specialized bean analysis

## Example Workflows

### Quality Control Workflow:
1. Set up a camera above a conveyor belt
2. Run `python scripts/inference.py --source 0 --web`
3. Connect to the web interface to monitor bean quality in real-time

### Batch Processing Workflow:
1. Place all bean images in a folder
2. Run `python scripts/run_trained_model.py --source "C:\MyBeanImages" --conf 0.3`
3. Check the results in `runs/detect/predict*/` for quality assessment

## Test Results

We tested the system with an external folder containing 38 images (`C:\Users\USER\Desktop\mixed`). The model successfully processed all images, detecting:
- 31 images with beans (either good or bad)
- 7 images with no detections
- Various combinations of good and bad beans

The system demonstrates robust performance across different image conditions and successfully identifies both good and bad beans with confidence scores.