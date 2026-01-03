# Bean Detection Project - Optimized Version

This is the optimized version of the bean detection project, containing only the essential files needed to run the trained model.

## Directory Structure

```
beans/
├── trained_model/
│   └── my_model.pt              # Trained model weights
├── data/
│   └── yolo/
│       ├── data.yaml            # Dataset configuration
│       ├── images/
│       │   └── test/            # Test images for inference
│       └── labels/
│           └── test/            # Labels for test images
├── scripts/
│   ├── run_trained_model.py     # Script to run the trained model
│   └── inference.py             # Original inference script
├── runs/
│   └── detect/
│       ├── predict/             # Results from inference
│       └── predict2/            # Results from inference
└── config/
    └── data.yaml                # Alternative configuration
```

## Running the Model

To run inference with the trained model:

```bash
# Run on a single image
python scripts/run_trained_model.py --source data/yolo/images/test/IMG_20251203_070550.jpg

# Run on all images in the test directory
python scripts/run_trained_model.py --source data/yolo/images/test

# Run with different confidence threshold
python scripts/run_trained_model.py --source data/yolo/images/test --conf 0.5
```

## Files Removed in Optimization

The following files and directories were removed to create this optimized version:

1. **YOLOOOO directory** - Duplicate dataset with different structure
2. **Old training runs** - `runs/detect/train2/` through `runs/detect/train6/`
3. **Training scripts** - `scripts/train.py`, `scripts/resume_training.py`, `scripts/check_training_status.py`
4. **Documentation files** - `scripts/TRAINING.md`, `scripts/TRAINING_MONITORING.md`
5. **Web files** - `index.html`, `script.js`, `style.css` (if not needed for the model)
6. **Miscellaneous** - `directives/`, `execute/`, `yolo11n.pt`, `.roo/`

This optimization reduces the project size while maintaining all functionality needed for inference with the trained model.