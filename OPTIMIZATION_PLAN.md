# Bean Detection Project - Optimized Structure

This document describes the optimized directory structure for the bean detection project.

## Essential Files and Directories

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

## Files That Can Be Removed

1. **YOLOOOO directory** - Duplicate dataset with different structure
2. **Old training runs** - `runs/detect/train2/` through `runs/detect/train6/`
3. **Training scripts** - `scripts/train.py`, `scripts/resume_training.py`, `scripts/check_training_status.py`
4. **Documentation files** - `scripts/TRAINING.md`, `scripts/TRAINING_MONITORING.md`
5. **Web files** - `index.html`, `script.js`, `style.css` (if not needed for the model)
6. **Miscellaneous** - `directives/`, `execute/`, `yolo11n.pt`, `.roo/`

## Files to Keep for Model Inference

1. `trained_model/my_model.pt` - The trained model
2. `scripts/run_trained_model.py` - Our inference script
3. `scripts/inference.py` - Original inference script
4. `data/yolo/data.yaml` - Dataset configuration
5. `data/yolo/images/test/` - Test images
6. `config/data.yaml` - Alternative configuration