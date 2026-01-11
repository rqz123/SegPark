# SegPark

YOLOv8 segmentation model for parking spot detection using computer vision.

## Overview

SegPark uses YOLOv8-Seg (segmentation) to detect and segment parking spots in images. The model provides precise polygon masks for each parking spot, enabling accurate spot localization and centroid calculation.

## Features

- **YOLOv8-Seg Training**: Train custom segmentation models on parking lot images
- **Interactive Inference**: File dialog for easy image selection (macOS)
- **Automatic Model Detection**: Uses the most recently trained model automatically
- **Centroid Calculation**: Computes center points for each detected parking spot
- **Image Preprocessing**: Resize utility for preparing images for CVAT annotation
- **Error Handling**: Comprehensive validation and user-friendly error messages
- **Python 3.11/3.12 Compatible**: Version checks prevent PyTorch incompatibility issues

## Requirements

- **Python**: 3.11 or 3.12 (Python 3.14+ is not yet supported by PyTorch)
- **macOS**: Native file dialog uses AppleScript
- **Dependencies**: ultralytics, torch, torchvision, opencv-python, numpy

## Installation

### Quick Setup

```bash
# Make setup script executable
chmod +x setup.sh

# Run automated setup
./setup.sh
```

### Manual Setup

```bash
# Install Python 3.11 (if needed)
brew install python@3.11

# Create virtual environment
python3.11 -m venv .venv311

# Activate environment
source .venv311/bin/activate

# Install dependencies
pip install ultralytics
```

## Usage

### Training

Train the YOLOv8-Seg model on your parking spot dataset:

```bash
source .venv311/bin/activate
python train_seg.py
```

Training parameters:
- Model: YOLOv8n-seg (nano)
- Epochs: 50
- Image size: 640px
- Task: Segmentation

Output saved to: `runs/segment/train*/weights/best.pt`

### Inference

Run inference on parking lot images:

```bash
source .venv311/bin/activate
python predict_slot.py
```

The script will:
1. Open a file selection dialog
2. Load the most recent trained model automatically
3. Detect parking spots and calculate centroids
4. Display results with spot numbers and center points

**Controls**: Close the window by clicking the close button or pressing any key.

### Image Preprocessing

Resize images to 640px for CVAT annotation:

```bash
source .venv311/bin/activate
python resize_images.py
```

- Input: `samples/raw_images/`
- Output: `dataset/images/train/`
- Maintains aspect ratio with Lanczos resampling

## Dataset Structure

```
dataset/
├── data.yaml          # Dataset configuration
├── train.txt          # Training image list
├── images/
│   └── train/         # Training images
└── labels/
    └── train/         # YOLO segmentation labels (.txt)
```

### Label Format

YOLO segmentation format (one file per image):
```
<class_id> <x1> <y1> <x2> <y2> <x3> <y3> ...
```

Coordinates are normalized (0-1) polygon vertices.

## Scripts

### `train_seg.py`
Trains YOLOv8-Seg model on the parking spot dataset.
- Python version validation
- Automatic model downloading
- Progress visualization

### `predict_slot.py`
Inference script with interactive file selection.
- Native macOS file dialog
- Automatic latest model detection
- Centroid calculation for each spot
- Visual results with annotations

### `resize_images.py`
Preprocesses images for CVAT annotation workflow.
- Resizes to 640px (longest side)
- Preserves aspect ratio
- Skips already-small images

### `setup.sh`
Automated environment setup script.
- Detects Python 3.11/3.12
- Creates virtual environment
- Installs all dependencies

## Troubleshooting

### Python 3.14 Error
```
ERROR: Python 3.14 is not supported by PyTorch/Ultralytics
```
**Solution**: Install Python 3.11 or 3.12:
```bash
brew install python@3.11
python3.11 -m venv .venv311
source .venv311/bin/activate
```

### No Module Named 'ultralytics'
```
ERROR: Missing required module: No module named 'ultralytics'
```
**Solution**: Install dependencies:
```bash
source .venv311/bin/activate
pip install ultralytics
```

### Model Not Found
```
ERROR: Trained model not found
```
**Solution**: Train the model first:
```bash
python train_seg.py
```

### NumPy Compatibility Issue
```
ERROR: numpy 2.x is incompatible with PyTorch 2.2.2
```
**Solution**: Downgrade NumPy:
```bash
pip install "numpy<2.0"
```

## Model Performance

The trained model (YOLOv8n-seg) provides:
- Fast inference (real-time capable)
- Accurate polygon segmentation masks
- Precise centroid calculation
- Low memory footprint

## Acknowledgments

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) - Object detection and segmentation framework
- [CVAT](https://github.com/opencv/cvat) - Computer Vision Annotation Tool

## License

This project uses YOLOv8, which is licensed under AGPL-3.0.
