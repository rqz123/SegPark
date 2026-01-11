#!/usr/bin/env python3
"""
YOLOv8-Seg Training Script for Parking Spot Detection

IMPORTANT: Requires Python 3.11 or 3.12 (PyTorch doesn't support 3.14 yet)
See README_INSTALLATION.md for setup instructions.
"""

import sys

# Check Python version
if sys.version_info >= (3, 14):
    print("=" * 70)
    print("ERROR: Python 3.14 is not supported by PyTorch/Ultralytics")
    print("=" * 70)
    print("\nPlease use Python 3.11 or 3.12:")
    print("\n1. Install Python 3.11:")
    print("   brew install python@3.11")
    print("\n2. Create new virtual environment:")
    print("   python3.11 -m venv .venv311")
    print("\n3. Activate it:")
    print("   source .venv311/bin/activate")
    print("\n4. Install ultralytics:")
    print("   pip install ultralytics")
    print("\n5. Run this script again")
    print("\nSee README_INSTALLATION.md for detailed instructions.")
    print("=" * 70)
    sys.exit(1)

try:
    from ultralytics import YOLO
except ImportError:
    print("=" * 70)
    print("ERROR: ultralytics module not found")
    print("=" * 70)
    print("\nPlease install it:")
    print("   pip install ultralytics")
    print("\nSee README_INSTALLATION.md for detailed instructions.")
    print("=" * 70)
    sys.exit(1)

# Load the Segmentation model (Note the '-seg' suffix)
# yolov8n-seg.pt is the Nano Segmentation model (fastest)
print("Loading YOLOv8n-seg model...")
model = YOLO('yolov8n-seg.pt')

# Train
print("Starting training...")
results = model.train(
    data='/Users/richardzhang/Works/SegPark/dataset/data.yaml',
    epochs=50,
    imgsz=640,
    task='segment',  # Explicitly state this is a segmentation task
    plots=True
)

print("\n" + "=" * 70)
print("Training complete!")
print("Best model saved to: runs/segment/train/weights/best.pt")
print("=" * 70)
