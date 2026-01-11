#!/bin/bash

# Quick setup script for YOLOv8-Seg on Python 3.11/3.12

echo "=========================================="
echo "YOLOv8-Seg Setup Script"
echo "=========================================="
echo ""

# Check for Python 3.11
if command -v python3.11 &> /dev/null; then
    PYTHON_CMD="python3.11"
    VENV_NAME=".venv311"
    echo "✓ Found Python 3.11"
elif command -v python3.12 &> /dev/null; then
    PYTHON_CMD="python3.12"
    VENV_NAME=".venv312"
    echo "✓ Found Python 3.12"
else
    echo "✗ Python 3.11 or 3.12 not found!"
    echo ""
    echo "Please install Python 3.11:"
    echo "  brew install python@3.11"
    echo ""
    exit 1
fi

echo ""
echo "Creating virtual environment with $PYTHON_CMD..."
$PYTHON_CMD -m venv $VENV_NAME

echo ""
echo "Activating virtual environment..."
source $VENV_NAME/bin/activate

echo ""
echo "Installing ultralytics..."
pip install --upgrade pip
pip install ultralytics

echo ""
echo "=========================================="
echo "✓ Setup Complete!"
echo "=========================================="
echo ""
echo "To activate the environment in the future, run:"
echo "  source $VENV_NAME/bin/activate"
echo ""
echo "Then you can run:"
echo "  python train_seg.py      # To train"
echo "  python predict_slot.py   # To run inference"
echo ""
