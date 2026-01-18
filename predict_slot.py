#!/usr/bin/env python3
"""
YOLOv8-Seg Inference Script for Parking Spot Detection

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
    print("\nSee README_INSTALLATION.md for detailed instructions.")
    print("=" * 70)
    sys.exit(1)

try:
    from ultralytics import YOLO
    import cv2
    import numpy as np
except ImportError as e:
    print("=" * 70)
    print(f"ERROR: Missing required module: {e}")
    print("=" * 70)
    print("\nPlease install ultralytics:")
    print("   pip install ultralytics")
    print("\nSee README_INSTALLATION.md for detailed instructions.")
    print("=" * 70)
    sys.exit(1)

import os
import glob
import subprocess

# Find the latest training run automatically
def find_latest_model():
    """Find the most recent trained model."""
    segment_runs = glob.glob('runs/segment/train*/weights/best.pt')
    if segment_runs:
        # Sort by modification time, get the most recent
        latest = max(segment_runs, key=os.path.getmtime)
        return latest
    return None

def select_image_file():
    """Open a native file dialog to select an image file."""
    print("\n" + "=" * 70)
    print("Opening file selection dialog...")
    print("Please select an image file from the dialog that should appear.")
    print("(If no dialog appears, press Ctrl+C and check your samples folder)")
    print("=" * 70 + "\n")
    
    try:
        # Get absolute path to samples directory
        samples_dir = os.path.abspath('samples/raw_images')
        if not os.path.exists(samples_dir):
            samples_dir = os.path.expanduser('~')
        
        # Use PowerShell to show Windows file picker
        script = f'''
Add-Type -AssemblyName System.Windows.Forms
$openFileDialog = New-Object System.Windows.Forms.OpenFileDialog
$openFileDialog.InitialDirectory = "{samples_dir}"
$openFileDialog.Filter = "Image Files|*.jpg;*.jpeg;*.png;*.bmp|All Files|*.*"
$openFileDialog.Title = "Select an image file"
if ($openFileDialog.ShowDialog() -eq 'OK') {{
    $openFileDialog.FileName
}}
        '''
        
        result = subprocess.run(
            ['powershell', '-Command', script],
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        
        if result.returncode == 0:
            file_path = result.stdout.strip()
            if file_path:
                print(f"✓ Selected: {file_path}\n")
                return file_path
        else:
            print(f"Warning: Dialog cancelled or error (code {result.returncode})\n")
            if result.stderr:
                print(f"Error: {result.stderr}\n")
    except subprocess.TimeoutExpired:
        print("Warning: File selection timed out after 5 minutes.\n")
    except Exception as e:
        print(f"Warning: Could not open file dialog: {e}\n")
    
    return None

# Model and image paths
model_path = find_latest_model() or 'runs/segment/train/weights/best.pt'

print("=" * 70)
print("MODEL SELECTION")
print("=" * 70)
print(f"Selected model: {model_path}")
print(f"Model file size: {os.path.getsize(model_path) / (1024*1024):.2f} MB")
print(f"Last modified: {os.path.getmtime(model_path)}")
print("=" * 70)
print()

# Check if model exists
if not os.path.exists(model_path):
    print("=" * 70)
    print("ERROR: Trained model not found")
    print("=" * 70)
    print(f"\nModel path: {model_path}")
    
    # Show available models
    available_models = glob.glob('runs/segment/*/weights/best.pt')
    if available_models:
        print("\nAvailable trained models found:")
        for model in available_models:
            print(f"  - {model}")
        print("\nUpdate 'model_path' in the script to use one of these.")
    else:
        print("\nNo trained models found in runs/segment/")
    
    print("\nPlease train the model first:")
    print("   python train_seg.py")
    print("=" * 70)
    sys.exit(1)

# Load the model
print(f"Loading model from: {model_path}")
try:
    model = YOLO(model_path)
    print("✓ Model loaded successfully")
except Exception as e:
    print("=" * 70)
    print("ERROR: Failed to load model")
    print("=" * 70)
    print(f"\nError details: {e}")
    print("\nThe model file may be corrupted. Try retraining:")
    print("   python train_seg.py")
    print("=" * 70)
    sys.exit(1)

def run_inference(img_path, confidence):
    """Run inference and display results"""
    print(f"\nRunning inference on: {os.path.basename(img_path)}")
    print(f"Confidence threshold: {confidence}")
    
    try:
        results = model(img_path, conf=confidence, verbose=False)
        print("✓ Inference completed")
    except Exception as e:
        print(f"✗ Inference failed: {e}")
        return None
    
    parking_spots_found = 0
    for r in results:
        # 1. Visualize the Output
        im_array = r.plot()
        
        # 2. Extract the Data
        if r.masks is not None and len(r.masks) > 0:
            masks = r.masks.xy
            print(f"✓ Found {len(masks)} parking spot(s)")
            
            for i, mask in enumerate(masks, 1):
                if len(mask) < 3:
                    continue
                
                M = cv2.moments(mask)
                if M['m00'] != 0:
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])
                    parking_spots_found += 1
                    
                    # Draw center point and number
                    cv2.circle(im_array, (cx, cy), 10, (0, 0, 255), -1)
                    cv2.putText(im_array, f"#{i}", (cx+15, cy+5), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        else:
            print("⚠ No parking spots detected")
        
        return im_array, parking_spots_found
    
    return None, 0

def display_image(im_array, parking_spots_found, confidence):
    """Display image with resize if needed"""
    max_display_height = 900
    max_display_width = 1600
    h, w = im_array.shape[:2]
    
    if h > max_display_height or w > max_display_width:
        scale = min(max_display_height / h, max_display_width / w)
        new_w = int(w * scale)
        new_h = int(h * scale)
        display_image = cv2.resize(im_array, (new_w, new_h))
    else:
        display_image = im_array
    
    window_name = "Parking Spot Detection"
    
    # Add control info to image
    info_text = [
        f"Detections: {parking_spots_found} | Confidence: {confidence:.2f}",
        "Controls: +/- adjust conf | L load new image | Q quit"
    ]
    y_pos = 30
    for text in info_text:
        cv2.putText(display_image, text, (10, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        y_pos += 30
    
    cv2.imshow(window_name, display_image)

# Interactive loop
img_path = select_image_file() or 'samples/parking_test.jpg'
if not os.path.exists(img_path):
    print("ERROR: Image not found")
    sys.exit(1)

confidence = 0.05
im_array = None
parking_spots_found = 0

print("\n" + "=" * 70)
print("INTERACTIVE MODE")
print("=" * 70)
print("Controls:")
print("  +/=     : Increase confidence by 0.01")
print("  -       : Decrease confidence by 0.01")
print("  L       : Load new image")
print("  Q/ESC   : Quit")
print("=" * 70)

# Initial inference
result = run_inference(img_path, confidence)
if result:
    im_array, parking_spots_found = result

running = True
while running:
    if im_array is not None:
        display_image(im_array, parking_spots_found, confidence)
    
    key = cv2.waitKey(100) & 0xFF
    
    if key == ord('q') or key == ord('Q') or key == 27:  # Q or ESC
        running = False
        
    elif key == ord('+') or key == ord('='):  # Increase confidence
        confidence = min(1.0, confidence + 0.01)
        print(f"\nConfidence: {confidence:.2f}")
        result = run_inference(img_path, confidence)
        if result:
            im_array, parking_spots_found = result
            
    elif key == ord('-') or key == ord('_'):  # Decrease confidence
        confidence = max(0.01, confidence - 0.01)
        print(f"\nConfidence: {confidence:.2f}")
        result = run_inference(img_path, confidence)
        if result:
            im_array, parking_spots_found = result
            
    elif key == ord('l') or key == ord('L'):  # Load new image
        new_path = select_image_file()
        if new_path and os.path.exists(new_path):
            img_path = new_path
            print(f"\nLoaded: {img_path}")
            result = run_inference(img_path, confidence)
            if result:
                im_array, parking_spots_found = result
        else:
            print("No image selected or file not found")
    
    # Check if window was closed (only if we're not loading a new image)
    if key != ord('l') and key != ord('L'):
        try:
            if cv2.getWindowProperty("Parking Spot Detection", cv2.WND_PROP_VISIBLE) < 1:
                running = False
        except:
            pass

cv2.destroyAllWindows()
print("\n" + "=" * 70)
print("Session ended")
print("=" * 70)
