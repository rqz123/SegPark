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
    """Open a native file dialog to select an image file using AppleScript."""
    print("\n" + "=" * 70)
    print("Opening file selection dialog...")
    print("Please select an image file from the dialog that should appear.")
    print("(If no dialog appears, press Ctrl+C and check your samples folder)")
    print("=" * 70 + "\n")
    
    try:
        # Get absolute path to samples directory
        samples_dir = os.path.abspath('../PilotPark2/samples')
        if not os.path.exists(samples_dir):
            samples_dir = os.path.expanduser('~')
        
        # Use AppleScript to show native macOS file picker
        script = f'''
        tell application "System Events"
            activate
            set imageFile to choose file with prompt "Select an image file:" of type {{"public.image"}} default location POSIX file "{samples_dir}"
            return POSIX path of imageFile
        end tell
        '''
        
        result = subprocess.run(
            ['osascript', '-e', script],
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        
        if result.returncode == 0:
            file_path = result.stdout.strip()
            if file_path:
                print(f"✓ Selected: {file_path}\n")
                return file_path
        elif result.returncode == -128:
            print("Dialog cancelled by user.\n")
        else:
            print(f"Warning: Dialog error (code {result.returncode})\n")
            if result.stderr:
                print(f"Error: {result.stderr}\n")
    except subprocess.TimeoutExpired:
        print("Warning: File selection timed out after 5 minutes.\n")
    except Exception as e:
        print(f"Warning: Could not open file dialog: {e}\n")
    
    return None

# Model and image paths
model_path = find_latest_model() or 'runs/segment/train/weights/best.pt'

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

# Select image file with dialog
img_path = select_image_file() or 'samples/parking_test.jpg'

# Check if test image exists
if not os.path.exists(img_path):
    print("=" * 70)
    print("ERROR: Test image not found")
    sys.exit(1)

# Inference
print(f"\nRunning inference on: {img_path}")
try:
    results = model(img_path, conf=0.1)  # Set low confidence threshold to detect all spots
    print("✓ Inference completed")
except Exception as e:
    print("=" * 70)
    print("ERROR: Inference failed")
    print("=" * 70)
    print(f"\nError details: {e}")
    print("\nThe image may be corrupted or in an unsupported format.")
    print("=" * 70)
    sys.exit(1)

# Process results
parking_spots_found = 0
for r in results:
    try:
        # 1. Visualize the Output (Draws the filled polygon on the image)
        im_array = r.plot() 
        
        # 2. Extract the Data for your Algorithm
        if r.masks is not None and len(r.masks) > 0:
            # Get the polygon points (x, y) coordinates of the area contour
            masks = r.masks.xy 
            
            print(f"\n✓ Found {len(masks)} parking spot(s)")
            
            for i, mask in enumerate(masks, 1):
                try:
                    # 'mask' is an array of [[x1, y1], [x2, y2], ...]
                    if len(mask) < 3:
                        print(f"  ⚠ Spot {i}: Invalid mask (too few points)")
                        continue
                    
                    # Calculate Centroid (Center of the parking spot)
                    M = cv2.moments(mask)
                    if M['m00'] != 0:
                        cx = int(M['m10'] / M['m00'])
                        cy = int(M['m01'] / M['m00'])
                        
                        print(f"  Spot {i}: Center at ({cx}, {cy})")
                        parking_spots_found += 1
                        
                        # Draw the specific center point
                        cv2.circle(im_array, (cx, cy), 10, (0, 0, 255), -1)
                        # Optionally add spot number
                        cv2.putText(im_array, f"#{i}", (cx+15, cy+5), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    else:
                        print(f"  ⚠ Spot {i}: Could not calculate centroid (zero area)")
                        
                except Exception as e:
                    print(f"  ✗ Error processing spot {i}: {e}")
                    continue
        else:
            print("\n⚠ No parking spots detected in the image")
            print("\nPossible reasons:")
            print("  - Model needs more training")
            print("  - Image quality is poor")
            print("  - No parking spots visible in this image")
            print("  - Detection threshold is too high")
        
        # Display results
        try:
            window_name = "Parking Spot Detection"
            cv2.imshow(window_name, im_array)
            print(f"\n{'='*70}")
            print(f"Summary: {parking_spots_found} parking spot(s) detected")
            print("Close the window when done viewing (or press any key)...")
            print(f"{'='*70}")
            
            # Wait for window close or key press
            while True:
                if cv2.waitKey(100) >= 0:  # Any key pressed
                    break
                if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:  # Window closed
                    break
            
            cv2.destroyAllWindows()
        except Exception as e:
            print(f"\n⚠ Could not display image window: {e}")
            print("Saving result to file instead...")
            output_path = 'result_output.jpg'
            cv2.imwrite(output_path, im_array)
            print(f"✓ Result saved to: {output_path}")
            
    except Exception as e:
        print("=" * 70)
        print("ERROR: Failed to process results")
        print("=" * 70)
        print(f"\nError details: {e}")
        print("=" * 70)
        sys.exit(1)
