#!/usr/bin/env python3
"""
Image Resizer for CVAT Upload
Resizes images to speed up CVAT processing while maintaining aspect ratio.
"""

from PIL import Image
import os
import sys

# Configuration
input_folder = "samples/raw_images"
output_folder = "dataset/images/train"
target_size = 640  # Longest side in pixels (use 1280 for higher quality)

def resize_images(input_dir, output_dir, target):
    """
    Resize all images in input_dir and save to output_dir.
    
    Args:
        input_dir: Source directory containing images
        output_dir: Destination directory for resized images
        target: Target size for the longest side (in pixels)
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Check if input directory exists
    if not os.path.exists(input_dir):
        print(f"ERROR: Input folder '{input_dir}' does not exist!")
        print(f"\nPlease create it and add your images:")
        print(f"  mkdir -p {input_dir}")
        sys.exit(1)
    
    # Get list of image files
    image_files = [f for f in os.listdir(input_dir) 
                   if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
    
    if not image_files:
        print(f"WARNING: No images found in '{input_dir}'")
        print("\nSupported formats: .png, .jpg, .jpeg, .bmp, .tiff")
        sys.exit(1)
    
    print(f"Found {len(image_files)} image(s) to resize")
    print(f"Target size: {target}px (longest side)")
    print(f"Output folder: {output_dir}")
    print("=" * 60)
    
    processed = 0
    skipped = 0
    errors = 0
    
    for filename in image_files:
        img_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)
        
        try:
            with Image.open(img_path) as img:
                original_size = img.size
                
                # Calculate new dimensions to keep aspect ratio
                ratio = target / max(img.width, img.height)
                
                # Skip if image is already smaller than target
                if ratio >= 1.0:
                    print(f"⊘ {filename}: Already small enough ({original_size[0]}x{original_size[1]})")
                    # Copy without resizing
                    img.save(output_path, quality=95)
                    skipped += 1
                    continue
                
                new_size = (int(img.width * ratio), int(img.height * ratio))
                
                # Resize using high-quality Lanczos filter
                img_resized = img.resize(new_size, Image.Resampling.LANCZOS)
                
                # Save with high quality
                if filename.lower().endswith('.jpg') or filename.lower().endswith('.jpeg'):
                    img_resized.save(output_path, quality=95, optimize=True)
                else:
                    img_resized.save(output_path)
                
                print(f"✓ {filename}: {original_size[0]}x{original_size[1]} → {new_size[0]}x{new_size[1]}")
                processed += 1
                
        except Exception as e:
            print(f"✗ {filename}: Error - {e}")
            errors += 1
    
    # Summary
    print("=" * 60)
    print(f"Resizing complete!")
    print(f"  Resized: {processed}")
    print(f"  Skipped: {skipped} (already small enough)")
    print(f"  Errors:  {errors}")
    print(f"\nOutput location: {os.path.abspath(output_dir)}")

if __name__ == "__main__":
    print("=" * 60)
    print("Image Resizer for CVAT")
    print("=" * 60)
    print()
    
    # Allow command-line arguments to override defaults
    if len(sys.argv) > 1:
        input_folder = sys.argv[1]
    if len(sys.argv) > 2:
        output_folder = sys.argv[2]
    if len(sys.argv) > 3:
        try:
            target_size = int(sys.argv[3])
        except ValueError:
            print(f"ERROR: Invalid target size '{sys.argv[3]}'. Must be a number.")
            sys.exit(1)
    
    print(f"Configuration:")
    print(f"  Input:  {input_folder}")
    print(f"  Output: {output_folder}")
    print(f"  Size:   {target_size}px")
    print()
    
    resize_images(input_folder, output_folder, target_size)
    
    print("\nTip: For higher quality, use target_size=1280")
    print("Usage: python resize_images.py [input_folder] [output_folder] [target_size]")
