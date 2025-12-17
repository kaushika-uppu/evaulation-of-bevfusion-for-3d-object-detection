#!/usr/bin/env python3
"""
Create video from BEV detection visualization images.
"""

import os
import cv2
import argparse
from pathlib import Path
import glob

def create_video_from_images(image_dir: str, output_path: str, fps: float = 2.0, 
                            pattern: str = "frame_*.png"):
    """
    Create a video from a sequence of images.
    
    Args:
        image_dir: Directory containing images
        output_path: Output video file path
        fps: Frames per second for the video
        pattern: Glob pattern to match image files
    """
    image_dir = Path(image_dir)
    
    # Find all images matching the pattern
    image_files = sorted(image_dir.glob(pattern))
    
    if len(image_files) == 0:
        print(f"Error: No images found matching pattern '{pattern}' in {image_dir}")
        return False
    
    print(f"Found {len(image_files)} images")
    
    # Read first image to get dimensions
    first_image = cv2.imread(str(image_files[0]))
    if first_image is None:
        print(f"Error: Could not read first image: {image_files[0]}")
        return False
    
    height, width, channels = first_image.shape
    print(f"Image dimensions: {width}x{height}")
    
    # Define codec and create VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    if not out.isOpened():
        print(f"Error: Could not open video writer for {output_path}")
        return False
    
    # Write images to video
    for i, img_path in enumerate(image_files):
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"Warning: Could not read image {img_path}, skipping")
            continue
        
        # Resize if dimensions don't match
        if img.shape[:2] != (height, width):
            img = cv2.resize(img, (width, height))
        
        out.write(img)
        
        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{len(image_files)} images")
    
    # Release everything
    out.release()
    print(f"\nVideo created successfully: {output_path}")
    print(f"  Frames: {len(image_files)}")
    print(f"  FPS: {fps}")
    print(f"  Duration: {len(image_files) / fps:.2f} seconds")
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description='Create video from BEV detection visualization images',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create video from resnet50_all directory
  python3 create_video.py visualizations/resnet50_all output.mp4

  # Create video with custom FPS
  python3 create_video.py visualizations/resnet50_all output.mp4 --fps 5.0

  # Create video with different image pattern
  python3 create_video.py visualizations/resnet50_all output.mp4 --pattern "*.png"
        """
    )
    parser.add_argument('image_dir', type=str,
                       help='Directory containing images')
    parser.add_argument('output', type=str,
                       help='Output video file path (e.g., output.mp4)')
    parser.add_argument('--fps', type=float, default=2.0,
                       help='Frames per second (default: 2.0)')
    parser.add_argument('--pattern', type=str, default='frame_*.jpg',
                       help='Glob pattern to match images (default: frame_*.jpg)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.image_dir):
        print(f"Error: Image directory not found: {args.image_dir}")
        return
    
    create_video_from_images(args.image_dir, args.output, args.fps, args.pattern)


if __name__ == '__main__':
    main()

