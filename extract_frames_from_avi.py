#!/usr/bin/env python3
"""
Extract frames from stereo AVI videos and split left/right images.

This script processes .avi videos from the ASU (car camera recordings)
and extracts frames for creating the UBM test set.

The .avi are at 60FPS, even though they were recorded at 30FPS.
Default sampling: Every 60 frames = 1 second at 60 FPS = 2 seconds real-world time.

Input:  Stereo stitched videos (2560×720, left+right side-by-side)
Output: Individual left and right frames (1280×720 each)

Usage:
    python extract_frames_from_avi.py media/20_11_2025_Rioveggio_Test_LidarTest1.avi --interval 60 --prefix lidar1
    python extract_frames_from_avi.py media/20_11_2025_Rioveggio_Test_LidarTest2.avi --interval 60 --prefix lidar2
"""

import cv2
import os
import argparse
from pathlib import Path


def extract_frames(avi_path, output_dir, frame_interval=60, prefix='frame'):
    """
    Extract frames from AVI and split stereo images.

    Args:
        avi_path: Path to .avi file
        output_dir: Output directory for frames
        frame_interval: Extract every Nth frame (60 = 1 second at 60fps = 2 seconds real-world)
        prefix: Prefix for output filenames (default: 'frame')
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Open video
    print(f"Opening video: {avi_path}")
    cap = cv2.VideoCapture(avi_path)

    if not cap.isOpened():
        print(f"ERROR: Could not open video file: {avi_path}")
        return

    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Video properties:")
    print(f"  Total frames: {total_frames}")
    print(f"  FPS: {fps}")
    print(f"  Resolution: {width}×{height}")
    print(f"  Expected stereo: 2560×720 (2× 1280×720)")
    print()

    # Check if stereo format
    if width != 2560 or height != 720:
        print(f"WARNING: Expected stereo resolution 2560×720, got {width}×{height}")
        print(f"         Will split at midpoint anyway ({width//2}×{height})")
        print()

    # Extract frames
    frame_count = 0
    extracted = 0

    print(f"Extracting every {frame_interval} frames...")
    print(f"Expected output: ~{total_frames // frame_interval} stereo pairs")
    print()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            # Split stereo image down the middle
            height, width = frame.shape[:2]
            mid = width // 2

            left_img = frame[:, :mid]    # Left camera: 0:mid
            right_img = frame[:, mid:]   # Right camera: mid:width

            # Save both images
            left_path = output_path / f'{prefix}_left_{extracted:04d}.jpg'
            right_path = output_path / f'{prefix}_right_{extracted:04d}.jpg'

            cv2.imwrite(str(left_path), left_img)
            cv2.imwrite(str(right_path), right_img)

            extracted += 1

            if extracted % 10 == 0:
                print(f"  Extracted {extracted} stereo pairs...")

        frame_count += 1

    cap.release()

    print()
    print("=" * 50)
    print("EXTRACTION COMPLETE")
    print("=" * 50)
    print(f"Total frames processed: {frame_count}")
    print(f"Stereo pairs extracted: {extracted}")
    print(f"Total images saved: {extracted * 2}")
    print(f"Output directory: {output_path}")
    print()
    print("Next steps:")
    print("  1. Upload images to Roboflow")
    print("  2. Annotate cones (5 classes)")
    print("  3. Export as YOLOv11 format")
    print("  4. Evaluate models on UBM test set")


def main():
    parser = argparse.ArgumentParser(
        description="Extract frames from stereo AVI videos",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Extract every 60 frames (1 second at 60fps = 2 seconds real-world time)
    python extract_frames_from_avi.py media/20_11_2025_Rioveggio_Test_LidarTest1.avi --output ubm_test_set/images --prefix lidar1

    # Extract more frequently (every 30 frames = 0.5 seconds at 60fps)
    python extract_frames_from_avi.py media/20_11_2025_Rioveggio_Test_LidarTest1.avi --output ubm_test_set/images --interval 30 --prefix lidar1

    # Process both test videos
    python extract_frames_from_avi.py media/20_11_2025_Rioveggio_Test_LidarTest1.avi --output ubm_test_set/images --prefix lidar1
    python extract_frames_from_avi.py media/20_11_2025_Rioveggio_Test_LidarTest2.avi --output ubm_test_set/images --prefix lidar2
        """
    )

    parser.add_argument('avi_path', help='Path to input .avi file')
    parser.add_argument('--output', '-o', default='ubm_test_set/images',
                        help='Output directory for extracted frames (default: ubm_test_set/images)')
    parser.add_argument('--interval', '-i', type=int, default=60,
                        help='Extract every Nth frame (default: 60 = 1 second at 60fps = 2 seconds real-world time)')
    parser.add_argument('--prefix', '-p', default='frame',
                        help='Prefix for output filenames (default: frame)')

    args = parser.parse_args()

    # Validate input file
    if not Path(args.avi_path).exists():
        print(f"ERROR: File not found: {args.avi_path}")
        return

    # Extract frames
    extract_frames(args.avi_path, args.output, args.interval, args.prefix)


if __name__ == "__main__":
    main()
