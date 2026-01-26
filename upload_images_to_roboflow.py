#!/usr/bin/env python3
"""
Bulk upload images to Roboflow for UBM Test Set annotation.

Uploads all extracted frames from ubm_test_set/images/ to Roboflow project.
Each image is tagged with metadata (camera side, lidar test) for organization.

Usage:
    python upload_to_roboflow.py
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from roboflow import Roboflow

# Load API key from .env
load_dotenv()
ROBOFLOW_API_KEY = os.getenv('ROBOFLOW_API_KEY')

if not ROBOFLOW_API_KEY:
    print("ERROR: ROBOFLOW_API_KEY not found in .env file")
    sys.exit(1)

# Configuration
WORKSPACE_ID = 'fsae-okyoe'  # Your Roboflow workspace (from URL)
PROJECT_ID = 'ml4cv_project'  # Your project name
PROJECT_TYPE = 'object-detection'
BATCH_NAME = 'UBM-Rioveggio-Test-Nov2025'
IMAGES_DIR = 'ubm_test_set/images'

def create_or_get_project(rf):
    """Get existing project (must already exist on Roboflow)."""
    try:
        # Try to access existing project
        print(f"Looking for project: {PROJECT_ID} in workspace: {WORKSPACE_ID}")
        project = rf.workspace(WORKSPACE_ID).project(PROJECT_ID)
        print(f"✅ Found existing project: {PROJECT_ID}")
        return project
    except Exception as e:
        print(f"\n❌ ERROR: Could not find project '{PROJECT_ID}'")
        print(f"Error details: {e}")
        print("\nPossible issues:")
        print(f"  1. Project name mismatch - check exact name on Roboflow website")
        print(f"  2. Project URL format: https://app.roboflow.com/{WORKSPACE_ID}/PROJECT-NAME")
        print()
        print("Please provide the exact project name from your Roboflow URL")
        print("Example: If URL is https://app.roboflow.com/FSAE/my-cone-detector")
        print("         Then project name is: my-cone-detector")
        print()

        # Try to list available projects
        try:
            workspace = rf.workspace(WORKSPACE_ID)
            print(f"Available projects in workspace '{WORKSPACE_ID}':")
            # Note: The Roboflow API doesn't have a direct list_projects() method
            # User needs to manually check and provide correct name
        except:
            pass

        sys.exit(1)

def get_image_metadata(filename):
    """Extract metadata from filename (e.g., lidar1_left_0000.jpg)."""
    parts = filename.stem.split('_')

    if len(parts) >= 3:
        lidar = parts[0]  # lidar1 or lidar2
        camera = parts[1]  # left or right
        frame_num = parts[2]  # 0000, 0001, etc.

        return {
            'lidar': lidar,
            'camera': camera,
            'frame': frame_num,
            'tags': [lidar, camera, f'{lidar}_{camera}']
        }

    return {'tags': []}

def upload_images(project, images_dir):
    """Upload all images from directory to Roboflow."""
    images_path = Path(images_dir)

    if not images_path.exists():
        print(f"ERROR: Images directory not found: {images_dir}")
        print("Run extract_frames_from_avi.py first!")
        sys.exit(1)

    # Get all jpg images
    image_files = sorted(images_path.glob('*.jpg'))

    if not image_files:
        print(f"ERROR: No images found in {images_dir}")
        sys.exit(1)

    total_images = len(image_files)
    print(f"\n{'='*60}")
    print(f"UPLOADING {total_images} IMAGES TO ROBOFLOW")
    print(f"{'='*60}")
    print(f"Project: {PROJECT_ID}")
    print(f"Batch: {BATCH_NAME}")
    print(f"Split: test (unlabeled images for annotation)")
    print()

    successful = 0
    failed = 0

    for idx, image_path in enumerate(image_files, 1):
        try:
            # Extract metadata from filename
            metadata = get_image_metadata(image_path)

            # Upload image with metadata
            project.upload(
                image_path=str(image_path),
                batch_name=BATCH_NAME,
                split='test',  # Upload as test split (unlabeled for annotation)
                num_retry_uploads=3,
                tag_names=metadata['tags'],
            )

            successful += 1

            # Progress indicator
            if idx % 10 == 0 or idx == total_images:
                print(f"  Progress: {idx}/{total_images} ({successful} successful, {failed} failed)")

        except Exception as e:
            failed += 1
            print(f"  ⚠️  Failed to upload {image_path.name}: {e}")

    print()
    print(f"{'='*60}")
    print(f"UPLOAD COMPLETE")
    print(f"{'='*60}")
    print(f"Successful: {successful}/{total_images}")
    print(f"Failed: {failed}/{total_images}")
    print()

    if successful > 0:
        print(f"✅ Images uploaded to Roboflow!")
        print()
        print(f"Next steps:")
        print(f"  1. Go to: https://app.roboflow.com/{WORKSPACE_ID}/{PROJECT_ID}")
        print(f"  2. Review uploaded images in batch: {BATCH_NAME}")
        print(f"  3. Configure 5 classes (if not already):")
        print(f"     - blue_cone")
        print(f"     - yellow_cone")
        print(f"     - orange_cone")
        print(f"     - large_orange_cone")
        print(f"     - unknown_cone")
        print(f"  4. Start annotating cones!")
        print(f"  5. Export as YOLOv11 format when done")

    return successful, failed

def main():
    print()
    print(f"{'='*60}")
    print(f"ROBOFLOW BULK IMAGE UPLOAD")
    print(f"{'='*60}")
    print()
    print(f"API Key: {ROBOFLOW_API_KEY[:10]}... (loaded from .env)")
    print(f"Workspace: {WORKSPACE_ID}")
    print(f"Project: {PROJECT_ID}")
    print(f"Images directory: {IMAGES_DIR}")
    print()

    # Initialize Roboflow
    print("Connecting to Roboflow...")
    try:
        rf = Roboflow(api_key=ROBOFLOW_API_KEY)
        print("✅ Connected to Roboflow")
    except Exception as e:
        print(f"ERROR: Failed to connect to Roboflow: {e}")
        print("Check your API key in .env file")
        sys.exit(1)

    # Create or get project
    print()
    project = create_or_get_project(rf)

    # Upload images
    successful, failed = upload_images(project, IMAGES_DIR)

    if failed > 0:
        print(f"\n⚠️  {failed} images failed to upload. Check errors above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
