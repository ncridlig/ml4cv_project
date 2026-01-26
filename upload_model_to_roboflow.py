#!/usr/bin/env python3
"""
Upload YOLO11 trained model weights to Roboflow for Label Assist.

This enables auto-annotation of new images using your trained YOLO11 model.
Reduces manual annotation time by ~95%.

Roboflow Requirements:
- YOLOv11 models must be trained using ultralytics

Usage:
    python upload_model_to_roboflow.py
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
WORKSPACE_ID = 'fsae-okyoe'
PROJECT_ID = 'ml4cv_project'
MODEL_PATH = 'runs/detect/runs/baseline/yolov11n_300ep_baseline8/weights'
MODEL_TYPE = 'yolov11'  # YOLO11 format (officially supported by Roboflow)
MODEL_FILE = 'best.pt'

def upload_model_versioned(rf, version_number=1):
    """
    Upload model to specific dataset version (versioned deployment).

    Args:
        rf: Roboflow instance
        version_number: Dataset version to link model to
    """
    print(f"\n{'='*60}")
    print(f"VERSIONED DEPLOYMENT (Dataset Version {version_number})")
    print(f"{'='*60}\n")

    try:
        # Access project
        project = rf.workspace(WORKSPACE_ID).project(PROJECT_ID)
        print(f"✅ Found project: {PROJECT_ID}")

        # Access specific version
        version = project.version(version_number)
        print(f"✅ Found dataset version: {version_number}")

        # Check if model exists
        model_file = Path(MODEL_PATH) / MODEL_FILE
        if not model_file.exists():
            print(f"\n❌ ERROR: Model file not found: {model_file}")
            print(f"Expected location: runs/detect/runs/baseline/yolov11n_300ep_baseline8/weights/best.pt")
            sys.exit(1)

        print(f"✅ Found model weights: {model_file}")
        print(f"   Size: {model_file.stat().st_size / (1024*1024):.1f} MB")
        print()

        # Deploy model
        print(f"Uploading YOLO11 model to Roboflow...")
        print(f"  Model type: {MODEL_TYPE}")
        print(f"  Model path: {MODEL_PATH}")
        print(f"  Model file: {MODEL_FILE}")
        print()
        print("⏳ This may take 5-10 minutes (uploading + processing)...")
        print()

        version.deploy(MODEL_TYPE, MODEL_PATH, MODEL_FILE)

        print()
        print(f"{'='*60}")
        print(f"✅ MODEL UPLOAD COMPLETE")
        print(f"{'='*60}\n")

        print(f"Your YOLO11 model is now available in Roboflow!")
        print()
        print(f"🎯 Next Steps - Use Label Assist:")
        print(f"  1. Go to: https://app.roboflow.com/{WORKSPACE_ID}/{PROJECT_ID}/annotate")
        print(f"  2. Open any image from the UBM batch")
        print(f"  3. Click the magic wand icon (✨) in the toolbar")
        print(f"  4. Select 'Your Models' tab")
        print(f"  5. Choose your YOLO11 model")
        print(f"  6. Click 'Run' - it will auto-annotate the image!")
        print(f"  7. Review and correct predictions")
        print(f"  8. Move to next image - Label Assist continues automatically")
        print()
        print(f"💡 Expected time savings:")
        print(f"   Without Label Assist: 2-3 hours for 96 images")
        print(f"   With Label Assist: ~30-45 minutes (just corrections)")
        print(f"   Time saved: ~2 hours (95% faster!)")
        print()

    except Exception as e:
        print(f"\n❌ ERROR: Failed to upload model: {e}")
        print()
        print("Troubleshooting:")
        print("  1. Check dataset version exists on Roboflow")
        print("  2. Ensure model path is correct")
        print("  3. Verify API key has upload permissions")
        sys.exit(1)

def upload_model_versionless(rf, project_ids):
    """
    Upload model at workspace level (versionless deployment).
    Shares model across multiple projects.

    Args:
        rf: Roboflow instance
        project_ids: List of project IDs to deploy to
    """
    print(f"\n{'='*60}")
    print(f"VERSIONLESS DEPLOYMENT (Workspace Level)")
    print(f"{'='*60}\n")

    try:
        workspace = rf.workspace(WORKSPACE_ID)
        print(f"✅ Found workspace: {WORKSPACE_ID}")

        # Check if model exists
        model_file = Path(MODEL_PATH) / MODEL_FILE
        if not model_file.exists():
            print(f"\n❌ ERROR: Model file not found: {model_file}")
            sys.exit(1)

        print(f"✅ Found model weights: {model_file}")
        print()

        # Deploy model
        print(f"Uploading YOLO11 model to workspace...")
        print(f"  Projects: {', '.join(project_ids)}")
        print()
        print("⏳ This may take 5-10 minutes...")
        print()

        workspace.deploy_model(
            model_type=MODEL_TYPE,
            model_path=MODEL_PATH,
            project_ids=project_ids,
            model_name="yolo11n-ubm-production",
            filename=MODEL_FILE
        )

        print()
        print(f"✅ MODEL UPLOAD COMPLETE (Workspace Level)")
        print()

    except Exception as e:
        print(f"\n❌ ERROR: Failed to upload model: {e}")
        sys.exit(1)

def main():
    print()
    print(f"{'='*60}")
    print(f"UPLOAD YOLO11 MODEL TO ROBOFLOW")
    print(f"{'='*60}\n")

    print(f"Configuration:")
    print(f"  API Key: {ROBOFLOW_API_KEY[:10]}... (from .env)")
    print(f"  Workspace: {WORKSPACE_ID}")
    print(f"  Project: {PROJECT_ID}")
    print(f"  Model: YOLOv11n Baseline (0.7065 mAP50 on test set)")
    print(f"  Weights: {MODEL_PATH}/{MODEL_FILE}")
    print()

    # Check roboflow version
    try:
        import roboflow
        print(f"  Roboflow version: {roboflow.__version__}")
        if roboflow.__version__ < "1.1.53":
            print(f"  ⚠️  Warning: Recommended version >= 1.1.53")
            print(f"     Run: pip install --upgrade roboflow")
    except:
        pass

    print()

    # Connect to Roboflow
    print("Connecting to Roboflow...")
    try:
        rf = Roboflow(api_key=ROBOFLOW_API_KEY)
        print("✅ Connected to Roboflow")
    except Exception as e:
        print(f"❌ ERROR: Failed to connect: {e}")
        sys.exit(1)

    # Ask user which deployment method
    print()
    print("Choose deployment method:")
    print("  1. Versioned (links to dataset version 1) - RECOMMENDED")
    print("  2. Versionless (workspace level, shares across projects)")
    print()

    choice = input("Enter choice (1 or 2, default=1): ").strip() or "1"

    if choice == "1":
        upload_model_versioned(rf, version_number=1)
    elif choice == "2":
        upload_model_versionless(rf, project_ids=[PROJECT_ID])
    else:
        print("Invalid choice")
        sys.exit(1)

if __name__ == "__main__":
    main()
