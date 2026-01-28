#!/usr/bin/env python3
"""
Visualize fsoco-ubm annotations with YOLO predictions.

Usage:
    # Visualize annotations only (ground truth)
    python3 visualize_fsoco_ubm.py --annotations-only

    # Visualize predictions from a model
    python3 visualize_fsoco_ubm.py --model runs/detect/runs/baseline/yolov11n_300ep_FSOCO_correct/weights/best.pt

    # Visualize both annotations and predictions side-by-side
    python3 visualize_fsoco_ubm.py --model runs/detect/runs/yolo26/yolo26n_300ep_FSOCO/weights/best.pt --show-both
"""
import argparse
import cv2
from pathlib import Path
from ultralytics import YOLO
import random

# Class names and colors
CLASS_NAMES = {
    0: 'blue_cone',
    1: 'large_orange_cone',
    2: 'orange_cone',
    3: 'unknown_cone',
    4: 'yellow_cone'
}

CLASS_COLORS = {
    0: (255, 0, 0),      # Blue
    1: (0, 140, 255),    # Orange (large)
    2: (0, 165, 255),    # Orange
    3: (128, 128, 128),  # Gray (unknown)
    4: (0, 255, 255)     # Yellow
}

def draw_annotations(image, label_file):
    """Draw ground truth annotations on image."""
    img = image.copy()
    h, w = img.shape[:2]

    if not label_file.exists():
        return img

    with open(label_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue

            cls_id = int(parts[0])
            x_center = float(parts[1]) * w
            y_center = float(parts[2]) * h
            width = float(parts[3]) * w
            height = float(parts[4]) * h

            # Calculate bounding box corners
            x1 = int(x_center - width / 2)
            y1 = int(y_center - height / 2)
            x2 = int(x_center + width / 2)
            y2 = int(y_center + height / 2)

            # Draw bbox
            color = CLASS_COLORS.get(cls_id, (255, 255, 255))
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

            # Draw label
            label = CLASS_NAMES.get(cls_id, f'class_{cls_id}')
            cv2.putText(img, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return img

def draw_predictions(image, results, conf_threshold=0.25):
    """Draw model predictions on image."""
    img = image.copy()

    if len(results) == 0 or results[0].boxes is None:
        return img

    boxes = results[0].boxes
    for i in range(len(boxes)):
        box = boxes.xyxy[i].cpu().numpy()
        conf = boxes.conf[i].cpu().numpy()
        cls_id = int(boxes.cls[i].cpu().numpy())

        if conf < conf_threshold:
            continue

        x1, y1, x2, y2 = map(int, box)
        color = CLASS_COLORS.get(cls_id, (255, 255, 255))

        # Draw bbox
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        # Draw label with confidence
        label = f'{CLASS_NAMES.get(cls_id, f"class_{cls_id}")} {conf:.2f}'
        cv2.putText(img, label, (x1, y1 - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return img

def main():
    parser = argparse.ArgumentParser(description='Visualize fsoco-ubm annotations')
    parser.add_argument('--model', type=str, default=None,
                       help='Path to model weights for predictions')
    parser.add_argument('--conf', type=float, default=0.25,
                       help='Confidence threshold for predictions')
    parser.add_argument('--annotations-only', action='store_true',
                       help='Show only ground truth annotations')
    parser.add_argument('--show-both', action='store_true',
                       help='Show annotations and predictions side-by-side')
    parser.add_argument('--num-images', type=int, default=20,
                       help='Number of random images to visualize')
    parser.add_argument('--save-dir', type=str, default='runs/visualizations',
                       help='Directory to save visualizations')

    args = parser.parse_args()

    # Setup paths
    data_dir = Path('datasets/ml4cv_project-2/test')
    images_dir = data_dir / 'images'
    labels_dir = data_dir / 'labels'
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Get all images
    image_files = sorted(list(images_dir.glob('*.jpg')))
    print(f"Found {len(image_files)} images in {images_dir}")

    # Randomly sample images
    if len(image_files) > args.num_images:
        image_files = random.sample(image_files, args.num_images)
        print(f"Randomly selected {args.num_images} images")

    # Load model if provided
    model = None
    if args.model and not args.annotations_only:
        print(f"Loading model: {args.model}")
        model = YOLO(args.model)
        print(f"Model loaded. Confidence threshold: {args.conf}")

    print()
    print("=" * 80)
    print("VISUALIZING fsoco-ubm DATASET")
    print("=" * 80)
    print()
    print("Controls:")
    print("  - Press SPACE to go to next image")
    print("  - Press 'q' to quit")
    print("  - Press 's' to save current image")
    print()

    for idx, img_file in enumerate(image_files):
        print(f"\nImage {idx+1}/{len(image_files)}: {img_file.name}")

        # Load image
        img = cv2.imread(str(img_file))
        if img is None:
            print(f"  ❌ Failed to load image")
            continue

        # Find corresponding label file
        label_file = labels_dir / img_file.name.replace('.jpg', '.txt')

        # Count annotations
        num_annotations = 0
        if label_file.exists():
            with open(label_file, 'r') as f:
                num_annotations = len(f.readlines())

        print(f"  Ground truth annotations: {num_annotations}")

        # Draw annotations
        img_annotated = draw_annotations(img, label_file)

        # Draw predictions if model provided
        if model:
            results = model(img, conf=args.conf, verbose=False)
            img_predictions = draw_predictions(img, results, args.conf)

            num_predictions = len(results[0].boxes) if len(results) > 0 and results[0].boxes is not None else 0
            print(f"  Model predictions: {num_predictions}")

            # Calculate metrics
            if num_annotations > 0:
                recall_approx = min(num_predictions / num_annotations, 1.0)
                print(f"  Approximate recall: {recall_approx:.2f} ({num_predictions}/{num_annotations})")

        # Display
        if args.show_both and model:
            # Side-by-side comparison
            combined = cv2.hconcat([img_annotated, img_predictions])
            cv2.putText(combined, "Ground Truth", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            cv2.putText(combined, "Predictions", (img.shape[1] + 10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            display_img = combined
            save_name = f"comparison_{idx:03d}_{img_file.stem}.jpg"
        elif model:
            # Predictions only
            display_img = img_predictions
            save_name = f"predictions_{idx:03d}_{img_file.stem}.jpg"
        else:
            # Annotations only
            display_img = img_annotated
            save_name = f"annotations_{idx:03d}_{img_file.stem}.jpg"

        # Resize for display if too large
        max_width = 1080
        if display_img.shape[1] > max_width:
            scale = max_width / display_img.shape[1]
            new_height = int(display_img.shape[0] * scale)
            display_img = cv2.resize(display_img, (max_width, new_height))

        # Show image
        cv2.imshow('fsoco-ubm Visualization', display_img)

        key = cv2.waitKey(0)
        if key == ord('q'):
            print("\nQuitting...")
            break
        elif key == ord('s'):
            save_path = save_dir / save_name
            cv2.imwrite(str(save_path), display_img)
            print(f"  💾 Saved to: {save_path}")

    cv2.destroyAllWindows()
    print()
    print("=" * 80)
    print(f"Visualization complete. Images saved to: {save_dir}")
    print("=" * 80)

if __name__ == '__main__':
    main()
