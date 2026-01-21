"""
Verify YOLO Dataset Label Correctness
Check label format, coordinate ranges, and visualize bboxes on compressed measurements
"""

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import cv2
from pathlib import Path
import sys
sys.path.insert(0, 'src/phase5')
sys.path.insert(0, 'src/phase1')


def validate_yolo_label_format(label_file):
    """Validate YOLO label format"""
    errors = []
    labels = []

    with open(label_file, 'r') as f:
        for line_num, line in enumerate(f, 1):
            parts = line.strip().split()

            # Check format: class_id x_center y_center width height
            if len(parts) != 5:
                errors.append(
                    f"Line {line_num}: Expected 5 values, got {len(parts)}")
                continue

            try:
                class_id = int(parts[0])
                x_center, y_center, width, height = map(float, parts[1:])

                # Check class_id range (0-13 for 14 classes)
                if class_id < 0 or class_id > 13:
                    errors.append(
                        f"Line {line_num}: Invalid class_id {class_id} (should be 0-13)")

                # Check normalized coordinates [0, 1]
                if not (0 <= x_center <= 1):
                    errors.append(
                        f"Line {line_num}: x_center {x_center} out of range [0,1]")
                if not (0 <= y_center <= 1):
                    errors.append(
                        f"Line {line_num}: y_center {y_center} out of range [0,1]")
                if not (0 < width <= 1):
                    errors.append(
                        f"Line {line_num}: width {width} out of range (0,1]")
                if not (0 < height <= 1):
                    errors.append(
                        f"Line {line_num}: height {height} out of range (0,1]")

                labels.append((class_id, x_center, y_center, width, height))

            except ValueError as e:
                errors.append(f"Line {line_num}: Parse error - {e}")

    return labels, errors


def denormalize_bbox(bbox, img_width, img_height):
    """Convert normalized YOLO bbox to pixel coordinates"""
    class_id, x_center, y_center, width, height = bbox

    # Convert to pixel coordinates
    x_center_px = x_center * img_width
    y_center_px = y_center * img_height
    width_px = width * img_width
    height_px = height * img_height

    # Convert to corner coordinates
    x_min = x_center_px - width_px / 2
    y_min = y_center_px - height_px / 2
    x_max = x_center_px + width_px / 2
    y_max = y_center_px + height_px / 2

    return class_id, int(x_min), int(y_min), int(x_max), int(y_max)


def visualize_labels_on_image(img_path, label_path, output_path=None):
    """Visualize YOLO labels on compressed measurement"""

    # Read image
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"❌ Failed to load image: {img_path}")
        return False

    img_height, img_width = img.shape[:2]

    # Parse labels
    labels, errors = validate_yolo_label_format(label_path)

    if errors:
        print(f"\n⚠️  Label Errors in {label_path.name}:")
        for error in errors[:5]:  # Show first 5 errors
            print(f"   - {error}")
        if len(errors) > 5:
            print(f"   ... and {len(errors)-5} more errors")
        return False

    # Class names
    class_names = [
        'Speed30', 'Speed60', 'Speed90',
        'NoOvertake', 'NoOvertakeTruck',
        'RightOfWay', 'Priority',
        'GiveWay', 'Stop', 'NoEntry',
        'NoEntryTruck', 'Roundabout', 'EndNoOvertake',
        'EndNoOvertakeTruck'
    ]

    # Draw bboxes
    img_vis = img.copy()

    for bbox in labels:
        class_id, x_min, y_min, x_max, y_max = denormalize_bbox(
            bbox, img_width, img_height)

        # Draw rectangle
        color = (0, 255, 0)  # Green
        cv2.rectangle(img_vis, (x_min, y_min), (x_max, y_max), color, 2)

        # Draw label
        label_text = f"{class_names[class_id]}"
        cv2.putText(img_vis, label_text, (x_min, y_min - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Display using matplotlib
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Original image
    axes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axes[0].set_title(f"Original Measurement\n{img_path.name}", fontsize=10)
    axes[0].axis('off')

    # With bboxes
    axes[1].imshow(cv2.cvtColor(img_vis, cv2.COLOR_BGR2RGB))
    axes[1].set_title(
        f"With Ground Truth Labels ({len(labels)} signs)\n{label_path.name}", fontsize=10)
    axes[1].axis('off')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"   💾 Saved visualization: {output_path}")
    else:
        plt.savefig('outputs/label_verification.png',
                    dpi=150, bbox_inches='tight')
        print(f"   💾 Saved visualization: outputs/label_verification.png")

    plt.close()

    return True


def verify_dataset(dataset_dir, num_samples=5):
    """Verify multiple samples from dataset"""

    dataset_path = Path(dataset_dir)
    train_img_dir = dataset_path / 'images' / 'train'
    train_label_dir = dataset_path / 'labels' / 'train'

    if not train_img_dir.exists():
        print(f"❌ Dataset not found: {dataset_dir}")
        return

    # Get all images
    images = sorted(train_img_dir.glob('*.jpg'))

    print(f"\n{'='*70}")
    print(f"📋 Verifying YOLO Dataset: {dataset_dir}")
    print(f"{'='*70}")
    print(f"\nFound {len(images)} training images")

    # Sample images
    sample_images = images[:num_samples] if len(
        images) > num_samples else images

    print(f"\n🔍 Checking {len(sample_images)} samples...\n")

    valid_count = 0
    error_count = 0

    for i, img_path in enumerate(sample_images, 1):
        label_path = train_label_dir / img_path.with_suffix('.txt').name

        print(f"[{i}/{len(sample_images)}] {img_path.name}")

        if not label_path.exists():
            print(f"   ⚠️  Missing label file!")
            error_count += 1
            continue

        # Validate format
        labels, errors = validate_yolo_label_format(label_path)

        if errors:
            print(f"   ❌ {len(errors)} format errors")
            for error in errors[:2]:
                print(f"      - {error}")
            error_count += 1
        else:
            print(f"   ✅ {len(labels)} labels - Format valid")
            valid_count += 1

            # Show label details
            for class_id, x_c, y_c, w, h in labels:
                print(
                    f"      Class {class_id}: center=({x_c:.3f}, {y_c:.3f}), size=({w:.3f}, {h:.3f})")

    print(f"\n{'='*70}")
    print(f"📊 Validation Summary:")
    print(f"{'='*70}")
    print(f"   ✅ Valid: {valid_count}/{len(sample_images)}")
    print(f"   ❌ Errors: {error_count}/{len(sample_images)}")

    # Visualize first sample
    if valid_count > 0 and len(sample_images) > 0:
        print(f"\n🎨 Generating visualization for first sample...")
        img_path = sample_images[0]
        label_path = train_label_dir / img_path.with_suffix('.txt').name

        output_dir = Path('outputs/label_verification')
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{img_path.stem}_verified.png"

        visualize_labels_on_image(img_path, label_path, output_path)


if __name__ == "__main__":
    print("🧪 YOLO Label Verification\n")

    # Verify test dataset
    verify_dataset("outputs/yolo_dataset_test", num_samples=10)

    print(f"\n{'='*70}")
    print("✅ Verification Complete!")
    print(f"{'='*70}")
