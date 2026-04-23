"""
Validate trained YOLOv8 model on test set
"""

from ultralytics import YOLO
import torch
from pathlib import Path


def main():
    # Find best weights
    weights_path = Path('runs/train/yolo_cure_tsd/weights/best.pt')

    if not weights_path.exists():
        print(f"❌ Trained weights not found at {weights_path}")
        print("   Train model first using: python train_yolo_local.py")
        return

    print("=" * 80)
    print("🔍 VALIDATING MODEL")
    print("=" * 80)
    print(f"Weights: {weights_path}")
    print()

    # Load trained model
    model = YOLO(str(weights_path))

    # Validate on test set
    metrics = model.val(
        data='data/yolo_dataset_full/data.yaml',
        split='val',
        imgsz=640,
        batch=16,
        device=0,
        plots=True,
        save_json=True,
        save_hybrid=False,
    )

    # Print results
    print()
    print("=" * 80)
    print("📊 VALIDATION RESULTS")
    print("=" * 80)
    print(f"mAP50: {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")
    print(f"Precision: {metrics.box.mp:.4f}")
    print(f"Recall: {metrics.box.mr:.4f}")
    print()
    print(f"Results saved to: {Path('runs/train/yolo_cure_tsd')}")
    print()


if __name__ == '__main__':
    main()
