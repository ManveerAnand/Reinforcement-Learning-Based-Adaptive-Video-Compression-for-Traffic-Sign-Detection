"""
YOLOv8 Training Script for CURE-TSD Dataset
Train traffic sign detection on locally generated dataset
"""

from ultralytics import YOLO
import torch
import os
from pathlib import Path


def main():
    # Check for checkpoint to resume from
    checkpoint_path = Path('runs/train/yolo_cure_tsd/weights/last.pt')
    resume_training = checkpoint_path.exists()
    
    if resume_training:
        print("\n" + "="*80)
        print("🔄 CHECKPOINT DETECTED - RESUMING TRAINING")
        print("="*80)
        print(f"Found checkpoint: {checkpoint_path}")
        print("Training will resume from last saved epoch")
        print("Clearing CUDA cache to prevent OOM...")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        print()
    
    # Configuration
    CONFIG = {
        'data': 'data/yolo_dataset_full/data.yaml',
        'model': 'yolov8n.pt' if not resume_training else str(checkpoint_path),  # Use checkpoint if resuming
        'epochs': 100,
        'imgsz': 640,
        'batch': 16,  # Fresh start - can use original batch size
        'device': 0,  # RTX 4060
        'workers': 8,  # CPU threads for data loading
        'project': 'runs/train',
        'name': 'yolo_cure_tsd',
        'exist_ok': True,
        'pretrained': True,
        'optimizer': 'AdamW',
        'verbose': True,
        'seed': 42,
        'deterministic': False,
        'single_cls': False,
        'rect': False,
        'cos_lr': True,
        'close_mosaic': 10,
        'resume': resume_training,  # Auto-detect and resume
        'amp': True,  # Use Tensor Cores (automatic mixed precision)
        'fraction': 1.0,
        'profile': False,
        'freeze': None,
        'lr0': 0.01,
        'lrf': 0.01,
        'momentum': 0.937,
        'weight_decay': 0.0005,
        'warmup_epochs': 3.0,
        'warmup_momentum': 0.8,
        'warmup_bias_lr': 0.1,
        'box': 7.5,
        'cls': 0.5,
        'dfl': 1.5,
        'label_smoothing': 0.0,
        'patience': 50,
        'save': True,
        'save_period': -1,
        'cache': False,  # Set True if enough RAM (loads entire dataset to RAM)
        'plots': True,
        'overlap_mask': True,
        'mask_ratio': 4,
    }

    # Verify GPU
    print("=" * 80)
    print("🔍 SYSTEM CHECK")
    print("=" * 80)
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")
        print(
            f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print()

    # Check dataset
    data_yaml = Path(CONFIG['data'])
    if not data_yaml.exists():
        print(f"❌ ERROR: Dataset not found at {data_yaml}")
        print(f"   Current directory: {os.getcwd()}")
        print(f"   Please wait for dataset generation to complete!")
        return

    print(f"✅ Dataset found: {data_yaml}")
    print()

    # Load model
    print("=" * 80)
    print("📦 LOADING MODEL")
    print("=" * 80)
    if resume_training:
        print(f"Loading from checkpoint: {CONFIG['model']}")
    else:
        print(f"Loading pretrained model: {CONFIG['model']}")
    model = YOLO(CONFIG['model'])
    print(f"✅ Model loaded successfully")
    print()

    # Train
    print("=" * 80)
    print("🚀 STARTING TRAINING")
    print("=" * 80)
    print(f"Epochs: {CONFIG['epochs']}")
    print(f"Batch size: {CONFIG['batch']}")
    print(f"Image size: {CONFIG['imgsz']}")
    print(f"Device: {CONFIG['device']} (RTX 4060)")
    print(f"Mixed Precision (AMP): {CONFIG['amp']} (Tensor Cores enabled)")
    print()

    # Start training
    results = model.train(**CONFIG)

    # Summary
    print()
    print("=" * 80)
    print("✅ TRAINING COMPLETE!")
    print("=" * 80)
    print(f"Best weights: runs/train/{CONFIG['name']}/weights/best.pt")
    print(f"Last weights: runs/train/{CONFIG['name']}/weights/last.pt")
    print(f"Results: runs/train/{CONFIG['name']}/results.png")
    print()
    print("Next steps:")
    print("1. Validate model: python validate_yolo.py")
    print("2. Test inference: python test_inference.py")
    print("3. View TensorBoard: tensorboard --logdir runs/train")
    print()


if __name__ == '__main__':
    main()
