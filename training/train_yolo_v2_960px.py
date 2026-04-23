"""
YOLO11s Training — SCI Compressed Traffic Signs V2
===================================================
Reliable training script with checkpoint validation and auto-resume.
"""

import os
os.environ['YOLO_VERBOSE'] = 'true'

from ultralytics import YOLO
import torch
import yaml
from pathlib import Path


def validate_checkpoint(ckpt_path):
    """Check checkpoint for NaN/Inf in both model weights and optimizer state."""
    if not ckpt_path.exists():
        return False
    try:
        ckpt = torch.load(str(ckpt_path), map_location='cpu', weights_only=False)
        if not isinstance(ckpt, dict):
            return True

        # Check model weights
        if 'model' in ckpt:
            obj = ckpt['model']
            sd = obj.state_dict() if hasattr(obj, 'state_dict') else (obj if isinstance(obj, dict) else None)
            if sd:
                for name, p in sd.items():
                    if isinstance(p, torch.Tensor) and p.is_floating_point():
                        if torch.isnan(p).any() or torch.isinf(p).any():
                            print(f"  [!] NaN in model weights: {name}")
                            return False

        # Check optimizer state
        if 'optimizer' in ckpt and isinstance(ckpt['optimizer'], dict):
            for pid, pstate in ckpt['optimizer'].get('state', {}).items():
                for k, v in pstate.items():
                    if isinstance(v, torch.Tensor) and v.is_floating_point():
                        if torch.isnan(v).any() or torch.isinf(v).any():
                            print(f"  [!] NaN in optimizer: param {pid}/{k}")
                            return False
        return True
    except Exception as e:
        print(f"  [!] Checkpoint load error: {e}")
        return False


def get_epoch(ckpt_path):
    try:
        ckpt = torch.load(str(ckpt_path), map_location='cpu', weights_only=False)
        return ckpt.get('epoch', None) if isinstance(ckpt, dict) else None
    except Exception:
        return None


def main():
    print("=" * 60)
    print("  YOLO11s Training — SCI Measurements V2")
    print("=" * 60)

    # System info
    print(f"  PyTorch: {torch.__version__}")
    print(f"  CUDA:    {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU:     {torch.cuda.get_device_name(0)}")
        vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"  VRAM:    {vram:.1f} GB")

    # ── Checkpoint validation ──
    run_dir = Path('runs/train/yolo11s_cure_tsd/weights')
    last_pt = run_dir / 'last.pt'
    best_pt = run_dir / 'best.pt'

    model_path = 'models/yolo11s.pt'
    resume = False

    print()
    if last_pt.exists():
        print(f"  Found checkpoint: last.pt ({last_pt.stat().st_size/1e6:.1f} MB)")
        if validate_checkpoint(last_pt):
            epoch = get_epoch(last_pt)
            print(f"  ✓ Valid checkpoint at epoch {epoch}")
            model_path = str(last_pt)
            resume = True
        else:
            print(f"  ✗ Corrupted! Deleting last.pt...")
            last_pt.unlink()
            # Try best.pt as fallback
            if best_pt.exists():
                print(f"  Trying best.pt fallback...")
                if validate_checkpoint(best_pt):
                    epoch = get_epoch(best_pt)
                    print(f"  ✓ best.pt valid at epoch {epoch}")
                    import shutil
                    shutil.copy2(best_pt, last_pt)
                    model_path = str(last_pt)
                    resume = True
                else:
                    print(f"  ✗ best.pt also corrupted, deleting...")
                    best_pt.unlink()
    else:
        print("  No checkpoint found — starting fresh")

    # ── Dataset check ──
    data_yaml = Path('data/yolo_dataset_v2/data.yaml')
    if not data_yaml.exists():
        print(f"  ERROR: {data_yaml} not found!")
        return

    with open(data_yaml) as f:
        data = yaml.safe_load(f)
    data_root = Path(data['path'])
    train_n = len(list((data_root / data['train']).glob('*.jpg')))
    val_n = len(list((data_root / data['val']).glob('*.jpg')))

    print()
    print(f"  Dataset: {train_n:,} train / {val_n:,} val")
    print(f"  Model:   {model_path}")
    print(f"  Resume:  {resume}")
    print(f"  AMP:     OFF (fp32 — stable at 960px)")
    print(f"  LR:      0.002 → cosine decay")
    print()

    # ── Load & train ──
    model = YOLO(model_path)
    params = sum(p.numel() for p in model.model.parameters())
    print(f"  Parameters: {params:,}")
    print()
    print("=" * 60)
    print("  TRAINING START")
    print("  Ctrl+C to pause — run same command to resume")
    print("=" * 60)

    results = model.train(
        data='data/yolo_dataset_v2/data.yaml',
        epochs=200,
        imgsz=960,
        batch=4,            # halved for lower VRAM + safer fp16
        device=0,
        workers=8,
        project='runs/train',
        name='yolo11s_cure_tsd',
        exist_ok=True,
        pretrained=True,
        optimizer='AdamW',
        verbose=True,
        seed=42,
        deterministic=False,
        single_cls=False,
        cos_lr=True,
        close_mosaic=10,
        resume=resume,
        amp=False,              # fp32 — AMP NaNs at 960px regardless of batch
        lr0=0.002,
        lrf=0.001,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3.0,
        warmup_bias_lr=0.1,
        box=7.5,
        cls=0.5,
        dfl=1.5,
        label_smoothing=0.0,
        patience=50,
        save=True,
        cache=False,
        plots=True,
    )

    print()
    print("=" * 60)
    print("  ✅ TRAINING COMPLETE")
    print("=" * 60)
    print(f"  Best: runs/train/yolo11s_cure_tsd/weights/best.pt")
    print()
    print("  Next:")
    print("  python scripts/evaluate_fixed_B.py --model runs/train/yolo11s_cure_tsd/weights/best.pt")
    print("  python scripts/evaluate_rl_agent.py --model runs/train/yolo11s_cure_tsd/weights/best.pt")


if __name__ == '__main__':
    main()
