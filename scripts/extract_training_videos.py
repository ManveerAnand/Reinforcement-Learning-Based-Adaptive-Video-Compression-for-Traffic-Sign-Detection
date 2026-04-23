"""
Extract ALL training videos (01_*) from the CURE-TSD ZIP.
The training script lazy-loads 1 video at a time, so disk is the only constraint.
Expected: ~1500+ videos, ~68 GB on disk.
"""

import zipfile
import os
import shutil
import time
from pathlib import Path

ZIP_PATH = r"E:\Manveer\CURE-TSD.zip"
VIDEO_DIR = Path(r"D:\Dev 2.0\CS307\Research\RL_Video_Compression\data\cure-tsd\data")
LABEL_DIR = Path(r"D:\Dev 2.0\CS307\Research\RL_Video_Compression\data\cure-tsd\labels")


def main():
    print("=" * 60)
    print("FULL TRAINING SET EXTRACTION")
    print("=" * 60)

    VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    LABEL_DIR.mkdir(parents=True, exist_ok=True)

    zf = zipfile.ZipFile(ZIP_PATH, 'r')
    all_names = zf.namelist()

    # Find ALL 01_* training videos and their labels
    train_videos = [n for n in all_names if n.startswith("data/01_") and n.endswith(".mp4")]
    train_labels = [n for n in all_names if n.startswith("labels/01_") and n.endswith(".txt")]

    # Estimate size
    total_size = sum(zf.getinfo(n).file_size for n in train_videos)

    print(f"\nTraining videos in ZIP: {len(train_videos)}")
    print(f"Label files in ZIP: {len(train_labels)}")
    print(f"Estimated disk usage: {total_size / 1e9:.1f} GB")

    # Check disk space
    import shutil as sh
    free = sh.disk_usage(str(VIDEO_DIR.drive + "\\")).free / 1e9
    print(f"Free disk space: {free:.1f} GB")

    if total_size / 1e9 > free * 0.9:
        print(f"\nWARNING: Not enough space! Need {total_size/1e9:.1f} GB but only {free:.1f} GB free.")
        print("Extracting a subset instead (first 25 sign classes)...")
        train_videos = [n for n in train_videos
                        if int(os.path.basename(n).split("_")[1]) <= 25]
        total_size = sum(zf.getinfo(n).file_size for n in train_videos)
        print(f"Subset: {len(train_videos)} videos, {total_size/1e9:.1f} GB")

    # Extract videos
    print(f"\nExtracting {len(train_videos)} training videos...")
    start = time.time()
    extracted = 0
    skipped = 0

    for i, name in enumerate(train_videos):
        basename = os.path.basename(name)
        dest = VIDEO_DIR / basename

        if dest.exists():
            skipped += 1
            continue

        with zf.open(name) as src, open(dest, 'wb') as dst:
            shutil.copyfileobj(src, dst)
        extracted += 1

        if extracted % 100 == 0:
            elapsed = time.time() - start
            rate = extracted / elapsed
            remaining = (len(train_videos) - skipped - extracted) / rate if rate > 0 else 0
            print(f"  [{extracted + skipped}/{len(train_videos)}] "
                  f"{elapsed:.0f}s elapsed, ~{remaining:.0f}s remaining")

    # Extract labels
    print(f"\nExtracting {len(train_labels)} label files...")
    for name in train_labels:
        basename = os.path.basename(name)
        dest = LABEL_DIR / basename
        if not dest.exists():
            with zf.open(name) as src, open(dest, 'wb') as dst:
                shutil.copyfileobj(src, dst)

    zf.close()

    elapsed = time.time() - start

    # Verify
    all_train = sorted(VIDEO_DIR.glob("01_*.mp4"))
    all_labels = sorted(LABEL_DIR.glob("01_*.txt"))
    total_disk = sum(v.stat().st_size for v in all_train) / 1e9

    print(f"\n{'='*60}")
    print(f"EXTRACTION COMPLETE")
    print(f"{'='*60}")
    print(f"  Training videos: {len(all_train)}")
    print(f"  Label files: {len(all_labels)}")
    print(f"  Disk usage: {total_disk:.1f} GB")
    print(f"  Time: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"  Extracted: {extracted}, Skipped: {skipped}")

    # Challenge coverage
    print(f"\nSign class coverage:")
    signs = set(v.stem.split("_")[1] for v in all_train)
    print(f"  {len(signs)} sign classes: {min(signs)}-{max(signs)}")

    print(f"\nReady for training with: python training/train_rl_agent_v2.py")


if __name__ == "__main__":
    main()
