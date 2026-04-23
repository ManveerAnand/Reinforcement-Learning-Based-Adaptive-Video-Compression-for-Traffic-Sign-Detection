"""
Extract validation videos and labels from the full CURE-TSD ZIP.

Extracts only the 280 validation videos (signs 01-05, all challenge types)
and 5 label files from the 233GB ZIP  no need to extract the full dataset.

Usage:
    python scripts/extract_validation_set.py
"""

import zipfile
import os
import sys
import time
import shutil
from pathlib import Path

# Configuration
ZIP_PATH = r"E:\Manveer\CURE-TSD.zip"
PROJECT_ROOT = Path(__file__).parent.parent

# Output directories
VIDEO_DIR = PROJECT_ROOT / "data" / "cure-tsd" / "data"
LABEL_DIR = PROJECT_ROOT / "data" / "cure-tsd" / "labels"
MASKS_DIR = PROJECT_ROOT / "data" / "masks"

# Source masks directory (already exists in scripts/data/masks/)
SOURCE_MASKS_DIR = PROJECT_ROOT / "scripts" / "data" / "masks"

# Sign classes to extract (01-05 = the 5 used in prior evaluation)
SIGN_IDS = ["01", "02", "03", "04", "05"]


def extract_validation_data():
    """Extract validation videos and labels from the ZIP."""
    print("=" * 80)
    print("CURE-TSD VALIDATION SET EXTRACTION")
    print("=" * 80)
    print(f"\nZIP: {ZIP_PATH}")
    print(f"Videos -> {VIDEO_DIR}")
    print(f"Labels -> {LABEL_DIR}")
    print(f"Masks  -> {MASKS_DIR}")

    # Verify ZIP exists
    if not os.path.exists(ZIP_PATH):
        print(f"\n ZIP not found: {ZIP_PATH}")
        sys.exit(1)

    # Create directories
    VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    LABEL_DIR.mkdir(parents=True, exist_ok=True)
    MASKS_DIR.mkdir(parents=True, exist_ok=True)

    # Open ZIP
    print(f"\n Opening ZIP ({os.path.getsize(ZIP_PATH) / 1e9:.1f} GB)...")
    zf = zipfile.ZipFile(ZIP_PATH, 'r')
    all_names = zf.namelist()
    print(f"   Total entries in ZIP: {len(all_names)}")

    # Identify files to extract
    # Validation videos: data/02_XX_*.mp4 where XX is in SIGN_IDS
    val_videos = []
    val_labels = []

    for name in all_names:
        # Match validation videos
        if name.startswith("data/02_"):
            # Extract sign ID (position 1 after split)
            basename = os.path.basename(name)
            parts = basename.replace(".mp4", "").split("_")
            if len(parts) >= 2 and parts[1] in SIGN_IDS:
                val_videos.append(name)

        # Match validation labels
        if name.startswith("labels/02_"):
            basename = os.path.basename(name)
            sign_id = basename.replace("labels/", "").replace(".txt", "").split("_")[1] if "_" in basename else ""
            if sign_id in SIGN_IDS:
                val_labels.append(name)

    print(f"\n Files to extract:")
    print(f"   Validation videos: {len(val_videos)}")
    print(f"   Label files: {len(val_labels)}")

    if len(val_videos) == 0:
        print("\n No matching videos found! Check ZIP structure.")
        # Debug: show some entries
        print("   First 20 entries:")
        for n in all_names[:20]:
            print(f"     {n}")
        sys.exit(1)

    # Estimate size
    total_size = sum(zf.getinfo(n).file_size for n in val_videos + val_labels)
    print(f"   Estimated size: {total_size / 1e9:.2f} GB")

    # Extract videos
    print(f"\n Extracting {len(val_videos)} validation videos...")
    start = time.time()

    for i, name in enumerate(val_videos):
        # Extract to data/cure-tsd/data/  strip the "data/" prefix from zip path
        basename = os.path.basename(name)
        dest = VIDEO_DIR / basename

        if dest.exists():
            if i % 50 == 0:
                print(f"   [{i+1}/{len(val_videos)}] Skipping (exists): {basename}")
            continue

        # Extract
        with zf.open(name) as src, open(dest, 'wb') as dst:
            shutil.copyfileobj(src, dst)

        if (i + 1) % 20 == 0 or i == 0:
            elapsed = time.time() - start
            rate = (i + 1) / elapsed
            eta = (len(val_videos) - i - 1) / rate if rate > 0 else 0
            print(f"   [{i+1}/{len(val_videos)}] {basename} "
                  f"({elapsed:.0f}s elapsed, ~{eta:.0f}s remaining)")

    # Extract labels
    print(f"\n Extracting {len(val_labels)} label files...")
    for name in val_labels:
        basename = os.path.basename(name)
        dest = LABEL_DIR / basename

        with zf.open(name) as src, open(dest, 'wb') as dst:
            shutil.copyfileobj(src, dst)
        print(f"    {basename}")

    zf.close()

    # Set up masks
    print(f"\n Setting up SCI masks...")
    if SOURCE_MASKS_DIR.exists():
        for mask_file in SOURCE_MASKS_DIR.glob("mask_*.npy"):
            dest = MASKS_DIR / mask_file.name
            if not dest.exists():
                shutil.copy2(mask_file, dest)
                print(f"    Copied {mask_file.name}")
            else:
                print(f"     {mask_file.name} already exists")
    else:
        print(f"     Source masks not found at {SOURCE_MASKS_DIR}")

    # Verify
    print(f"\n Verification:")
    video_count = len(list(VIDEO_DIR.glob("02_*.mp4")))
    label_count = len(list(LABEL_DIR.glob("02_*.txt")))
    mask_count = len(list(MASKS_DIR.glob("mask_*.npy")))

    print(f"   Videos: {video_count} (expected: {len(val_videos)})")
    print(f"   Labels: {label_count} (expected: {len(val_labels)})")
    print(f"   Masks:  {mask_count} (expected: 6)")

    elapsed = time.time() - start
    print(f"\n  Total time: {elapsed:.0f}s ({elapsed/60:.1f} min)")

    if video_count >= len(val_videos) and label_count >= len(val_labels):
        print("\n EXTRACTION COMPLETE!")
        print(f"\n Data ready at: {VIDEO_DIR.parent}")
    else:
        print("\n  Some files may be missing. Check the output above.")

    # Summary of challenge types
    print(f"\n Challenge type breakdown:")
    videos = sorted(VIDEO_DIR.glob("02_*.mp4"))
    challenge_counts = {}
    for v in videos:
        parts = v.stem.split("_")
        if len(parts) >= 4:
            ct = parts[3]
            challenge_counts[ct] = challenge_counts.get(ct, 0) + 1

    challenge_names = {
        "00": "Challenge-Free",
        "01": "Decolorization",
        "02": "Lens Blur",
        "03": "Codec Error",
        "04": "Darkening",
        "05": "Dirty Lens",
        "06": "Exposure",
        "07": "Gaussian Blur",
        "08": "Noise",
        "09": "Rain",
        "10": "Shadow",
        "11": "Snow",
    }

    for code in sorted(challenge_counts.keys()):
        name = challenge_names.get(code, f"Unknown ({code})")
        print(f"   {code} ({name}): {challenge_counts[code]} videos")


if __name__ == "__main__":
    extract_validation_data()
