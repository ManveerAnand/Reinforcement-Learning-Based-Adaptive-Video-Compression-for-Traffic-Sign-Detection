"""
FAST Dataset Generator V2 — Fixed version
Uses 4 workers (not 14), loads masks inside workers to avoid memory explosion.
Resumes from existing progress.

Usage: python scripts/generate_dataset_v2_fast.py
"""

import time
import sys
import random
import yaml
import numpy as np
import cv2
from pathlib import Path
from multiprocessing import Pool, cpu_count

# ── Constants ──
H, W = 1236, 1628
MASKS_DIR = Path("data/masks")
SEED = 42


def get_mask(B):
    """Load or generate mask for given B (called inside worker)."""
    mask_path = MASKS_DIR / f"mask_B{B}_{H}x{W}.npy"
    if mask_path.exists():
        return np.load(mask_path)
    np.random.seed(SEED)
    mask = np.random.binomial(1, 0.5, (H, W, B)).astype(np.float32)
    np.save(mask_path, mask)
    return mask


def load_labels(label_file, B, stride):
    """Parse CURE-TSD labels and create YOLO-format labels per chunk."""
    if not label_file.exists():
        return {}

    frame_annotations = {}
    with open(label_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('frameNumber'):
                continue
            parts = line.split('_')
            if len(parts) != 10:
                continue
            try:
                frame_num = int(parts[0]) - 1
                sign_class = int(parts[1]) - 1
                llx, lly = int(parts[2]), int(parts[3])
                lrx, lry = int(parts[4]), int(parts[5])
                ulx, uly = int(parts[6]), int(parts[7])
                urx, ury = int(parts[8]), int(parts[9])

                x_min = min(llx, lrx, ulx, urx)
                x_max = max(llx, lrx, ulx, urx)
                y_min = min(lly, lry, uly, ury)
                y_max = max(lly, lry, uly, ury)

                if frame_num not in frame_annotations:
                    frame_annotations[frame_num] = []
                frame_annotations[frame_num].append({
                    'class': sign_class,
                    'bbox': [x_min, y_min, x_max, y_max]
                })
            except (ValueError, IndexError):
                continue

    if not frame_annotations:
        return {}

    img_w, img_h = 1628, 1236
    max_frame = max(frame_annotations.keys()) + 1
    chunk_labels = {}

    for start in range(0, max_frame - B + 1, stride):
        end = start + B - 1
        seen_signs = {}

        for f in range(start, end + 1):
            if f not in frame_annotations:
                continue
            for ann in frame_annotations[f]:
                cx = (ann['bbox'][0] + ann['bbox'][2]) / 2
                cy = (ann['bbox'][1] + ann['bbox'][3]) / 2
                key = (ann['class'], round(cx / 50), round(cy / 50))

                if key not in seen_signs:
                    seen_signs[key] = ann
                else:
                    old = seen_signs[key]
                    old_area = (old['bbox'][2]-old['bbox'][0]) * (old['bbox'][3]-old['bbox'][1])
                    new_area = (ann['bbox'][2]-ann['bbox'][0]) * (ann['bbox'][3]-ann['bbox'][1])
                    if new_area > old_area:
                        seen_signs[key] = ann

        if not seen_signs:
            continue

        yolo_labels = []
        for ann in seen_signs.values():
            x_min, y_min, x_max, y_max = ann['bbox']
            cx = max(0, min(1, ((x_min + x_max) / 2) / img_w))
            cy = max(0, min(1, ((y_min + y_max) / 2) / img_h))
            w = max(0.001, min(1, (x_max - x_min) / img_w))
            h = max(0.001, min(1, (y_max - y_min) / img_h))
            yolo_labels.append(f"{ann['class']} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

        chunk_labels[(start, end)] = yolo_labels

    return chunk_labels


def process_single_video(args):
    """Process one video. Mask is loaded inside worker, NOT passed as arg."""
    video_path_str, label_file_str, B, out_img_str, out_lbl_str = args

    video_path = Path(video_path_str)
    label_file = Path(label_file_str)
    out_img = Path(out_img_str)
    out_lbl = Path(out_lbl_str)
    video_id = video_path.stem

    try:
        # Load mask inside worker (not passed — avoids pickle overhead)
        mask = get_mask(B)

        # Load video
        cap = cv2.VideoCapture(str(video_path))
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()

        if len(frames) < B:
            return 0

        chunk_labels = load_labels(label_file, B, stride=B)
        if not chunk_labels:
            return 0

        saved = 0
        for start_frame in range(0, len(frames) - B + 1, B):
            end_frame = start_frame + B - 1
            labels = chunk_labels.get((start_frame, end_frame), [])
            if not labels:
                continue

            # SCI compress
            Y = np.zeros((H, W), dtype=np.float32)
            for b in range(B):
                gray = cv2.cvtColor(frames[start_frame + b], cv2.COLOR_BGR2GRAY).astype(np.float32)
                if gray.shape != (H, W):
                    gray = cv2.resize(gray, (W, H))
                Y += mask[:, :, b] * gray

            if Y.max() > 0:
                Y = (Y / Y.max()) * 255.0

            img_name = f"{video_id}_frames_{start_frame:03d}-{end_frame:03d}_B{B:02d}.jpg"
            cv2.imwrite(str(out_img / img_name), np.clip(Y, 0, 255).astype(np.uint8))

            with open(out_lbl / img_name.replace('.jpg', '.txt'), 'w') as f:
                for label in labels:
                    f.write(label + '\n')
            saved += 1

        return saved

    except Exception as e:
        return 0


def main():
    B_VALUES = [6, 8, 10, 12, 14, 16, 18, 20]
    VIDEOS_DIR = Path("data/cure-tsd/data")
    LABELS_DIR = Path("data/cure-tsd/labels")
    OUTPUT_DIR = Path("data/yolo_dataset_v2")
    TRAIN_SPLIT = 0.8
    NUM_WORKERS = 4  # Safe for memory

    print("=" * 70)
    print("  DATASET GENERATOR V2 (4 workers, memory-safe)")
    print("=" * 70)
    print(f"  B values:  {B_VALUES}")
    print(f"  Workers:   {NUM_WORKERS}")
    print(f"  Output:    {OUTPUT_DIR}")

    # Find videos
    all_videos = sorted(VIDEOS_DIR.glob("*.mp4"))
    print(f"  Videos:    {len(all_videos)}")

    # Split
    random.seed(SEED)
    shuffled = all_videos.copy()
    random.shuffle(shuffled)
    n_train = int(len(shuffled) * TRAIN_SPLIT)
    splits = {'train': shuffled[:n_train], 'val': shuffled[n_train:]}
    print(f"  Train: {len(splits['train'])}, Val: {len(splits['val'])}")

    # Create dirs
    for s in ['train', 'val']:
        (OUTPUT_DIR / 'images' / s).mkdir(parents=True, exist_ok=True)
        (OUTPUT_DIR / 'labels' / s).mkdir(parents=True, exist_ok=True)

    # Pre-generate masks (just ensure they exist on disk)
    print("\n  Ensuring masks exist...")
    for B in B_VALUES:
        mask_path = MASKS_DIR / f"mask_B{B}_{H}x{W}.npy"
        if not mask_path.exists():
            print(f"    Generating mask B={B}...")
            np.random.seed(SEED)
            mask = np.random.binomial(1, 0.5, (H, W, B)).astype(np.float32)
            np.save(mask_path, mask)
            print(f"    Saved {mask_path}")
        else:
            print(f"    B={B}: exists")

    total_start = time.time()
    total_new = 0

    for split_name, videos in splits.items():
        out_img = OUTPUT_DIR / 'images' / split_name
        out_lbl = OUTPUT_DIR / 'labels' / split_name

        # Find already-processed videos
        existing = set()
        for f in out_img.glob("*.jpg"):
            name = f.stem
            for sep in ['_frames_', '_f']:
                if sep in name:
                    existing.add(name.split(sep)[0])
                    break

        # Build tasks (skip existing)
        random.seed(SEED + (0 if split_name == 'train' else 1))
        tasks = []
        for v in videos:
            vid_id = v.stem
            label_id = '_'.join(vid_id.split('_')[:2])
            lbl = LABELS_DIR / f"{label_id}.txt"
            B = random.choice(B_VALUES)  # Must call even for skipped (keeps RNG in sync)
            if not lbl.exists() or vid_id in existing:
                continue
            tasks.append((str(v), str(lbl), B, str(out_img), str(out_lbl)))

        print(f"\n{'='*70}")
        print(f"  {split_name.upper()}: {len(tasks)} new videos ({len(existing)} already done)")
        print(f"{'='*70}")

        if not tasks:
            continue

        split_start = time.time()
        split_new = 0

        with Pool(processes=NUM_WORKERS) as pool:
            for idx, saved in enumerate(pool.imap_unordered(process_single_video, tasks), 1):
                split_new += saved
                if idx % 50 == 0 or idx == len(tasks):
                    elapsed = time.time() - split_start
                    rate = idx / elapsed
                    remaining = (len(tasks) - idx) / rate / 60
                    print(f"    [{idx:4d}/{len(tasks)}] +{split_new} imgs | "
                          f"{rate:.1f} vid/s | ETA: {remaining:.0f}m")

        total_new += split_new
        print(f"  Done: +{split_new} new images in {(time.time()-split_start)/60:.1f}m")

    # Create data.yaml
    class_names = [
        'Speed Limit 30', 'Speed Limit 60', 'Speed Limit 90',
        'No Overtaking (All)', 'No Overtaking (Trucks)',
        'Right-of-Way at Next Intersection', 'Priority Road',
        'Give Way', 'Stop', 'No Entry',
        'No Entry (Trucks)', 'Roundabout', 'End of No Overtaking (All)',
        'End of No Overtaking (Trucks)'
    ]

    train_count = len(list((OUTPUT_DIR / 'images' / 'train').glob('*.jpg')))
    val_count = len(list((OUTPUT_DIR / 'images' / 'val').glob('*.jpg')))

    data = {
        'path': str(OUTPUT_DIR.absolute()),
        'train': 'images/train',
        'val': 'images/val',
        'nc': len(class_names),
        'names': class_names,
    }
    with open(OUTPUT_DIR / 'data.yaml', 'w') as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    print(f"\n{'='*70}")
    print(f"  COMPLETE — {(time.time()-total_start)/60:.1f} minutes")
    print(f"  Train: {train_count} | Val: {val_count} | New: {total_new}")
    print(f"  Config: {OUTPUT_DIR / 'data.yaml'}")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
