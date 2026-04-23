"""
Fixed-B Baseline Evaluation
Runs the same eval pipeline but with B locked at a fixed value.
Usage: python scripts/evaluate_fixed_B.py --B 6 10 14 18
"""

import argparse
import sys
import time
import json
import csv
import cv2
import numpy as np
import torch
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.phase1.sci_compressor import SCICompressor
from src.phase1.feature_extractor import FeatureExtractor
from ultralytics import YOLO


def load_ground_truth(video_path, labels_dir):
    """Load CURE-TSD ground truth labels (same as evaluate_rl_agent.py)."""
    vid_parts = video_path.stem.split("_")
    label_name = f"{vid_parts[0]}_{vid_parts[1]}.txt"
    label_file = labels_dir / label_name

    if not label_file.exists():
        return {}

    CRITICAL_CLASSES = [1, 6, 10]
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
                sign_class = int(parts[1])
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
                    'bbox': [x_min, y_min, x_max, y_max],
                    'class': sign_class,
                    'is_critical': sign_class in CRITICAL_CLASSES,
                })
            except (ValueError, IndexError):
                continue
    return frame_annotations


def get_challenge_info(video_name):
    """Extract challenge type from video name."""
    parts = video_name.split("_")
    ch_names = {
        "00": "Clear", "01": "Decolor", "02": "LensBlur", "03": "Codec",
        "04": "Dark", "05": "Dirty", "06": "Expose", "07": "GaussBlur",
        "08": "Noise", "09": "Rain", "10": "Shadow", "11": "Snow", "12": "Haze"
    }
    if parts[2] == "00":
        return "Clear", 0
    else:
        ch_code = parts[3] if len(parts) > 3 else "00"
        ch_level = int(parts[4]) if len(parts) > 4 else 0
        return ch_names.get(ch_code, f"Ch{ch_code}"), ch_level


def evaluate_video_fixed_B(yolo_model, frames, ground_truth, masks_dict,
                            fixed_B, feature_extractor):
    """Run evaluation with a fixed B (no RL agent)."""
    if not frames:
        return None

    h, w = frames[0].shape[:2]
    compressor = SCICompressor(frame_height=h, frame_width=w)

    idx = 0
    detection_count = 0
    total_confidence = 0.0
    gt_total = 0
    B_values = []

    while idx < len(frames):
        chunk = frames[idx:idx + fixed_B]
        if len(chunk) < fixed_B:
            chunk = chunk + [frames[-1]] * (fixed_B - len(chunk))

        # SCI compress
        if fixed_B in masks_dict:
            measurement = compressor.compress(chunk, fixed_B)
        else:
            measurement = compressor.compress(chunk, fixed_B)

        # YOLO detection
        meas_input = cv2.cvtColor(measurement, cv2.COLOR_GRAY2RGB) if len(measurement.shape) == 2 else measurement
        results = yolo_model.predict(meas_input, verbose=False, conf=0.15)

        for r in results:
            if r.boxes is not None and len(r.boxes) > 0:
                for box in r.boxes:
                    total_confidence += float(box.conf[0])
                    detection_count += 1

        # Count GT
        for frame_num in range(idx, min(idx + fixed_B, len(frames))):
            if frame_num in ground_truth:
                gt_total += len(ground_truth[frame_num])

        B_values.append(fixed_B)
        idx += fixed_B

    avg_B = fixed_B
    bw_savings = (1 - 1 / fixed_B) * 100

    return {
        'detections': detection_count,
        'avg_confidence': total_confidence / max(1, detection_count),
        'ground_truth_total': gt_total,
        'avg_B': float(avg_B),
        'B_std': 0.0,
        'bandwidth_savings': bw_savings,
        'num_steps': len(B_values)
    }


def run_fixed_B_evaluation(
    B_values_to_test,
    model_path='runs/train/yolo_cure_tsd/weights/best.pt',
    val_videos_dir='data/cure-tsd/data',
    val_pattern='02_*.mp4',
    labels_dir='data/cure-tsd/labels',
    masks_dir='data/masks',
    output_dir='outputs/benchmarks'
):
    print("=" * 70)
    print("  FIXED-B BASELINE EVALUATION")
    print(f"  B values: {B_values_to_test}")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"  YOLO:   {model_path}")
    print(f"  Device: {device}")

    # Load YOLO
    print("  Loading YOLO model...", end=" ")
    yolo_model = YOLO(model_path)
    yolo_model.to(device)
    print("OK")

    # Load masks
    masks_dict = {}
    masks_path = Path(masks_dir)
    for mask_file in masks_path.glob("mask_B*_*.npy"):
        B_val = int(mask_file.stem.split('_')[1].replace('B', ''))
        masks_dict[B_val] = True  # Just track which exist
    print(f"  Masks: {sorted(masks_dict.keys())}")

    # Find videos
    videos = sorted(Path(val_videos_dir).glob(val_pattern))
    print(f"  Videos: {len(videos)}")
    print("=" * 70)

    feature_extractor = FeatureExtractor()
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    all_results = {}

    for fixed_B in B_values_to_test:
        print(f"\n{'='*70}")
        print(f"  Evaluating Fixed B = {fixed_B}")
        print(f"{'='*70}")

        results = []
        start_time = time.time()

        for vi, video_path in enumerate(videos):
            video_name = video_path.stem
            challenge_type, challenge_level = get_challenge_info(video_name)

            # Load frames
            cap = cv2.VideoCapture(str(video_path))
            frames = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(frame)
            cap.release()

            if not frames:
                continue

            # Load GT
            ground_truth = load_ground_truth(video_path, Path(labels_dir))

            # Evaluate
            result = evaluate_video_fixed_B(
                yolo_model, frames, ground_truth, masks_dict,
                fixed_B, feature_extractor
            )

            if result is None:
                continue

            result['video'] = video_name
            result['challenge_type'] = challenge_type
            result['challenge_level'] = challenge_level
            results.append(result)

            if (vi + 1) % 20 == 0 or vi == 0:
                elapsed = time.time() - start_time
                eta = elapsed / (vi + 1) * (len(videos) - vi - 1) / 60
                print(f"  [{vi+1}/{len(videos)}] {challenge_type:>10s} | "
                      f"det={result['detections']:3d} | B={fixed_B} | "
                      f"ETA {eta:.0f}m")

        elapsed = time.time() - start_time

        # Summary for this B
        avg_det = np.mean([r['detections'] for r in results])
        avg_conf = np.mean([r['avg_confidence'] for r in results if r['avg_confidence'] > 0])
        bw = (1 - 1/fixed_B) * 100

        print(f"\n  B={fixed_B}: {avg_det:.1f} avg detections, "
              f"conf={avg_conf:.3f}, BW savings={bw:.1f}%, "
              f"time={elapsed/60:.1f}m")

        # Per-challenge breakdown
        ch_data = defaultdict(list)
        for r in results:
            ch_data[r['challenge_type']].append(r['detections'])

        all_results[fixed_B] = {
            'avg_det': float(avg_det),
            'avg_conf': float(avg_conf),
            'bw_savings': float(bw),
            'per_challenge': {ch: float(np.mean(dets)) for ch, dets in ch_data.items()},
            'results': results
        }

    # ── Final comparison table ──
    print(f"\n{'='*70}")
    print("  FIXED-B BASELINE COMPARISON")
    print(f"{'='*70}")
    print(f"  {'B':>4s}  {'Avg Det':>8s}  {'Conf':>6s}  {'BW Save':>8s}")
    print(f"  {'-'*30}")
    for B in sorted(all_results.keys()):
        d = all_results[B]
        print(f"  {B:4d}  {d['avg_det']:8.1f}  {d['avg_conf']:6.3f}  {d['bw_savings']:7.1f}%")

    # Save to JSON
    summary = {
        'experiment': 'Fixed-B Baselines',
        'yolo_model': model_path,
        'num_videos': len(videos),
        'baselines': {}
    }
    for B in sorted(all_results.keys()):
        d = all_results[B]
        summary['baselines'][str(B)] = {
            'avg_detections': d['avg_det'],
            'avg_confidence': d['avg_conf'],
            'bw_savings': d['bw_savings'],
            'per_challenge': d['per_challenge']
        }

    out_file = output_path / 'fixed_B_baselines.json'
    with open(out_file, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Saved: {out_file}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fixed-B baseline evaluation')
    parser.add_argument('--B', type=int, nargs='+', default=[6, 10, 14, 18],
                        help='B values to test (default: 6 10 14 18)')
    parser.add_argument('--model', type=str,
                        default='runs/train/yolo_cure_tsd/weights/best.pt',
                        help='YOLO model path')
    args = parser.parse_args()

    run_fixed_B_evaluation(
        B_values_to_test=args.B,
        model_path=args.model
    )
