"""
RL Agent Evaluation - Production-Ready
=======================================
Evaluate trained DQN agent on the full 280 validation videos.
Produces results for the ICANN 2026 paper.

Fixes from previous version:
  1. sys.path.append BEFORE src imports (was after -> ModuleNotFoundError)
  2. Ground truth label loading uses SeqID_SignID format (was per-video -> always empty)
  3. Single detection pass per step (was running detection twice)
  4. No Windows-breaking Unicode characters
  5. Clean progress output with per-challenge breakdown

Usage:
  python scripts/evaluate_rl_agent.py              # Full 280-video run
  python scripts/evaluate_rl_agent.py --quick 10   # Quick test on 10 videos
"""

import sys
import io
from pathlib import Path

# MUST be before src.* imports
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import time
import json
import argparse

from src.phase1.sci_compressor import SCICompressor
from src.phase1.feature_extractor import FeatureExtractor


# ============================================================
# DQN Agent (matches training architecture exactly)
# ============================================================

class QNetwork(nn.Module):
    def __init__(self, state_size=7, action_size=3, hidden_size=128):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


class DQNAgent:
    def __init__(self, state_size=7, action_size=3, hidden_size=128):
        self.state_size = state_size
        self.action_size = action_size
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.qnetwork_local = QNetwork(state_size, action_size, hidden_size).to(self.device)

    def act(self, state):
        """Greedy action selection (no epsilon for evaluation)."""
        state_t = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            q_values = self.qnetwork_local(state_t)
        return int(np.argmax(q_values.cpu().numpy()))


# ============================================================
# Video & Label Loading
# ============================================================

def load_video_frames(video_path, max_frames=300):
    """Load frames from video file, quietly."""
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    while len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames


def load_ground_truth(video_path, labels_dir):
    """
    Load CURE-TSD ground truth labels.
    Label files are SeqID_SignID.txt (e.g., 02_01.txt), NOT per-video.
    Format: frameNumber_signType_llx_lly_lrx_lry_ulx_uly_urx_ury
    Example: 001_09_942_593_958_593_942_608_958_608
    """
    vid_parts = video_path.stem.split("_")
    label_name = f"{vid_parts[0]}_{vid_parts[1]}.txt"
    label_file = labels_dir / label_name

    if not label_file.exists():
        return {}

    # Critical sign types (Speed Limit=1, STOP=6, Yield=10)
    CRITICAL_CLASSES = [1, 6, 10]

    frame_annotations = {}
    with open(label_file, 'r') as f:
        for line in f:
            line = line.strip()
            # Skip header or empty lines
            if not line or line.startswith('frameNumber'):
                continue

            parts = line.split('_')
            if len(parts) != 10:
                continue

            try:
                frame_num = int(parts[0]) - 1  # Convert to 0-indexed (matches training)
                sign_class = int(parts[1])

                # Bounding box corners
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
    """Extract challenge type and level from video name."""
    parts = video_name.split("_")
    ch_names = {
        "00": "Clear", "01": "Decolor", "02": "LensBlur", "03": "Codec",
        "04": "Dark", "05": "Dirty", "06": "Expose", "07": "GaussBlur",
        "08": "Noise", "09": "Rain", "10": "Shadow", "11": "Snow", "12": "Haze"
    }
    if parts[2] == "00":
        return "Clear", 0, "00"
    else:
        ch_code = parts[3] if len(parts) > 3 else "00"
        ch_level = int(parts[4]) if len(parts) > 4 else 0
        return ch_names.get(ch_code, f"Ch{ch_code}"), ch_level, ch_code


# ============================================================
# Core Evaluation
# ============================================================

def evaluate_video(agent, yolo_model, frames, ground_truth, masks_dict,
                   feature_extractor, device):
    """
    Run the RL agent on a single video with greedy policy.
    Matches the training pipeline exactly:
      1. Extract features from current frame
      2. Normalize state [0,1]
      3. Agent selects action (greedy)
      4. Apply SCI compression
      5. Run YOLO detection
      6. Compute reward
    """
    if not frames:
        return None

    h, w = frames[0].shape[:2]
    compressor = SCICompressor(frame_height=h, frame_width=w)

    current_B = 10
    prev_frame = frames[0]
    idx = 0

    # Per-step tracking
    B_sequence = []
    all_detections = []
    step_rewards = []
    actions = []
    total_confidence = 0.0
    detection_count = 0
    gt_total = 0

    while idx < len(frames):
        # Get chunk of B frames
        chunk = frames[idx:idx + current_B]
        if len(chunk) < current_B:
            chunk = chunk + [frames[-1]] * (current_B - len(chunk))

        current_frame = chunk[0]

        # ── Step 1: SCI Compress ──
        if current_B in masks_dict:
            measurement = compressor.compress(chunk, current_B)
        else:
            measurement = current_frame

        # ── Step 2: YOLO Detection ──
        meas_input = cv2.cvtColor(measurement, cv2.COLOR_GRAY2RGB) if len(measurement.shape) == 2 else measurement
        results = yolo_model.predict(meas_input, verbose=False, conf=0.15)

        detections = []
        for r in results:
            if r.boxes is not None and len(r.boxes) > 0:
                for box in r.boxes:
                    det = {
                        'bbox': box.xyxy[0].cpu().numpy(),
                        'confidence': float(box.conf[0]),
                        'class': int(box.cls[0])
                    }
                    detections.append(det)
                    total_confidence += det['confidence']
                    detection_count += 1

        all_detections.extend(detections)

        # Count ground truth signs in this frame range
        gt_in_chunk = 0
        critical_gt_in_chunk = 0
        for frame_num in range(idx, min(idx + current_B, len(frames))):
            if frame_num in ground_truth:
                gt_in_chunk += len(ground_truth[frame_num])
                critical_gt_in_chunk += sum(
                    1 for ann in ground_truth[frame_num] if ann.get('is_critical', False)
                )
        gt_total += gt_in_chunk

        # ── Step 3: Extract Features & Normalize ──
        features = feature_extractor.extract_state(
            current_frame, prev_frame, detections, current_B
        )
        features_norm = features.copy()
        features_norm[0] = features[0] / 50.0      # flow [0,50] -> [0,1]
        features_norm[3] = features[3] / 1000.0    # blur [0,1000] -> [0,1]
        features_norm[4] = features[4] / 255.0     # brightness [0,255] -> [0,1]
        features_norm = np.clip(features_norm, 0, 1).astype(np.float32)

        # ── Step 4: Agent Selects Action ──
        action = agent.act(features_norm)
        actions.append(action)

        # ── Step 5: Compute Reward (matches training env V4e exactly) ──
        # Proportional detection score (power 1.5, matches training)
        gt_count_for_score = max(1, gt_in_chunk)
        det_ratio = min(1.0, len(detections) / gt_count_for_score)
        det_score = det_ratio ** 1.5

        # Scene complexity kappa
        flow_norm = features_norm[0]
        edge_density = features_norm[1]
        blur_norm = features_norm[3]
        kappa = 0.5 * flow_norm + 0.3 * edge_density + 0.2 * (1 - blur_norm)

        # Scene-adaptive weights (wider range [0.3, 0.7])
        w_det = 0.3 + 0.4 * kappa
        w_bw = 0.7 - 0.4 * kappa

        # High-B penalty for complex scenes only
        B_penalty = 0.0
        if kappa > 0.6 and current_B > 12:
            B_penalty = 0.15 * (current_B - 12) / 8.0

        # Critical misses
        critical_misses = max(0, critical_gt_in_chunk - min(len(detections), critical_gt_in_chunk))

        # Reward (lambda=0.1, matches training)
        reward = w_det * det_score + w_bw * (current_B / 20.0) - 0.1 * critical_misses - B_penalty

        step_rewards.append(reward)

        # ── Step 6: Update B ──
        B_sequence.append(current_B)

        if action == 0:
            current_B = max(6, current_B - 2)
        elif action == 2:
            current_B = min(20, current_B + 2)

        prev_frame = current_frame
        idx += len(chunk)

    # ── Compute Results ──
    avg_B = np.mean(B_sequence) if B_sequence else 10.0
    action_dist = {0: actions.count(0), 1: actions.count(1), 2: actions.count(2)}

    return {
        'detections': detection_count,
        'avg_confidence': total_confidence / max(1, detection_count),
        'ground_truth_total': gt_total,
        'avg_B': avg_B,
        'B_std': np.std(B_sequence) if B_sequence else 0.0,
        'B_min': min(B_sequence) if B_sequence else 10,
        'B_max': max(B_sequence) if B_sequence else 10,
        'total_reward': sum(step_rewards),
        'avg_reward': np.mean(step_rewards) if step_rewards else 0.0,
        'num_steps': len(B_sequence),
        'B_sequence': B_sequence,
        'compression_ratio': avg_B,
        'bandwidth_savings': (1 - 1/avg_B) * 100 if avg_B > 0 else 0.0,
        'actions_decrease': action_dist[0],
        'actions_keep': action_dist[1],
        'actions_increase': action_dist[2],
    }


def format_time(seconds):
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        return f"{h}h {m}m"


# ============================================================
# Main Experiment
# ============================================================

def run_experiment(
    agent_path='runs/rl_training_v2/best_model_v2.pth',
    model_path='runs/train/yolo_cure_tsd/weights/best.pt',
    val_videos_dir='data/cure-tsd/data',
    val_pattern='02_*.mp4',
    labels_dir='data/cure-tsd/labels',
    masks_dir='data/masks',
    output_dir='outputs/benchmarks',
    num_videos=None
):
    print("=" * 70)
    print("  RL AGENT EVALUATION (v2)")
    print("  Greedy Policy on 280 Validation Videos")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"  Agent:    {agent_path}")
    print(f"  YOLO:     {model_path}")
    print(f"  Device:   {device}")
    print(f"  Videos:   {val_videos_dir}/{val_pattern}")
    print("=" * 70)

    # ── Load Agent ──
    print("\n  Loading DQN agent...", end=" ")
    agent = DQNAgent(state_size=7, action_size=3, hidden_size=128)
    if not Path(agent_path).exists():
        print(f"FAILED - file not found: {agent_path}")
        return
    checkpoint = torch.load(agent_path, map_location=device, weights_only=True)
    agent.qnetwork_local.load_state_dict(checkpoint['qnetwork_local'])
    agent.qnetwork_local.eval()
    print("OK")

    # ── Load YOLO ──
    print("  Loading YOLO model...", end=" ")
    from ultralytics import YOLO
    yolo_model = YOLO(model_path)
    yolo_model.to(device)
    print("OK")

    # ── Load Masks ──
    print("  Loading SCI masks...", end=" ")
    B_choices = [6, 8, 10, 12, 15, 20]
    masks_dict = {}
    masks_path = Path(masks_dir)
    for B in B_choices:
        mask_file = masks_path / f"mask_B{B}_1236x1628.npy"
        if mask_file.exists():
            masks_dict[B] = np.load(mask_file)
    print(f"OK ({len(masks_dict)}/{len(B_choices)} loaded)")

    # ── Discover Videos ──
    val_videos = sorted(Path(val_videos_dir).glob(val_pattern))
    if num_videos:
        val_videos = val_videos[:num_videos]
        print(f"\n  [Quick mode: {num_videos} videos]")
    print(f"  Validation videos: {len(val_videos)}")

    labels_path = Path(labels_dir)
    feature_extractor = FeatureExtractor()

    # ── Run Evaluation ──
    results = []
    overall_start = time.time()
    errors = 0

    print(f"\n  {'='*62}")
    print(f"  {'Video':<30s} {'Det':>5s} {'AvgB':>5s} {'BW%':>5s} {'Reward':>7s} {'Time':>5s}")
    print(f"  {'-'*62}")

    for i, video_path in enumerate(val_videos):
        video_name = video_path.stem
        vid_start = time.time()

        try:
            # Load video
            frames = load_video_frames(video_path)
            if not frames:
                errors += 1
                continue

            # Load ground truth (correct label mapping)
            ground_truth = load_ground_truth(video_path, labels_path)

            # Evaluate
            result = evaluate_video(
                agent, yolo_model, frames, ground_truth,
                masks_dict, feature_extractor, device
            )

            if result is None:
                errors += 1
                continue

            # Add metadata
            challenge_name, challenge_level, challenge_code = get_challenge_info(video_name)
            result['video'] = video_name
            result['num_frames'] = len(frames)
            result['challenge_type'] = challenge_name
            result['challenge_level'] = challenge_level
            result['challenge_code'] = challenge_code
            results.append(result)

            vid_time = time.time() - vid_start
            elapsed = time.time() - overall_start
            eta = (elapsed / (i + 1)) * (len(val_videos) - i - 1)

            # Progress (every 10 videos or first/last)
            if (i + 1) % 10 == 0 or i == 0 or i == len(val_videos) - 1:
                print(f"  {video_name:<30s} "
                      f"{result['detections']:5d} "
                      f"{result['avg_B']:5.1f} "
                      f"{result['bandwidth_savings']:4.1f}% "
                      f"{result['total_reward']:7.2f} "
                      f"{vid_time:4.1f}s "
                      f"[{i+1}/{len(val_videos)}] ETA {format_time(eta)}")

        except Exception as e:
            print(f"  ERROR: {video_name}: {e}")
            errors += 1

    # ── Save Results ──
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(results)
    df_save = df.drop(columns=['B_sequence'], errors='ignore')
    df_save.to_csv(output_path / 'rl_agent_results.csv', index=False)

    # ── Summary ──
    elapsed = time.time() - overall_start

    print(f"\n  {'='*62}")
    print(f"\n{'='*70}")
    print(f"  EVALUATION RESULTS")
    print(f"{'='*70}")
    print(f"  Videos:           {len(results)} processed, {errors} errors")
    print(f"  Total Time:       {format_time(elapsed)}")
    print(f"  Time/Video:       {elapsed/max(1,len(results)):.1f}s")
    print(f"")
    print(f"  --- Detection Performance ---")
    print(f"  Avg Detections:   {df['detections'].mean():.1f} +/- {df['detections'].std():.1f}")
    print(f"  Avg Confidence:   {df['avg_confidence'].mean():.3f}")
    print(f"  Ground Truth:     {df['ground_truth_total'].mean():.1f} avg signs/video")
    print(f"")
    print(f"  --- Compression Strategy ---")
    print(f"  Avg B-value:      {df['avg_B'].mean():.2f} +/- {df['avg_B'].std():.2f}")
    print(f"  B Variability:    {df['B_std'].mean():.2f}")
    print(f"  B Range:          [{df['B_min'].min()}, {df['B_max'].max()}]")
    print(f"  Compression:      {df['compression_ratio'].mean():.1f}x")
    print(f"  BW Savings:       {df['bandwidth_savings'].mean():.1f}%")
    print(f"")
    print(f"  --- Agent Behavior ---")
    print(f"  Avg Reward/Step:  {df['avg_reward'].mean():.3f}")
    total_d = df['actions_decrease'].sum()
    total_k = df['actions_keep'].sum()
    total_i = df['actions_increase'].sum()
    total_a = total_d + total_k + total_i
    print(f"  Actions D/K/I:    {total_d}/{total_k}/{total_i} "
          f"({total_d/total_a*100:.0f}%/{total_k/total_a*100:.0f}%/{total_i/total_a*100:.0f}%)")

    # ── Per-Challenge Breakdown ──
    print(f"\n  --- Per-Challenge Type Breakdown ---")
    print(f"  {'Challenge':<12s} {'Videos':>6s} {'AvgDet':>7s} {'AvgB':>6s} {'BW%':>6s} {'Reward':>7s}")
    print(f"  {'-'*50}")

    challenge_summary = {}
    for ch_type in sorted(df['challenge_type'].unique()):
        ch_df = df[df['challenge_type'] == ch_type]
        ch_stats = {
            'count': len(ch_df),
            'avg_det': float(ch_df['detections'].mean()),
            'avg_B': float(ch_df['avg_B'].mean()),
            'bw_savings': float(ch_df['bandwidth_savings'].mean()),
            'avg_reward': float(ch_df['total_reward'].mean()),
        }
        challenge_summary[ch_type] = ch_stats
        print(f"  {ch_type:<12s} {ch_stats['count']:6d} "
              f"{ch_stats['avg_det']:7.1f} "
              f"{ch_stats['avg_B']:6.2f} "
              f"{ch_stats['bw_savings']:5.1f}% "
              f"{ch_stats['avg_reward']:7.2f}")

    print(f"{'='*70}")

    # ── Save Summary JSON ──
    summary = {
        'experiment': 'RL Agent v2 (Greedy)',
        'agent_path': str(agent_path),
        'num_videos': len(results),
        'total_time_minutes': elapsed / 60,
        'avg_detections': float(df['detections'].mean()),
        'std_detections': float(df['detections'].std()),
        'avg_confidence': float(df['avg_confidence'].mean()),
        'avg_B': float(df['avg_B'].mean()),
        'B_std': float(df['B_std'].mean()),
        'compression_ratio': float(df['compression_ratio'].mean()),
        'bandwidth_savings': float(df['bandwidth_savings'].mean()),
        'total_reward': float(df['total_reward'].mean()),
        'avg_reward_per_step': float(df['avg_reward'].mean()),
        'action_distribution': {
            'decrease': int(total_d),
            'keep': int(total_k),
            'increase': int(total_i)
        },
        'per_challenge': challenge_summary
    }

    with open(output_path / 'rl_agent_summary_v2.json', 'w') as f:
        json.dump(summary, f, indent=2)

    # Save B-sequences
    sequences = [{'video': r['video'], 'B_sequence': r['B_sequence']} for r in results]
    with open(output_path / 'rl_agent_B_sequences_v2.json', 'w') as f:
        json.dump(sequences, f, indent=2)

    print(f"\n  Results:      {output_path / 'rl_agent_results.csv'}")
    print(f"  Summary:      {output_path / 'rl_agent_summary_v2.json'}")
    print(f"  B-sequences:  {output_path / 'rl_agent_B_sequences_v2.json'}")
    print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate RL agent on validation set')
    parser.add_argument('--quick', type=int, default=None, help='Quick test on N videos')
    parser.add_argument('--model', type=str, default='runs/train/yolo_cure_tsd/weights/best.pt',
                        help='YOLO model path')
    args = parser.parse_args()

    run_experiment(num_videos=args.quick, model_path=args.model)

