"""
Real-Time Performance Profiling
================================
Measures actual latency of each pipeline component.
Run with: python profile_pipeline_latency.py --video <path_to_test_video>
"""

import time
import numpy as np
import cv2
import torch
import argparse
from pathlib import Path
from ultralytics import YOLO

# Import project modules
from src.phase1.feature_extractor import FeatureExtractor
from src.phase1.sci_compressor import SCICompressor
from src.phase4.dqn_agent import DQN

def profile_component(func, *args, n_runs=100, warmup=10):
    """Profile function latency with warmup."""
    # Warmup runs
    for _ in range(warmup):
        _ = func(*args)
    
    # Actual timing runs
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        result = func(*args)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)  # Convert to ms
    
    return {
        'mean_ms': np.mean(times),
        'std_ms': np.std(times),
        'min_ms': np.min(times),
        'max_ms': np.max(times),
        'p50_ms': np.percentile(times, 50),
        'p95_ms': np.percentile(times, 95),
        'p99_ms': np.percentile(times, 99)
    }

def load_test_video(video_path):
    """Load a short video clip for testing"""
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    
    while len(frames) < 300:  # Load first 300 frames
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    
    cap.release()
    print(f"✅ Loaded {len(frames)} frames from {video_path}")
    return frames

def profile_state_extraction(frames, n_runs=100):
    """Profile state feature extraction"""
    print("\n" + "="*80)
    print("1. STATE EXTRACTION PROFILING")
    print("="*80)
    
    extractor = FeatureExtractor()
    
    # Test with middle frames
    idx = len(frames) // 2
    frame = frames[idx]
    prev_frame = frames[idx - 1]
    
    def extract_features():
        return extractor.extract_features(frame, prev_frame)
    
    stats = profile_component(extract_features, n_runs=n_runs)
    
    print(f"\n{'Metric':<20} {'Value (ms)':<15}")
    print("-"*35)
    print(f"{'Mean':<20} {stats['mean_ms']:<15.2f}")
    print(f"{'Std Dev':<20} {stats['std_ms']:<15.2f}")
    print(f"{'Min':<20} {stats['min_ms']:<15.2f}")
    print(f"{'Max':<20} {stats['max_ms']:<15.2f}")
    print(f"{'Median (P50)':<20} {stats['p50_ms']:<15.2f}")
    print(f"{'P95':<20} {stats['p95_ms']:<15.2f}")
    print(f"{'P99':<20} {stats['p99_ms']:<15.2f}")
    
    return stats

def profile_dqn_inference(n_runs=100):
    """Profile DQN action selection"""
    print("\n" + "="*80)
    print("2. DQN INFERENCE PROFILING")
    print("="*80)
    
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = Path('runs/rl_training_adaptive/best_model_adaptive.pth')
    
    if not model_path.exists():
        print(f"❌ Model not found at {model_path}")
        return None
    
    # Create DQN model
    dqn = DQN(state_dim=7, hidden_dim=128, action_dim=3)
    checkpoint = torch.load(model_path, map_location=device)
    dqn.load_state_dict(checkpoint['model_state_dict'])
    dqn.to(device)
    dqn.eval()
    
    # Create dummy state
    dummy_state = torch.randn(1, 7).to(device)
    
    def dqn_forward():
        with torch.no_grad():
            return dqn(dummy_state)
    
    stats = profile_component(dqn_forward, n_runs=n_runs)
    
    print(f"\n{'Metric':<20} {'Value (ms)':<15}")
    print("-"*35)
    print(f"{'Mean':<20} {stats['mean_ms']:<15.3f}")
    print(f"{'Std Dev':<20} {stats['std_ms']:<15.3f}")
    print(f"{'Min':<20} {stats['min_ms']:<15.3f}")
    print(f"{'Max':<20} {stats['max_ms']:<15.3f}")
    print(f"{'Median (P50)':<20} {stats['p50_ms']:<15.3f}")
    print(f"{'P95':<20} {stats['p95_ms']:<15.3f}")
    
    return stats

def profile_sci_compression(frames, n_runs=50):
    """Profile SCI compression"""
    print("\n" + "="*80)
    print("3. SCI COMPRESSION PROFILING")
    print("="*80)
    
    compressor = SCICompressor()
    
    # Use first 50 frames for testing
    test_frames = frames[:50]
    
    def compress_frames():
        return compressor.compress_frames(test_frames, B=12)
    
    stats = profile_component(compress_frames, n_runs=n_runs, warmup=5)
    
    # Divide by number of frames to get per-frame latency
    per_frame_stats = {k: v / len(test_frames) for k, v in stats.items()}
    
    print(f"\n{'Metric':<20} {'Total (ms)':<15} {'Per Frame (ms)':<15}")
    print("-"*50)
    print(f"{'Mean':<20} {stats['mean_ms']:<15.2f} {per_frame_stats['mean_ms']:<15.2f}")
    print(f"{'Std Dev':<20} {stats['std_ms']:<15.2f} {per_frame_stats['std_ms']:<15.2f}")
    print(f"{'Min':<20} {stats['min_ms']:<15.2f} {per_frame_stats['min_ms']:<15.2f}")
    print(f"{'Max':<20} {stats['max_ms']:<15.2f} {per_frame_stats['max_ms']:<15.2f}")
    
    return per_frame_stats

def profile_yolo_inference(frames, n_runs=50):
    """Profile YOLOv8 inference"""
    print("\n" + "="*80)
    print("4. YOLOv8n INFERENCE PROFILING")
    print("="*80)
    
    # Load YOLO model
    model_path = Path('runs/train/yolo_cure_tsd/weights/best.pt')
    
    if not model_path.exists():
        print(f"❌ Model not found at {model_path}")
        # Try pretrained model
        model_path = Path('models/yolov8n.pt')
        if not model_path.exists():
            print("❌ No YOLO model found")
            return None
    
    model = YOLO(str(model_path))
    
    # Test with middle frame
    test_frame = frames[len(frames) // 2]
    
    def yolo_detect():
        return model(test_frame, verbose=False)
    
    stats = profile_component(yolo_detect, n_runs=n_runs)
    
    print(f"\n{'Metric':<20} {'Value (ms)':<15}")
    print("-"*35)
    print(f"{'Mean':<20} {stats['mean_ms']:<15.2f}")
    print(f"{'Std Dev':<20} {stats['std_ms']:<15.2f}")
    print(f"{'Min':<20} {stats['min_ms']:<15.2f}")
    print(f"{'Max':<20} {stats['max_ms']:<15.2f}")
    print(f"{'Median (P50)':<20} {stats['p50_ms']:<15.2f}")
    print(f"{'P95':<20} {stats['p95_ms']:<15.2f}")
    
    return stats

def generate_summary(state_stats, dqn_stats, sci_stats, yolo_stats):
    """Generate summary report"""
    print("\n" + "="*80)
    print(" "*25 + "PIPELINE LATENCY SUMMARY")
    print("="*80)
    
    components = [
        ('State Extraction', state_stats),
        ('DQN Inference', dqn_stats),
        ('SCI Compression', sci_stats),
        ('YOLO Detection', yolo_stats)
    ]
    
    print(f"\n{'Component':<25} {'Mean (ms)':<15} {'Std (ms)':<15} {'P95 (ms)':<15}")
    print("-"*70)
    
    total_mean = 0
    for name, stats in components:
        if stats is not None:
            print(f"{name:<25} {stats['mean_ms']:<15.2f} {stats['std_ms']:<15.2f} {stats['p95_ms']:<15.2f}")
            total_mean += stats['mean_ms']
        else:
            print(f"{name:<25} {'N/A':<15} {'N/A':<15} {'N/A':<15}")
    
    print("-"*70)
    print(f"{'TOTAL PIPELINE':<25} {total_mean:<15.2f}")
    print("="*70)
    
    # Calculate FPS
    fps = 1000 / total_mean if total_mean > 0 else 0
    target_fps = 30
    
    print(f"\n📊 Performance Metrics:")
    print(f"   • Total Latency: {total_mean:.2f} ms")
    print(f"   • Effective FPS: {fps:.1f} FPS")
    print(f"   • Target FPS: {target_fps} FPS")
    
    if fps >= target_fps:
        print(f"   ✅ REAL-TIME CAPABLE (meets {target_fps} FPS target)")
    else:
        gap = target_fps - fps
        print(f"   ⚠️  {gap:.1f} FPS below target (near real-time)")
    
    # Save results
    results = {
        'components': {
            name: stats for name, stats in components if stats is not None
        },
        'total_latency_ms': total_mean,
        'fps': fps,
        'real_time_capable': fps >= target_fps
    }
    
    import json
    output_file = Path('outputs') / 'latency_profile.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Results saved to: {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Profile pipeline latency')
    parser.add_argument('--video', type=str, help='Path to test video (optional)')
    parser.add_argument('--runs', type=int, default=100, help='Number of profiling runs')
    args = parser.parse_args()
    
    print("\n" + "🚀 PIPELINE LATENCY PROFILING")
    print("="*80)
    
    if args.video:
        frames = load_test_video(args.video)
    else:
        print("\n⚠️  No test video provided. Using synthetic frames for profiling.")
        print("   (Provide --video <path> for accurate measurements)")
        # Create synthetic frames
        frames = [np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8) for _ in range(300)]
    
    # Profile each component
    state_stats = profile_state_extraction(frames, n_runs=args.runs)
    dqn_stats = profile_dqn_inference(n_runs=args.runs)
    sci_stats = profile_sci_compression(frames, n_runs=min(args.runs, 50))
    yolo_stats = profile_yolo_inference(frames, n_runs=min(args.runs, 50))
    
    # Generate summary
    generate_summary(state_stats, dqn_stats, sci_stats, yolo_stats)

if __name__ == "__main__":
    main()
