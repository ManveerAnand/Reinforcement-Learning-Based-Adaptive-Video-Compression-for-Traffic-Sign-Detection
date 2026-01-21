"""
Integration Test for Phase 1 (without Gym environment due to installation issues)
Tests: Label Parser + Video Loader + Feature Extractor
"""

from phase1.feature_extractor import FeatureExtractor
from phase1.video_loader import VideoLoader
from phase1.label_parser import CURETSDLabelParser
import sys
sys.path.insert(0, 'src')


print("=" * 60)
print("PHASE 1 INTEGRATION TEST")
print("=" * 60)

# Test 1: Label Parser
print("\n📋 Test 1: Label Parser")
print("-" * 60)

label_path = "data/cure-tsd/labels/01_01.txt"
parser = CURETSDLabelParser(label_path)

print(f"✓ Loaded labels: {label_path}")
print(f"  Total frames: {parser.get_total_frames()}")
print(f"  Frames with signs: {len(parser.annotations)}")

# Get stats
stats = parser.get_statistics()
print(f"  Total signs: {stats['total_signs']}")

# Show first annotated frame
first_frame = min(parser.annotations.keys())
labels = parser.get_frame_labels(first_frame)
print(f"\n  Frame {first_frame} labels:")
for label in labels:
    print(f"    - {label['class_name']} (ID={label['class_id']})")
    print(
        f"      bbox=[{label['x_min']}, {label['y_min']}, {label['x_max']}, {label['y_max']}]")

# Test 2: Video Loader
print("\n🎥 Test 2: Video Loader")
print("-" * 60)

video_path = "data/cure-tsd/data/01_01_00_00_00.mp4"
loader = VideoLoader(video_path)

print(f"✓ Loaded video: {video_path}")
print(f"  Resolution: {loader.metadata['width']}x{loader.metadata['height']}")
print(f"  FPS: {loader.metadata['fps']}")
print(f"  Total frames: {len(loader)}")

# Load first frame
frame0 = loader.get_frame(0)
print(f"\n  First frame:")
print(f"    Shape: {frame0.shape}")
print(f"    Type: {frame0.dtype}")
print(f"    Mean intensity: {frame0.mean():.2f}")

# Test 3: Feature Extractor
print("\n🔬 Test 3: Feature Extractor")
print("-" * 60)

extractor = FeatureExtractor()

frame1 = loader.get_frame(1)
frame10 = loader.get_frame(10)

# Create dummy detections from ground truth
gt_labels = parser.get_frame_labels(1)
detections = [
    {'confidence': 0.95,
        'class_id': label['class_id'], 'is_critical': label['is_critical']}
    for label in gt_labels
]

print(f"✓ Extracting 7D state (frame 0 → 1)")

# Extract state
state = extractor.extract_state(frame1, frame0, detections, current_B=10)

print(f"\n  7D State Vector:")
print(f"    Shape: {state.shape}")
print(f"    Type: {state.dtype}")
print(f"    Values: {state}")

print(f"\n  Feature Breakdown:")
print(f"    1. Optical Flow:     {state[0]:.2f} px/frame")
print(f"    2. Edge Density:     {state[1]:.4f}")
print(f"    3. Sign Confidence:  {state[2]:.4f}")
print(f"    4. Blur Score:       {state[3]:.2f}")
print(f"    5. Brightness:       {state[4]:.2f}")
print(f"    6. Frame Difference: {state[5]:.4f}")
print(f"    7. B Normalized:     {state[6]:.4f} (B=10)")

# Test 4: End-to-end pipeline
print("\n🔄 Test 4: End-to-End Pipeline (10 frames)")
print("-" * 60)

print(f"{'Frame':<8} {'Flow':<8} {'Edges':<8} {'Conf':<8} {'Blur':<8} {'Brightness':<12} {'Diff':<8} {'B':<6}")
print("-" * 60)

prev_frame = loader.get_frame(0)
B = 10

for i in range(1, 11):
    curr_frame = loader.get_frame(i)
    gt_labels = parser.get_frame_labels(i)

    # Create detections
    dets = [
        {'confidence': 0.95,
            'class_id': l['class_id'], 'is_critical': l['is_critical']}
        for l in gt_labels
    ]

    # Extract state
    state = extractor.extract_state(curr_frame, prev_frame, dets, B)

    print(f"{i:<8} {state[0]:<8.2f} {state[1]:<8.4f} {state[2]:<8.4f} "
          f"{state[3]:<8.2f} {state[4]:<12.2f} {state[5]:<8.4f} {state[6]:<6.4f}")

    prev_frame = curr_frame

print("\n" + "=" * 60)
print("✅ PHASE 1 COMPLETE - All modules working!")
print("=" * 60)
print("\nSummary:")
print("  ✓ Label Parser: Parses CURE-TSD annotations")
print("  ✓ Video Loader: Loads MP4 frames efficiently")
print("  ✓ Feature Extractor: Computes 7D state vector")
print("  ✓ End-to-End: Pipeline processes video frames")
print("\nNote: Gym environment requires 'gymnasium' package")
print("      (Installation blocked by pip/conda errors on this system)")
print("\n🚀 Ready for Phase 2: YOLOv8 fine-tuning!")
