"""
Test SCI (Snapshot Compressive Imaging) Compression
"""

from pathlib import Path
import cv2
from video_loader import VideoLoader
from sci_compressor import SCICompressor
import sys
sys.path.insert(0, 'src/phase1')


if __name__ == "__main__":
    # Test SCI compression
    print("🧪 Testing SCI Compressor\n")

    # Load sample video
    video_path = "data/cure-tsd/data/01_01_00_00_00.mp4"
    print(f"Loading video: {video_path}\n")

    loader = VideoLoader(video_path)

    # Get video dimensions
    sample_frame = loader.get_frame(0)
    H, W = sample_frame.shape[:2]

    print(f"Video dimensions: {H}×{W}\n")

    # Initialize compressor
    compressor = SCICompressor(
        frame_height=H,
        frame_width=W,
        B_values=[6, 8, 10, 12, 15, 20]
    )

    # Test 1: Generate masks for all B values
    print("\n" + "="*60)
    print("TEST 1: Generate masks for all B values")
    print("="*60)
    compressor.generate_all_masks()

    # Test 2: Compress 10 frames with B=10
    print("\n" + "="*60)
    print("TEST 2: Compress 10 frames with B=10")
    print("="*60)

    B_test = 10
    frames = [loader.get_frame(i) for i in range(B_test)]

    print(f"\nCompressing frames 0-{B_test-1}...")
    Y = compressor.compress(frames, B_test)

    print(f"\n📊 Compression Result:")
    print(f"   Input: {B_test} frames × {H}×{W}×3")
    print(f"   Output: 1 measurement × {H}×{W}")
    print(f"   Y shape: {Y.shape}")
    print(f"   Y dtype: {Y.dtype}")
    print(f"   Y range: [{Y.min():.2f}, {Y.max():.2f}]")
    print(f"   Y mean: {Y.mean():.2f}")

    # Test 3: Compression statistics
    print("\n" + "="*60)
    print("TEST 3: Compression Statistics")
    print("="*60)

    for B in [6, 10, 15, 20]:
        stats = compressor.get_compression_stats(B)
        print(f"\nB={B}:")
        print(f"   Original: {stats['original_size_mb']:.2f} MB ({B} frames)")
        print(
            f"   Compressed: {stats['compressed_size_mb']:.2f} MB (1 measurement)")
        print(f"   Compression: {stats['compression_factor']:.1f}x")
        print(
            f"   Bandwidth savings: {stats['bandwidth_savings_percent']:.1f}%")

    # Test 4: Compress entire video sequence
    print("\n" + "="*60)
    print("TEST 4: Compress entire video sequence")
    print("="*60)

    # Load first 30 frames
    all_frames = [loader.get_frame(i) for i in range(30)]

    B_test = 10
    measurements, indices = compressor.compress_video_sequence(
        all_frames, B_test, stride=10)

    print(f"\n📊 Sequence Compression:")
    print(f"   Input: {len(all_frames)} frames")
    print(f"   Output: {len(measurements)} measurements")
    print(f"   Frame indices: {indices}")

    # Test 5: Save a measurement for visual inspection
    print("\n" + "="*60)
    print("TEST 5: Save measurement visualization")
    print("="*60)

    Y_vis = compressor.visualize_measurement(
        measurements[0], f"SCI Measurement (B={B_test})")

    output_path = "outputs/sci_test/measurement_B10.jpg"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(output_path, Y_vis)
    print(f"\n✅ Saved measurement to {output_path}")

    print("\n" + "="*60)
    print("✅ All SCI Compressor tests passed!")
    print("="*60)
