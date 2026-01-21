"""
Visual Testing of SCI Compression
Compare original frames vs compressed measurements for different B values
"""

import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import cv2
from video_loader import VideoLoader
from sci_compressor import SCICompressor
import sys
sys.path.insert(0, 'src/phase1')


def create_comparison_visualization(B_values=[6, 10, 15, 20]):
    """Create side-by-side comparison of original frames and SCI measurements"""

    print(f"🎨 Creating SCI Compression Visualization\n")

    # Load video
    video_path = "data/cure-tsd/data/01_01_00_00_00.mp4"
    loader = VideoLoader(video_path)

    # Get video dimensions
    sample_frame = loader.get_frame(0)
    H, W = sample_frame.shape[:2]

    # Initialize compressor
    compressor = SCICompressor(
        frame_height=H,
        frame_width=W,
        B_values=B_values
    )

    output_dir = Path("outputs/sci_test/comparisons")
    output_dir.mkdir(parents=True, exist_ok=True)

    for B in B_values:
        print(f"\n{'='*60}")
        print(f"Testing B={B} (compress {B} frames → 1 measurement)")
        print(f"{'='*60}")

        # Load B consecutive frames starting from frame 50
        start_frame = 50
        frames = [loader.get_frame(start_frame + i) for i in range(B)]

        print(f"Loaded frames {start_frame} to {start_frame + B - 1}")

        # Compress
        Y = compressor.compress(frames, B)

        print(f"Compressed measurement Y:")
        print(f"  Shape: {Y.shape}")
        print(f"  Range: [{Y.min():.2f}, {Y.max():.2f}]")
        print(f"  Mean: {Y.mean():.2f}")
        print(f"  Std: {Y.std():.2f}")

        # Create visualization
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(f'SCI Compression B={B} (Frames {start_frame}-{start_frame+B-1})',
                     fontsize=16, fontweight='bold')

        # Show first 3 original frames
        for i in range(min(3, B)):
            ax = axes[0, i]
            frame_rgb = cv2.cvtColor(frames[i], cv2.COLOR_BGR2RGB)
            ax.imshow(frame_rgb)
            ax.set_title(f'Original Frame {start_frame + i}')
            ax.axis('off')

        # Show compressed measurement (same in all bottom panels)
        for i in range(3):
            ax = axes[1, i]
            if i == 0:
                # Grayscale
                ax.imshow(Y, cmap='gray', vmin=0, vmax=255)
                ax.set_title(f'Compressed Y (Grayscale)')
            elif i == 1:
                # Enhanced contrast
                Y_norm = cv2.normalize(
                    Y, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                Y_eq = cv2.equalizeHist(Y_norm)
                ax.imshow(Y_eq, cmap='gray')
                ax.set_title(f'Y (Contrast Enhanced)')
            else:
                # Histogram
                ax.hist(Y.ravel(), bins=50, color='blue', alpha=0.7)
                ax.set_title(f'Pixel Intensity Histogram')
                ax.set_xlabel('Pixel Value')
                ax.set_ylabel('Frequency')
                ax.grid(True, alpha=0.3)

            if i < 2:
                ax.axis('off')

        plt.tight_layout()

        # Save
        save_path = output_dir / f"comparison_B{B}.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ Saved visualization to {save_path}")
        plt.close()

        # Also save individual Y measurement
        Y_uint8 = np.clip(Y, 0, 255).astype(np.uint8)
        y_path = output_dir / f"measurement_B{B}.jpg"
        cv2.imwrite(str(y_path), Y_uint8)

        # Save contrast-enhanced version
        Y_enhanced = cv2.equalizeHist(Y_uint8)
        y_enhanced_path = output_dir / f"measurement_B{B}_enhanced.jpg"
        cv2.imwrite(str(y_enhanced_path), Y_enhanced)
        print(f"✅ Saved Y to {y_path}")
        print(f"✅ Saved enhanced Y to {y_enhanced_path}")


def test_different_scenes():
    """Test SCI compression on different video frames (different scenes)"""

    print(f"\n{'='*60}")
    print(f"Testing SCI on Different Scenes")
    print(f"{'='*60}\n")

    video_path = "data/cure-tsd/data/01_01_00_00_00.mp4"
    loader = VideoLoader(video_path)

    sample_frame = loader.get_frame(0)
    H, W = sample_frame.shape[:2]

    compressor = SCICompressor(frame_height=H, frame_width=W, B_values=[10])

    output_dir = Path("outputs/sci_test/scenes")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Test on different frame ranges
    test_ranges = [
        (0, 10, "Beginning"),
        (50, 60, "Early"),
        (100, 110, "Middle"),
        (150, 160, "Late"),
        (200, 210, "Near End")
    ]

    for start, end, label in test_ranges:
        frames = [loader.get_frame(i) for i in range(start, end)]
        Y = compressor.compress(frames, B=10)

        print(f"{label} (frames {start}-{end-1}):")
        print(f"  Y range: [{Y.min():.2f}, {Y.max():.2f}]")
        print(f"  Y mean: {Y.mean():.2f}")
        print(f"  Y std: {Y.std():.2f}")

        # Save
        Y_uint8 = np.clip(Y, 0, 255).astype(np.uint8)
        save_path = output_dir / \
            f"scene_{label.lower().replace(' ', '_')}_B10.jpg"
        cv2.imwrite(str(save_path), Y_uint8)

        # Enhanced version
        Y_enhanced = cv2.equalizeHist(Y_uint8)
        enhanced_path = output_dir / \
            f"scene_{label.lower().replace(' ', '_')}_B10_enhanced.jpg"
        cv2.imwrite(str(enhanced_path), Y_enhanced)

    print(f"\n✅ Scene comparisons saved to {output_dir}/")


def inspect_mask_effect():
    """Visualize how binary masks affect the compression"""

    print(f"\n{'='*60}")
    print(f"Inspecting Binary Mask Effect")
    print(f"{'='*60}\n")

    video_path = "data/cure-tsd/data/01_01_00_00_00.mp4"
    loader = VideoLoader(video_path)

    sample_frame = loader.get_frame(0)
    H, W = sample_frame.shape[:2]

    compressor = SCICompressor(frame_height=H, frame_width=W, B_values=[10])

    # Load 10 frames
    frames = [loader.get_frame(i) for i in range(50, 60)]

    # Get masks
    masks = compressor.generate_masks(10)

    print(f"Mask properties:")
    print(f"  Shape: {masks.shape}")
    print(f"  Dtype: {masks.dtype}")
    print(f"  Unique values: {np.unique(masks)}")
    print(f"  Mean per mask:")
    for i in range(10):
        print(f"    Mask {i}: {masks[:, :, i].mean():.4f}")

    # Manual compression to see intermediate steps
    Y_manual = np.zeros((H, W), dtype=np.float32)

    output_dir = Path("outputs/sci_test/mask_effect")
    output_dir.mkdir(parents=True, exist_ok=True)

    for b in range(10):
        # Convert to grayscale
        frame_gray = cv2.cvtColor(
            frames[b], cv2.COLOR_BGR2GRAY).astype(np.float32)

        # Apply mask
        masked = masks[:, :, b] * frame_gray

        # Accumulate
        Y_manual += masked

        # Save masked frame
        masked_uint8 = np.clip(masked, 0, 255).astype(np.uint8)
        masked_path = output_dir / f"masked_frame_{b}.jpg"
        cv2.imwrite(str(masked_path), masked_uint8)

        print(f"  Frame {b}: gray_range=[{frame_gray.min():.0f}, {frame_gray.max():.0f}], "
              f"masked_range=[{masked.min():.0f}, {masked.max():.0f}], "
              f"Y_accum_mean={Y_manual.mean():.2f}")

    print(f"\nFinal Y (manual):")
    print(f"  Range: [{Y_manual.min():.2f}, {Y_manual.max():.2f}]")
    print(f"  Mean: {Y_manual.mean():.2f}")

    # Compare with automatic compression
    Y_auto = compressor.compress(frames, 10)
    print(f"\nFinal Y (automatic):")
    print(f"  Range: [{Y_auto.min():.2f}, {Y_auto.max():.2f}]")
    print(f"  Mean: {Y_auto.mean():.2f}")

    print(f"\nMatch: {np.allclose(Y_manual, Y_auto)}")

    print(f"\n✅ Mask effect visualization saved to {output_dir}/")


if __name__ == "__main__":
    print("="*60)
    print("SCI COMPRESSION VISUAL TESTING")
    print("="*60)

    # Test 1: Create comparison visualizations
    create_comparison_visualization(B_values=[6, 10, 15, 20])

    # Test 2: Different scenes
    test_different_scenes()

    # Test 3: Inspect mask effect
    inspect_mask_effect()

    print("\n" + "="*60)
    print("✅ ALL VISUAL TESTS COMPLETE")
    print("="*60)
    print("\nCheck outputs/sci_test/ for results:")
    print("  - comparisons/: Side-by-side original vs compressed")
    print("  - scenes/: Different video scenes")
    print("  - mask_effect/: Individual masked frames")
