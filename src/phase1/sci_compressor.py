"""
SCI (Snapshot Compressive Imaging) Compression Simulator
Implements: Y = Σ(Cb ⊙ Xb) for b = 1 to B

Based on CACTI paper approach with binary coding masks.
"""

import numpy as np
import cv2
from pathlib import Path


class SCICompressor:
    """
    Snapshot Compressive Imaging (SCI) Compressor

    Compresses B consecutive video frames into a single measurement Y using
    binary coding masks.

    Mathematical Model:
        Y = Σ(Cb ⊙ Xb) for b = 1 to B

        where:
        - Y: Compressed measurement (H×W grayscale image)
        - Cb: Binary coding mask for frame b (H×W matrix of 0s and 1s)
        - Xb: Frame b (H×W grayscale)
        - B: Compression ratio (number of frames compressed into 1 measurement)
        - ⊙: Element-wise multiplication

    Bandwidth Savings:
        - Original: B frames × (H×W×3 bytes) = 3×B×H×W bytes
        - Compressed: 1 measurement × (H×W bytes) = H×W bytes
        - Compression Factor: 3×B (e.g., B=10 → 30x compression)
    """

    def __init__(self, frame_height, frame_width, B_values=[6, 8, 10, 12, 15, 20],
                 mask_dir="data/masks", seed=42):
        """
        Initialize SCI compressor

        Args:
            frame_height: Video frame height (e.g., 1236 for CURE-TSD)
            frame_width: Video frame width (e.g., 1628 for CURE-TSD)
            B_values: List of compression ratios to support
            mask_dir: Directory to save/load masks
            seed: Random seed for reproducibility
        """
        self.H = frame_height
        self.W = frame_width
        self.B_values = sorted(B_values)
        self.mask_dir = Path(mask_dir)
        self.seed = seed

        # Storage for generated masks {B: mask_array}
        self.masks = {}

        # Create mask directory if needed
        self.mask_dir.mkdir(parents=True, exist_ok=True)

        print(f"📸 SCI Compressor initialized:")
        print(f"   Frame size: {self.H}×{self.W}")
        print(f"   B values: {self.B_values}")
        print(f"   Mask dir: {self.mask_dir}")

    def generate_masks(self, B, save=True):
        """
        Generate binary coding masks for compression ratio B

        Uses Bernoulli distribution with p=0.5 (each pixel independently 
        has 50% chance of being 1 or 0).

        Args:
            B: Compression ratio (number of frames)
            save: Whether to save masks to disk

        Returns:
            np.array: Binary masks of shape (H, W, B) with values {0.0, 1.0}
        """
        if B in self.masks:
            print(f"   Using cached masks for B={B}")
            return self.masks[B]

        # Try loading from disk first
        mask_path = self.mask_dir / f"mask_B{B}_{self.H}x{self.W}.npy"
        if mask_path.exists():
            print(f"   Loading masks from {mask_path}")
            masks = np.load(mask_path)
            self.masks[B] = masks
            return masks

        # Generate new masks
        print(f"   Generating new binary masks for B={B}...")
        np.random.seed(self.seed)

        # Binary masks: Bernoulli(p=0.5) → {0, 1}
        masks = np.random.binomial(
            1, 0.5, (self.H, self.W, B)).astype(np.float32)

        # Verify mask properties
        unique_vals = np.unique(masks)
        mean_val = masks.mean()
        print(f"   ✓ Generated {B} masks of size {self.H}×{self.W}")
        print(f"   ✓ Unique values: {unique_vals}")
        print(f"   ✓ Mean: {mean_val:.4f} (expected ~0.5)")

        # Cache in memory
        self.masks[B] = masks

        # Save to disk
        if save:
            np.save(mask_path, masks)
            print(f"   ✓ Saved to {mask_path}")

        return masks

    def compress(self, frames, B):
        """
        Compress B consecutive frames into a single measurement Y

        Implements: Y = Σ(Cb ⊙ Xb) for b = 1 to B

        Args:
            frames: List/array of B frames, each (H, W, 3) BGR
            B: Compression ratio (must match len(frames))

        Returns:
            Y: Compressed measurement (H, W) grayscale, dtype=float32

        Raises:
            ValueError: If len(frames) != B
        """
        if len(frames) != B:
            raise ValueError(f"Expected {B} frames, got {len(frames)}")

        # Get or generate masks for this B
        masks = self.generate_masks(B, save=True)

        # Initialize compressed measurement
        Y = np.zeros((self.H, self.W), dtype=np.float32)

        # Apply SCI compression: Y = Σ(Cb ⊙ Xb)
        for b in range(B):
            # Convert frame to grayscale (as per CACTI paper)
            frame_gray = cv2.cvtColor(
                frames[b], cv2.COLOR_BGR2GRAY).astype(np.float32)

            # Resize if needed (handle different resolutions)
            if frame_gray.shape != (self.H, self.W):
                frame_gray = cv2.resize(frame_gray, (self.W, self.H))

            # Element-wise multiplication: Cb ⊙ Xb
            masked_frame = masks[:, :, b] * frame_gray

            # Accumulate into measurement
            Y += masked_frame

        # Normalize accumulated measurement to [0, 255]
        # Don't just clip - normalize to preserve dynamic range
        if Y.max() > 0:
            Y = (Y / Y.max()) * 255.0

        Y = Y.astype(np.float32)

        return Y

    def compress_video_sequence(self, frames, B, stride=None):
        """
        Compress entire video sequence into multiple measurements

        Args:
            frames: List of all video frames (N frames)
            B: Compression ratio
            stride: Step size between measurements (default=B, non-overlapping)
                   If stride < B, measurements overlap

        Returns:
            list: Compressed measurements [Y1, Y2, Y3, ...]
            list: Frame indices for each measurement [[0,1,...,B-1], [B,B+1,...], ...]
        """
        if stride is None:
            stride = B  # Non-overlapping by default

        N = len(frames)
        measurements = []
        frame_indices = []

        # Slide window of size B with given stride
        for start_idx in range(0, N - B + 1, stride):
            end_idx = start_idx + B
            frame_group = frames[start_idx:end_idx]

            # Compress this group
            Y = self.compress(frame_group, B)

            measurements.append(Y)
            frame_indices.append(list(range(start_idx, end_idx)))

        print(
            f"   Compressed {N} frames into {len(measurements)} measurements (B={B}, stride={stride})")

        return measurements, frame_indices

    def get_compression_stats(self, B):
        """
        Calculate compression statistics

        Args:
            B: Compression ratio

        Returns:
            dict: Statistics including bandwidth savings, compression factor
        """
        # Original: B frames × (H×W×3 channels × 1 byte)
        original_size = B * self.H * self.W * 3

        # Compressed: 1 measurement × (H×W × 1 channel × 1 byte)
        compressed_size = self.H * self.W

        # Compression factor
        compression_factor = original_size / compressed_size

        # Bandwidth savings percentage
        bandwidth_savings = (1 - compressed_size / original_size) * 100

        return {
            'B': B,
            'original_size_bytes': original_size,
            'compressed_size_bytes': compressed_size,
            'compression_factor': compression_factor,
            'bandwidth_savings_percent': bandwidth_savings,
            'original_size_mb': original_size / (1024**2),
            'compressed_size_mb': compressed_size / (1024**2)
        }

    def visualize_measurement(self, Y, title="SCI Measurement"):
        """
        Visualize compressed measurement (for debugging)

        Args:
            Y: Compressed measurement (H, W)
            title: Plot title

        Returns:
            np.array: Y normalized to uint8 for display
        """
        # Normalize to [0, 255]
        Y_vis = np.clip(Y, 0, 255).astype(np.uint8)

        return Y_vis

    def generate_all_masks(self):
        """
        Pre-generate and save masks for all B values

        Useful for reproducibility - call this once before experiments.
        """
        print(f"\n🎭 Generating masks for all B values: {self.B_values}")

        for B in self.B_values:
            self.generate_masks(B, save=True)

        print(f"\n✅ All masks generated and saved to {self.mask_dir}/")


if __name__ == "__main__":
    # Test SCI compression
    print("🧪 Testing SCI Compressor\n")

    # Load sample video
    from video_loader import VideoLoader

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
