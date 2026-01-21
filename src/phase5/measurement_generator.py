"""
Phase 5: Measurement Generator
Generate SCI compressed measurements from CURE-TSD videos
"""

from typing import List, Tuple, Optional
import numpy as np
import cv2
from pathlib import Path
from video_loader import VideoLoader
from sci_compressor import SCICompressor
import sys
sys.path.insert(0, 'src/phase1')


class MeasurementGenerator:
    """
    Generate SCI compressed measurements from videos

    Wraps SCICompressor to batch-process videos and save measurements as images.
    """

    def __init__(self, B_values=[6, 8, 10, 12, 15, 20],
                 output_dir="data/cure-tsd-yolo/measurements",
                 frame_height=1236, frame_width=1628):
        """
        Initialize measurement generator

        Args:
            B_values: List of compression ratios to generate
            output_dir: Directory to save compressed measurements
            frame_height: Video frame height
            frame_width: Video frame width
        """
        self.B_values = B_values
        self.output_dir = Path(output_dir)
        self.frame_height = frame_height
        self.frame_width = frame_width

        # Initialize SCI compressor
        self.compressor = SCICompressor(
            frame_height=frame_height,
            frame_width=frame_width,
            B_values=B_values
        )

        # Pre-generate all masks
        print("🎭 Pre-generating masks for all B values...")
        self.compressor.generate_all_masks()

        print(f"\n📸 Measurement Generator initialized:")
        print(f"   B values: {B_values}")
        print(f"   Output directory: {output_dir}")

    def generate_measurements_from_video(self, video_path: Path, B: int,
                                         stride: Optional[int] = None,
                                         video_id: str = None) -> List[Tuple[Tuple[int, int], Path]]:
        """
        Generate compressed measurements from a video

        Args:
            video_path: Path to video file
            B: Compression ratio
            stride: Step size between measurements (default=B)
            video_id: Identifier for naming (e.g., '01_01')

        Returns:
            List of ((start_frame, end_frame), saved_image_path) tuples
        """
        if stride is None:
            stride = B

        if video_id is None:
            video_id = video_path.stem

        # Load video
        loader = VideoLoader(str(video_path))
        total_frames = loader.metadata['frame_count']

        print(f"\n📹 Processing: {video_path.name}")
        print(f"   Total frames: {total_frames}")
        print(f"   B={B}, stride={stride}")

        measurements = []

        # Generate measurements
        for start_frame in range(0, total_frames - B + 1, stride):
            end_frame = start_frame + B - 1

            # Load frames
            frames = [loader.get_frame(i)
                      for i in range(start_frame, end_frame + 1)]

            # Compress
            Y = self.compressor.compress(frames, B)

            # Save as image
            output_filename = f"{video_id}_frames_{start_frame:03d}-{end_frame:03d}_B{B:02d}.jpg"
            output_path = self.output_dir / output_filename
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Convert to uint8 and save
            Y_uint8 = np.clip(Y, 0, 255).astype(np.uint8)
            cv2.imwrite(str(output_path), Y_uint8)

            measurements.append(((start_frame, end_frame), output_path))

        print(f"   ✅ Generated {len(measurements)} measurements")

        return measurements

    def generate_measurements_for_dataset(self, video_list: List[Path],
                                          B_values: Optional[List[int]] = None,
                                          stride: Optional[int] = None) -> dict:
        """
        Generate measurements for multiple videos

        Args:
            video_list: List of video paths
            B_values: List of B values to generate (default: self.B_values)
            stride: Step size (default: B for each)

        Returns:
            Dict mapping (video_id, B) -> list of measurements
        """
        if B_values is None:
            B_values = self.B_values

        all_measurements = {}

        for video_path in video_list:
            video_id = video_path.stem

            for B in B_values:
                key = (video_id, B)
                measurements = self.generate_measurements_from_video(
                    video_path, B, stride=stride, video_id=video_id
                )
                all_measurements[key] = measurements

        return all_measurements


if __name__ == "__main__":
    print("🧪 Testing Measurement Generator\n")

    # Test on single video
    video_path = Path("data/cure-tsd/data/01_01_00_00_00.mp4")

    if not video_path.exists():
        print(f"❌ Video not found: {video_path}")
        exit(1)

    # Initialize generator
    generator = MeasurementGenerator(
        B_values=[10, 15],  # Test with 2 B values
        output_dir="outputs/measurements_test"
    )

    # Generate measurements with B=10
    print(f"\n{'='*70}")
    print("Test: Generate measurements with B=10")
    print(f"{'='*70}")

    measurements_B10 = generator.generate_measurements_from_video(
        video_path,
        B=10,
        stride=10,
        video_id="01_01"
    )

    print(f"\n📊 Results:")
    print(f"   Generated {len(measurements_B10)} measurements")
    print(f"\n   Sample measurements:")
    for i, (frame_range, img_path) in enumerate(measurements_B10[:3]):
        print(
            f"   {i+1}. Frames {frame_range[0]}-{frame_range[1]} → {img_path.name}")

    # Generate with B=15
    print(f"\n{'='*70}")
    print("Test: Generate measurements with B=15")
    print(f"{'='*70}")

    measurements_B15 = generator.generate_measurements_from_video(
        video_path,
        B=15,
        stride=15,
        video_id="01_01"
    )

    print(f"\n📊 Results:")
    print(f"   Generated {len(measurements_B15)} measurements")

    print(f"\n{'='*70}")
    print("✅ Measurement Generator Test Complete!")
    print(f"{'='*70}")
    print(f"\nCheck output: outputs/measurements_test/")
