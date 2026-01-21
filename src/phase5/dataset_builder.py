"""
Phase 5: YOLO Dataset Builder
Build training/validation datasets for YOLOv8 using compressed measurements
"""

from typing import List, Dict
import time
import random
import yaml
import shutil
from pathlib import Path
from label_converter import LabelConverter
from measurement_generator import MeasurementGenerator
import sys
sys.path.insert(0, 'src/phase5')
sys.path.insert(0, 'src/phase1')


class YOLODatasetBuilder:
    """
    Build YOLO dataset from compressed SCI measurements

    Directory structure:
        dataset/
            images/
                train/
                    01_01_frames_000-009_B10.jpg
                    ...
                val/
                    02_01_frames_000-009_B10.jpg
                    ...
            labels/
                train/
                    01_01_frames_000-009_B10.txt
                    ...
                val/
                    02_01_frames_000-009_B10.txt
                    ...
            data.yaml
    """

    def __init__(self,
                 cure_tsd_videos_dir="data/cure-tsd/data",
                 cure_tsd_labels_dir="data/cure-tsd/labels",
                 output_dir="data/yolo_dataset",
                 B_values=[6, 8, 10, 12, 15, 20],
                 gt_strategy="union",
                 presence_threshold=3,
                 train_split=0.8,
                 random_seed=42):
        """
        Initialize dataset builder

        Args:
            cure_tsd_videos_dir: Directory containing CURE-TSD videos
            cure_tsd_labels_dir: Directory containing CURE-TSD labels
            output_dir: Output directory for YOLO dataset
            B_values: List of compression ratios
            gt_strategy: Ground truth strategy ('union' or 'representative')
            presence_threshold: Minimum frames a sign must appear
            train_split: Fraction of data for training
            random_seed: Random seed for reproducibility
        """
        self.videos_dir = Path(cure_tsd_videos_dir)
        self.labels_dir = Path(cure_tsd_labels_dir)
        self.output_dir = Path(output_dir)

        # Convert to absolute paths
        self.output_dir = self.output_dir.resolve()
        self.B_values = B_values
        self.gt_strategy = gt_strategy
        self.presence_threshold = presence_threshold
        self.train_split = train_split
        self.random_seed = random_seed

        # Set random seed
        random.seed(random_seed)

        # Initialize components
        self.measurement_generator = MeasurementGenerator(
            B_values=B_values,
            output_dir=str(self.output_dir / "temp_measurements")
        )

        self.label_converter = LabelConverter(
            strategy=gt_strategy,
            presence_threshold=presence_threshold
        )

        print(f"\n📦 YOLO Dataset Builder initialized:")
        print(f"   Videos: {cure_tsd_videos_dir}")
        print(f"   Labels: {cure_tsd_labels_dir}")
        print(f"   Output: {output_dir}")
        print(f"   B values: {B_values}")
        print(f"   GT strategy: {gt_strategy}")
        print(f"   Presence threshold: {presence_threshold}")
        print(f"   Train/Val split: {train_split:.0%}/{1-train_split:.0%}")

    def get_video_list(self) -> List[Path]:
        """Get all CURE-TSD video files"""
        videos = sorted(self.videos_dir.glob("*.mp4"))
        print(f"\n📹 Found {len(videos)} videos")
        return videos

    def split_train_val(self, videos: List[Path]) -> Dict[str, List[Path]]:
        """Split videos into train/val sets"""
        # Shuffle videos
        shuffled = videos.copy()
        random.shuffle(shuffled)

        # Split
        n_train = int(len(shuffled) * self.train_split)
        train_videos = shuffled[:n_train]
        val_videos = shuffled[n_train:]

        print(f"\n✂️ Split: {len(train_videos)} train, {len(val_videos)} val")

        return {
            'train': train_videos,
            'val': val_videos
        }

    def build_dataset(self, max_videos_per_split=None, mixed_B=True):
        """
        Build complete YOLO dataset

        Args:
            max_videos_per_split: Limit videos per split (for testing)
            mixed_B: If True, randomly sample B values for each video.
                     If False, generate all B values for each video.

        Returns:
            Path to data.yaml config file
        """
        print(f"\n{'='*70}")
        print("🏗️ Building YOLO Dataset")
        print(f"{'='*70}")

        # Get videos
        all_videos = self.get_video_list()

        if max_videos_per_split:
            all_videos = all_videos[:max_videos_per_split * 2]
            print(f"   ⚠️ Limiting to {len(all_videos)} videos for testing")

        # Split train/val
        split = self.split_train_val(all_videos)

        # Create directory structure
        for split_name in ['train', 'val']:
            (self.output_dir / 'images' /
             split_name).mkdir(parents=True, exist_ok=True)
            (self.output_dir / 'labels' /
             split_name).mkdir(parents=True, exist_ok=True)

        # Process each split
        for split_name, videos in split.items():
            print(f"\n{'='*70}")
            print(
                f"Processing {split_name.upper()} split ({len(videos)} videos)")
            print(f"{'='*70}")

            # Track timing for ETA
            start_time = time.time()
            processed_count = 0

            for i, video_path in enumerate(videos, 1):
                video_start_time = time.time()

                video_id = video_path.stem
                # Extract first two parts (e.g., "01_01" from "01_01_00_00_00")
                label_id = '_'.join(video_id.split('_')[:2])
                label_file = self.labels_dir / f"{label_id}.txt"

                if not label_file.exists():
                    print(
                        f"   ⚠️ [{i}/{len(videos)}] Skipping {video_id} (no labels)")
                    continue

                # Calculate ETA
                if processed_count > 0:
                    elapsed = time.time() - start_time
                    avg_time_per_video = elapsed / processed_count
                    remaining_videos = len(videos) - i
                    eta_seconds = avg_time_per_video * remaining_videos
                    eta_hours = int(eta_seconds // 3600)
                    eta_minutes = int((eta_seconds % 3600) // 60)
                    eta_str = f"ETA: {eta_hours}h {eta_minutes}m"
                else:
                    eta_str = "ETA: calculating..."

                print(
                    f"\n   [{i}/{len(videos)}] Processing {video_id} ({eta_str})")

                # Choose B value(s)
                if mixed_B:
                    # Sample one random B value per video
                    B_to_process = [random.choice(self.B_values)]
                else:
                    # Process all B values
                    B_to_process = self.B_values

                for B in B_to_process:
                    # Generate measurements
                    measurements = self.measurement_generator.generate_measurements_from_video(
                        video_path, B, stride=B, video_id=video_id
                    )

                    # Process all labels for this video
                    video_measurements = self.label_converter.process_video_labels(
                        label_file=label_file,
                        B=B,
                        stride=B
                    )

                    # Create lookup dict for labels by frame range
                    labels_lookup = {
                        frame_range: labels for frame_range, labels in video_measurements}

                    # Match measurements with labels
                    for (start_frame, end_frame), img_path in measurements:
                        frame_range = (start_frame, end_frame)

                        # Get labels for this frame range
                        yolo_labels = labels_lookup.get(frame_range, [])

                        # Skip if no labels
                        if not yolo_labels:
                            continue

                        # Copy image to final location
                        final_img_path = self.output_dir / 'images' / split_name / img_path.name
                        shutil.copy(img_path, final_img_path)

                        # Save labels
                        final_label_path = self.output_dir / 'labels' / \
                            split_name / img_path.with_suffix('.txt').name
                        with open(final_label_path, 'w') as f:
                            for label in yolo_labels:
                                f.write(label + '\n')

                # Update processed count and show timing
                processed_count += 1
                video_time = time.time() - video_start_time
                print(f"   ⏱️  Video processed in {video_time:.1f}s")

        # Create data.yaml
        data_yaml_path = self.create_data_yaml()

        # Cleanup temp measurements
        temp_dir = self.output_dir / "temp_measurements"
        if temp_dir.exists():
            shutil.rmtree(temp_dir)

        print(f"\n{'='*70}")
        print("✅ Dataset build complete!")
        print(f"{'='*70}")

        return data_yaml_path

    def process_single_video(self, video_id: str, split: str, B_value=None):
        """
        Process a single video and add to dataset
        
        Args:
            video_id: Video ID (e.g., "01_01_00_00_00")
            split: Either 'train' or 'val'
            B_value: Specific B value to use, or None for random selection
        """
        video_path = self.videos_dir / f"{video_id}.mp4"
        
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")
        
        # Extract label file ID (first two parts)
        label_id = '_'.join(video_id.split('_')[:2])
        label_file = self.labels_dir / f"{label_id}.txt"
        
        if not label_file.exists():
            raise FileNotFoundError(f"Labels not found: {label_file}")
        
        # Create directories if they don't exist
        (self.output_dir / 'images' / split).mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'labels' / split).mkdir(parents=True, exist_ok=True)
        
        # Choose B value
        if B_value is None:
            B_value = random.choice(self.B_values)
        
        # Generate measurements
        measurements = self.measurement_generator.generate_measurements_from_video(
            video_path, B_value, stride=B_value, video_id=video_id
        )
        
        # Process labels
        video_measurements = self.label_converter.process_video_labels(
            label_file=label_file,
            B=B_value,
            stride=B_value
        )
        
        # Create lookup dict for labels by frame range
        labels_lookup = {
            frame_range: labels for frame_range, labels in video_measurements
        }
        
        # Match measurements with labels
        for (start_frame, end_frame), img_path in measurements:
            frame_range = (start_frame, end_frame)
            
            # Get labels for this frame range
            yolo_labels = labels_lookup.get(frame_range, [])
            
            # Skip if no labels
            if not yolo_labels:
                continue
            
            # Copy image to final location
            final_img_path = self.output_dir / 'images' / split / img_path.name
            shutil.copy(img_path, final_img_path)
            
            # Save labels
            final_label_path = self.output_dir / 'labels' / split / img_path.with_suffix('.txt').name
            with open(final_label_path, 'w') as f:
                for label in yolo_labels:
                    f.write(label + '\n')

    def create_data_yaml(self) -> Path:
        """Create YOLO data.yaml config file"""
        # Count files
        train_images = len(
            list((self.output_dir / 'images' / 'train').glob('*.jpg')))
        val_images = len(
            list((self.output_dir / 'images' / 'val').glob('*.jpg')))

        # CURE-TSD class names (14 classes)
        class_names = [
            'Speed Limit 30', 'Speed Limit 60', 'Speed Limit 90',
            'No Overtaking (All)', 'No Overtaking (Trucks)',
            'Right-of-Way at Next Intersection', 'Priority Road',
            'Give Way', 'Stop', 'No Entry',
            'No Entry (Trucks)', 'Roundabout', 'End of No Overtaking (All)',
            'End of No Overtaking (Trucks)'
        ]

        # Create config
        data = {
            'path': str(self.output_dir.absolute()),
            'train': 'images/train',
            'val': 'images/val',
            'nc': len(class_names),
            'names': class_names,
            'dataset_info': {
                'train_images': train_images,
                'val_images': val_images,
                'B_values': self.B_values,
                'gt_strategy': self.gt_strategy,
                'presence_threshold': self.presence_threshold
            }
        }

        # Save
        yaml_path = self.output_dir / 'data.yaml'
        with open(yaml_path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)

        print(f"\n📄 Created data.yaml:")
        print(f"   Train images: {train_images}")
        print(f"   Val images: {val_images}")
        print(f"   Classes: {len(class_names)}")
        print(f"   Saved to: {yaml_path}")

        return yaml_path


if __name__ == "__main__":
    print("🧪 Testing YOLO Dataset Builder\n")

    # Build small test dataset
    builder = YOLODatasetBuilder(
        output_dir="outputs/yolo_dataset_test",
        B_values=[10, 15],  # Only 2 B values for testing
        gt_strategy="union",
        presence_threshold=3,
        train_split=0.8
    )

    # Build with limited videos
    data_yaml = builder.build_dataset(
        max_videos_per_split=2,  # Only 4 total videos (2 train + 2 val)
        mixed_B=True  # Random B per video
    )

    print(f"\n{'='*70}")
    print("✅ Test Complete!")
    print(f"{'='*70}")
    print(f"\nDataset location: {builder.output_dir}")
    print(f"Config file: {data_yaml}")
