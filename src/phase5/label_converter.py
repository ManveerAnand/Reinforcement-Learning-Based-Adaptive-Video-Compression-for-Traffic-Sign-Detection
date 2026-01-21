"""
Phase 5: YOLOv8 Fine-tuning on Compressed Measurements
Label Converter - Convert CURE-TSD labels to YOLO format with union GT logic
"""

from label_parser import CURETSDLabelParser
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import sys
sys.path.insert(0, 'src/phase1')


class LabelConverter:
    """
    Convert CURE-TSD corner-based labels to YOLO format

    Supports two GT strategies:
    1. Representative: Middle frame + motion padding (default)
    2. Union: Min/max bounding box (paper replication)
    """

    def __init__(self, img_width=1628, img_height=1236,
                 strategy='representative', motion_padding=5,
                 presence_threshold=3):
        """
        Initialize label converter

        Args:
            img_width: Image width (CURE-TSD: 1628)
            img_height: Image height (CURE-TSD: 1236)
            strategy: 'representative' or 'union'
            motion_padding: Pixels to pad for motion blur (representative mode)
            presence_threshold: Include signs appearing in ≥k frames
        """
        self.img_width = img_width
        self.img_height = img_height
        self.strategy = strategy
        self.motion_padding = motion_padding
        self.presence_threshold = presence_threshold

        print(f"📋 Label Converter initialized:")
        print(f"   Strategy: {strategy}")
        print(f"   Image size: {img_width}×{img_height}")
        print(f"   Motion padding: {motion_padding}px")
        print(f"   Presence threshold: ≥{presence_threshold} frames")

    def create_union_ground_truth(self, frame_range: Tuple[int, int],
                                  labels_dict: Dict[int, List[Dict]]) -> List[Dict]:
        """
        Create union ground truth for compressed measurement Y

        Merges bounding boxes from all frames in range by class ID.

        Args:
            frame_range: (start_frame, end_frame) e.g., (0, 9) for B=10
            labels_dict: {frame_num: [list of label dicts]}

        Returns:
            union_labels: [list of merged labels]
        """
        start_frame, end_frame = frame_range
        sign_instances = {}  # {class_id: [list of boxes]}

        # Collect all instances
        for frame_num in range(start_frame, end_frame + 1):
            frame_labels = labels_dict.get(frame_num, [])

            for label in frame_labels:
                # Convert CURE-TSD class_id (1-14) to YOLO format (0-13)
                class_id = label['class_id'] - 1

                if class_id not in sign_instances:
                    sign_instances[class_id] = []

                sign_instances[class_id].append({
                    'x_min': label['x_min'],
                    'x_max': label['x_max'],
                    'y_min': label['y_min'],
                    'y_max': label['y_max'],
                    'frame': frame_num,
                    'is_critical': label.get('is_critical', False)
                })

        # Filter by presence threshold
        union_labels = []

        for class_id, boxes in sign_instances.items():
            # Check if sign appears in enough frames
            if len(boxes) < self.presence_threshold:
                continue  # Skip signs that don't appear often enough

            # Merge boxes
            if self.strategy == 'union':
                # Union bounding box: min/max coordinates
                merged_box = {
                    'class_id': class_id,
                    'x_min': min(b['x_min'] for b in boxes),
                    'x_max': max(b['x_max'] for b in boxes),
                    'y_min': min(b['y_min'] for b in boxes),
                    'y_max': max(b['y_max'] for b in boxes),
                    'is_critical': any(b['is_critical'] for b in boxes),
                    'num_frames': len(boxes)
                }

            elif self.strategy == 'representative':
                # Representative: Middle frame + motion padding
                middle_idx = len(boxes) // 2
                middle_box = boxes[middle_idx]

                merged_box = {
                    'class_id': class_id,
                    'x_min': max(0, middle_box['x_min'] - self.motion_padding),
                    'x_max': min(self.img_width, middle_box['x_max'] + self.motion_padding),
                    'y_min': max(0, middle_box['y_min'] - self.motion_padding),
                    'y_max': min(self.img_height, middle_box['y_max'] + self.motion_padding),
                    'is_critical': middle_box['is_critical'],
                    'num_frames': len(boxes)
                }

            else:
                raise ValueError(f"Unknown strategy: {self.strategy}")

            union_labels.append(merged_box)

        return union_labels

    def corner_to_yolo_format(self, label: Dict) -> Tuple[int, float, float, float, float]:
        """
        Convert corner-based bbox to YOLO format (normalized)

        Args:
            label: Dict with x_min, x_max, y_min, y_max, class_id

        Returns:
            (class_id, x_center, y_center, width, height) all normalized to [0, 1]
        """
        # Calculate center and dimensions
        x_center = (label['x_min'] + label['x_max']) / 2.0
        y_center = (label['y_min'] + label['y_max']) / 2.0
        width = label['x_max'] - label['x_min']
        height = label['y_max'] - label['y_min']

        # Normalize to [0, 1]
        x_center_norm = x_center / self.img_width
        y_center_norm = y_center / self.img_height
        width_norm = width / self.img_width
        height_norm = height / self.img_height

        # Clip to valid range
        x_center_norm = np.clip(x_center_norm, 0, 1)
        y_center_norm = np.clip(y_center_norm, 0, 1)
        width_norm = np.clip(width_norm, 0, 1)
        height_norm = np.clip(height_norm, 0, 1)

        return (label['class_id'], x_center_norm, y_center_norm, width_norm, height_norm)

    def convert_to_yolo_labels(self, frame_range: Tuple[int, int],
                               labels_dict: Dict[int, List[Dict]]) -> List[str]:
        """
        Convert labels for a frame range to YOLO format

        Args:
            frame_range: (start_frame, end_frame)
            labels_dict: {frame_num: [list of label dicts]}

        Returns:
            yolo_labels: List of YOLO format strings
        """
        # Create union ground truth
        union_labels = self.create_union_ground_truth(frame_range, labels_dict)

        # Convert to YOLO format
        yolo_labels = []

        for label in union_labels:
            class_id, x_center, y_center, width, height = self.corner_to_yolo_format(
                label)

            # YOLO format: class_id x_center y_center width height
            yolo_line = f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
            yolo_labels.append(yolo_line)

        return yolo_labels

    def save_yolo_label_file(self, yolo_labels: List[str], output_path: Path):
        """
        Save YOLO labels to text file

        Args:
            yolo_labels: List of YOLO format strings
            output_path: Path to save .txt file
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            for label in yolo_labels:
                f.write(label + '\n')

    def process_video_labels(self, label_file: Path, B: int,
                             stride: Optional[int] = None) -> List[Tuple[Tuple[int, int], List[str]]]:
        """
        Process entire video's labels for compression with ratio B

        Args:
            label_file: Path to CURE-TSD label file (e.g., 01_01.txt)
            B: Compression ratio (frames per measurement)
            stride: Step size between measurements (default=B, non-overlapping)

        Returns:
            List of (frame_range, yolo_labels) tuples
        """
        if stride is None:
            stride = B

        # Parse labels
        parser = CURETSDLabelParser(str(label_file))
        all_labels = parser.annotations  # {frame_num: [labels]}

        # Find frame range
        frame_numbers = sorted(all_labels.keys())
        if not frame_numbers:
            return []

        max_frame = max(frame_numbers)

        # Create measurements
        measurements = []

        for start_frame in range(0, max_frame - B + 2, stride):
            end_frame = start_frame + B - 1

            if end_frame > max_frame:
                break  # Don't go beyond available frames

            frame_range = (start_frame, end_frame)

            # Convert to YOLO labels
            yolo_labels = self.convert_to_yolo_labels(frame_range, all_labels)

            measurements.append((frame_range, yolo_labels))

        return measurements


if __name__ == "__main__":
    print("🧪 Testing Label Converter\n")

    # Test on sample label file
    label_file = Path("data/cure-tsd/labels/01_01.txt")

    if not label_file.exists():
        print(f"❌ Label file not found: {label_file}")
        exit(1)

    # Test both strategies
    for strategy in ['representative', 'union']:
        print(f"\n{'='*70}")
        print(f"Testing Strategy: {strategy.upper()}")
        print(f"{'='*70}")

        converter = LabelConverter(
            img_width=1628,
            img_height=1236,
            strategy=strategy,
            motion_padding=5,
            presence_threshold=3
        )

        # Process with B=10
        B = 10
        print(f"\nProcessing labels for B={B}...")

        measurements = converter.process_video_labels(
            label_file, B=B, stride=B)

        print(f"\n📊 Results:")
        print(f"   Total measurements: {len(measurements)}")

        # Show first few measurements
        for i, (frame_range, yolo_labels) in enumerate(measurements[:3]):
            print(
                f"\n   Measurement {i+1}: Frames {frame_range[0]}-{frame_range[1]}")
            print(f"      Signs detected: {len(yolo_labels)}")

            if yolo_labels:
                print(f"      Sample label: {yolo_labels[0]}")

        # Save sample
        if measurements:
            output_path = Path(
                f"outputs/yolo_labels_test/{strategy}_B{B}_sample.txt")
            frame_range, yolo_labels = measurements[0]
            converter.save_yolo_label_file(yolo_labels, output_path)
            print(f"\n   ✅ Saved sample to: {output_path}")

    print(f"\n{'='*70}")
    print("✅ Label Converter Test Complete!")
    print(f"{'='*70}")
