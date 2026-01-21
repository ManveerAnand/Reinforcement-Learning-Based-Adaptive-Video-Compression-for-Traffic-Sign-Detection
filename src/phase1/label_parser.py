"""
CURE-TSD Label Parser
Parses annotation files in YOLO format
"""

import numpy as np
from pathlib import Path


class CURETSDLabelParser:
    """Parse CURE-TSD annotation files"""
    
    # Class mapping for CURE-TSD dataset
    CLASS_NAMES = {
        1: 'Speed Limit 30',
        2: 'Speed Limit 50',
        3: 'Speed Limit 60',
        4: 'Speed Limit 70',
        5: 'Speed Limit 80',
        6: 'STOP',
        7: 'No Entry',
        8: 'No Parking',
        9: 'Priority Road',
        10: 'Yield',
        11: 'Roundabout',
        12: 'No Overtaking',
        13: 'No Left Turn',
        14: 'No Right Turn'
    }
    
    # Critical signs for safety
    CRITICAL_CLASSES = [1, 6, 10]  # Speed Limit, STOP, Yield
    
    def __init__(self, label_path):
        """
        Initialize label parser
        
        Args:
            label_path: Path to label file (e.g., '01_01.txt')
        """
        self.label_path = Path(label_path)
        self.annotations = self._parse()
    
    def _parse(self):
        """
        Parse label file into structured format
        
        CURE-TSD Label format: frameNumber_signType_llx_lly_lrx_lry_ulx_uly_urx_ury
        Example: 061_05_1288_241_1307_241_1288_262_1307_262
        
        Where:
        - frameNumber: Frame index (1-indexed in file, converted to 0-indexed)
        - signType: Sign class ID (01-14)
        - Bounding box corners: lower-left, lower-right, upper-left, upper-right
        
        Returns:
            dict: {frame_idx: [bbox1, bbox2, ...]}
        """
        annotations = {}
        
        if not self.label_path.exists():
            print(f"Warning: Label file not found: {self.label_path}")
            return annotations
        
        with open(self.label_path, 'r') as f:
            for line_num, line in enumerate(f, start=1):
                line = line.strip()
                
                # Skip header or empty lines
                if not line or line.startswith('frameNumber'):
                    continue
                
                parts = line.split('_')
                if len(parts) != 10:
                    print(f"Warning: Skipping invalid line {line_num}: {line}")
                    continue
                
                try:
                    frame_idx = int(parts[0]) - 1  # Convert to 0-indexed
                    class_id = int(parts[1])
                    
                    # Extract bounding box corners
                    llx, lly = int(parts[2]), int(parts[3])  # Lower-left
                    lrx, lry = int(parts[4]), int(parts[5])  # Lower-right
                    ulx, uly = int(parts[6]), int(parts[7])  # Upper-left
                    urx, ury = int(parts[8]), int(parts[9])  # Upper-right
                    
                    # Convert to x_min, y_min, x_max, y_max
                    x_min = min(llx, lrx, ulx, urx)
                    x_max = max(llx, lrx, ulx, urx)
                    y_min = min(lly, lry, uly, ury)
                    y_max = max(lly, lry, uly, ury)
                    
                    # Calculate center and dimensions
                    width = x_max - x_min
                    height = y_max - y_min
                    x_center = x_min + width / 2
                    y_center = y_min + height / 2
                    
                    bbox = {
                        'x_min': x_min,
                        'y_min': y_min,
                        'x_max': x_max,
                        'y_max': y_max,
                        'x_center': x_center,
                        'y_center': y_center,
                        'width': width,
                        'height': height,
                        'class_id': class_id,
                        'class_name': self.CLASS_NAMES.get(class_id, 'Unknown'),
                        'is_critical': class_id in self.CRITICAL_CLASSES
                    }
                    
                    if frame_idx not in annotations:
                        annotations[frame_idx] = []
                    
                    annotations[frame_idx].append(bbox)
                    
                except (ValueError, IndexError) as e:
                    print(f"Warning: Error parsing line {line_num}: {line} - {e}")
                    continue
        
        return annotations
    
    def get_frame_labels(self, frame_idx):
        """
        Get all bounding boxes for a specific frame
        
        Args:
            frame_idx: Frame index (0-based)
        
        Returns:
            list: List of bbox dicts
        """
        return self.annotations.get(frame_idx, [])
    
    def get_total_frames(self):
        """Get total number of annotated frames"""
        if not self.annotations:
            return 0
        return max(self.annotations.keys()) + 1
    
    def get_critical_signs(self, frame_idx):
        """
        Get STOP, Yield, Speed Limit signs from frame
        
        Args:
            frame_idx: Frame index
        
        Returns:
            list: List of critical sign bboxes
        """
        labels = self.get_frame_labels(frame_idx)
        return [l for l in labels if l['is_critical']]
    
    def get_num_signs(self, frame_idx):
        """Get number of signs in frame"""
        return len(self.get_frame_labels(frame_idx))
    
    def to_yolo_format(self, frame_idx):
        """
        Convert to YOLO detection format
        
        Returns:
            list: [(class_id, x_center, y_center, width, height), ...]
        """
        labels = self.get_frame_labels(frame_idx)
        return [(l['class_id'], l['x_center'], l['y_center'], 
                l['width'], l['height']) for l in labels]
    
    def get_statistics(self):
        """Get dataset statistics"""
        total_frames = self.get_total_frames()
        total_signs = sum(len(boxes) for boxes in self.annotations.values())
        frames_with_signs = len(self.annotations)
        
        # Count by class
        class_counts = {}
        for boxes in self.annotations.values():
            for box in boxes:
                class_id = box['class_id']
                class_counts[class_id] = class_counts.get(class_id, 0) + 1
        
        return {
            'total_frames': total_frames,
            'total_signs': total_signs,
            'frames_with_signs': frames_with_signs,
            'class_distribution': class_counts
        }


if __name__ == "__main__":
    # Test the parser
    import sys
    
    if len(sys.argv) > 1:
        label_path = sys.argv[1]
    else:
        label_path = "data/cure-tsd/labels/01_01.txt"
    
    parser = CURETSDLabelParser(label_path)
    
    print(f"📊 Label Parser Test: {label_path}")
    print(f"Total frames: {parser.get_total_frames()}")
    
    # Show first annotated frame labels
    first_frame = min(parser.annotations.keys()) if parser.annotations else None
    if first_frame is not None:
        print(f"\nFrame {first_frame} labels:")
        for bbox in parser.get_frame_labels(first_frame):
            print(f"  - {bbox['class_name']} (ID={bbox['class_id']}): "
                  f"bbox=[{bbox['x_min']}, {bbox['y_min']}, {bbox['x_max']}, {bbox['y_max']}]")
    
    # Statistics
    stats = parser.get_statistics()
    print(f"\n📈 Statistics:")
    print(f"  Total signs: {stats['total_signs']}")
    print(f"  Frames with signs: {stats['frames_with_signs']}")
    print(f"\n  Class distribution:")
    for class_id, count in sorted(stats['class_distribution'].items()):
        class_name = parser.CLASS_NAMES.get(class_id, 'Unknown')
        print(f"    {class_name} (ID={class_id}): {count}")
