"""
CURE-TSD Video Loader
Load and process video frames using OpenCV
"""

import cv2
import numpy as np
from pathlib import Path


class VideoLoader:
    """Load and process video files efficiently"""
    
    def __init__(self, video_path):
        """
        Initialize video loader
        
        Args:
            video_path: Path to video file (e.g., '01_01_00_00_00.mp4')
        """
        self.video_path = Path(video_path)
        
        if not self.video_path.exists():
            raise FileNotFoundError(f"Video not found: {self.video_path}")
        
        self.metadata = self._get_metadata()
        self._frames_cache = None
    
    def _get_metadata(self):
        """Extract video metadata"""
        cap = cv2.VideoCapture(str(self.video_path))
        
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {self.video_path}")
        
        metadata = {
            'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'fps': cap.get(cv2.CAP_PROP_FPS),
            'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            'codec': int(cap.get(cv2.CAP_PROP_FOURCC))
        }
        
        cap.release()
        return metadata
    
    def load_all_frames(self, cache=True):
        """
        Load all frames into memory (use for small videos)
        
        Args:
            cache: Store frames in memory for repeated access
        
        Returns:
            np.array: (num_frames, height, width, 3)
        """
        if cache and self._frames_cache is not None:
            return self._frames_cache
        
        cap = cv2.VideoCapture(str(self.video_path))
        frames = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        
        cap.release()
        
        frames_array = np.array(frames)
        
        if cache:
            self._frames_cache = frames_array
        
        return frames_array
    
    def get_frame(self, frame_idx):
        """
        Get specific frame by index
        
        Args:
            frame_idx: Frame index (0-based)
        
        Returns:
            np.array: Frame (height, width, 3) or None if invalid
        """
        # Check cache first
        if self._frames_cache is not None:
            if 0 <= frame_idx < len(self._frames_cache):
                return self._frames_cache[frame_idx]
            else:
                return None
        
        # Load specific frame
        cap = cv2.VideoCapture(str(self.video_path))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        cap.release()
        
        return frame if ret else None
    
    def frame_generator(self):
        """
        Memory-efficient frame generator for large videos
        
        Yields:
            tuple: (frame_idx, frame)
        """
        cap = cv2.VideoCapture(str(self.video_path))
        frame_idx = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            yield frame_idx, frame
            frame_idx += 1
        
        cap.release()
    
    def __len__(self):
        """Total number of frames"""
        return self.metadata['frame_count']
    
    def __repr__(self):
        return (f"VideoLoader('{self.video_path.name}', "
                f"{self.metadata['frame_count']} frames, "
                f"{self.metadata['width']}x{self.metadata['height']})")


if __name__ == "__main__":
    # Test the video loader
    import sys
    
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
    else:
        video_path = "data/cure-tsd/data/01_01_00_00_00.mp4"
    
    loader = VideoLoader(video_path)
    
    print(f"🎥 Video Loader Test: {video_path}")
    print(f"\nMetadata:")
    for key, value in loader.metadata.items():
        print(f"  {key}: {value}")
    
    print(f"\n📊 Total frames: {len(loader)}")
    
    # Test loading first frame
    frame = loader.get_frame(0)
    if frame is not None:
        print(f"\n✅ First frame loaded successfully!")
        print(f"   Shape: {frame.shape}")
        print(f"   Dtype: {frame.dtype}")
        print(f"   Mean pixel value: {frame.mean():.2f}")
    
    # Test loading all frames (for small video)
    print(f"\n⏳ Loading all frames into memory...")
    all_frames = loader.load_all_frames(cache=True)
    print(f"✅ Loaded {all_frames.shape[0]} frames")
    print(f"   Total size: {all_frames.nbytes / 1024 / 1024:.2f} MB")
