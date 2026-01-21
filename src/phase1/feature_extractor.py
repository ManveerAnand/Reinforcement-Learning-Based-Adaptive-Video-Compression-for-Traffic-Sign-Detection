"""
Feature Extractor for RL Video Compression
Extract 7D state vector from video frames
"""

import cv2
import numpy as np


class FeatureExtractor:
    """Extract 7D state vector from video frames for RL agent"""

    def __init__(self):
        """Initialize feature extractor"""
        pass

    def extract_state(self, curr_frame, prev_frame, yolo_detections=None, current_B=10):
        """
        Extract complete 7D state vector

        Args:
            curr_frame: Current frame (H, W, 3) BGR
            prev_frame: Previous frame (H, W, 3) BGR
            yolo_detections: List of YOLO detections [{'confidence': 0.9, ...}, ...]
            current_B: Current compression ratio (6-20)

        Returns:
            np.array: [flow, edges, conf, blur, brightness, diff, B_norm] - shape (7,)
        """
        # 1. Optical Flow Magnitude (vehicle speed proxy)
        flow = self.compute_optical_flow(prev_frame, curr_frame)

        # 2. Edge Density (scene complexity)
        edges = self.compute_edge_density(curr_frame)

        # 3. Sign Confidence (YOLOv8 max confidence)
        conf = self.compute_sign_confidence(yolo_detections)

        # 4. Blur Score (motion/focus blur)
        blur = self.compute_blur_score(curr_frame)

        # 5. Brightness (lighting condition)
        brightness = self.compute_brightness(curr_frame)

        # 6. Frame Difference (temporal change)
        diff = self.compute_frame_difference(prev_frame, curr_frame)

        # 7. Current B (normalized compression ratio)
        B_norm = self.normalize_B(current_B)

        state = np.array([flow, edges, conf, blur, brightness, diff, B_norm],
                         dtype=np.float32)

        return state

    def compute_optical_flow(self, prev_frame, curr_frame):
        """
        Calculate optical flow magnitude (vehicle speed proxy)
        FAST VERSION: Use simple frame difference instead of Farneback

        Args:
            prev_frame, curr_frame: BGR frames

        Returns:
            float: Average flow magnitude [0, 50] pixels/frame
        """
        # FAST APPROXIMATION: Use mean absolute difference as proxy for motion
        # This is 100x faster than optical flow but still captures motion
        scale = 0.1  # Even more downsampling for speed
        h, w = prev_frame.shape[:2]
        new_h, new_w = int(h * scale), int(w * scale)

        prev_small = cv2.resize(prev_frame, (new_w, new_h))
        curr_small = cv2.resize(curr_frame, (new_w, new_h))

        prev_gray = cv2.cvtColor(
            prev_small, cv2.COLOR_BGR2GRAY).astype(np.float32)
        curr_gray = cv2.cvtColor(
            curr_small, cv2.COLOR_BGR2GRAY).astype(np.float32)

        # Simple frame difference as motion proxy (MUCH faster than optical flow)
        diff = np.abs(curr_gray - prev_gray)
        avg_motion = np.mean(diff) * 2.0  # Scale to match optical flow range

        # Normalize to [0, 50] range (typical highway speed ~30-40 pixels/frame)
        flow_normalized = float(np.clip(avg_motion, 0, 50))

        return flow_normalized

    def compute_edge_density(self, frame):
        """
        Calculate edge density (scene complexity indicator)
        OPTIMIZED VERSION: Downsampled for speed

        Args:
            frame: BGR frame

        Returns:
            float: Edge density [0, 1]
        """
        # Downsample for faster computation
        scale = 0.25
        h, w = frame.shape[:2]
        new_h, new_w = int(h * scale), int(w * scale)
        frame_small = cv2.resize(frame, (new_w, new_h))

        gray = cv2.cvtColor(frame_small, cv2.COLOR_BGR2GRAY)

        # Canny edge detection
        edges = cv2.Canny(gray, threshold1=50, threshold2=150)

        # Normalize by image size and max intensity
        total_pixels = new_h * new_w
        edge_density = edges.sum() / (total_pixels * 255)

        return float(np.clip(edge_density, 0, 1))

    def compute_sign_confidence(self, detections):
        """
        Get maximum confidence from YOLO detections

        Args:
            detections: List of detection dicts with 'confidence' key

        Returns:
            float: Max confidence [0, 1], or 0.0 if no detections
        """
        if not detections or detections is None:
            return 0.0

        confidences = [d.get('confidence', 0.0) for d in detections]

        if not confidences:
            return 0.0

        return float(max(confidences))

    def compute_blur_score(self, frame):
        """
        Calculate blur using Laplacian variance
        OPTIMIZED VERSION: Downsampled for speed

        Lower variance = more blurred image

        Args:
            frame: BGR frame

        Returns:
            float: Blur score [0, 1000] (higher = sharper)
        """
        # Downsample for faster computation
        scale = 0.25
        h, w = frame.shape[:2]
        new_h, new_w = int(h * scale), int(w * scale)
        frame_small = cv2.resize(frame, (new_w, new_h))

        gray = cv2.cvtColor(frame_small, cv2.COLOR_BGR2GRAY)

        # Laplacian variance (standard blur metric)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        variance = laplacian.var()

        # Normalize to [0, 1000] range
        # Typical sharp images: 500-1000
        # Typical blurred images: 0-100
        blur_score = float(np.clip(variance, 0, 1000))

        return blur_score

    def compute_brightness(self, frame):
        """
        Calculate mean brightness (lighting condition)

        Args:
            frame: BGR frame

        Returns:
            float: Mean pixel intensity [0, 255]
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mean_brightness = gray.mean()

        return float(mean_brightness)

    def compute_frame_difference(self, prev_frame, curr_frame):
        """
        Calculate L2 norm of frame difference (temporal change)
        OPTIMIZED VERSION: Downsampled for speed

        Args:
            prev_frame, curr_frame: BGR frames

        Returns:
            float: Normalized frame difference [0, 1]
        """
        # Downsample for faster computation (4x smaller = 16x faster)
        scale = 0.25
        h, w = prev_frame.shape[:2]
        new_h, new_w = int(h * scale), int(w * scale)

        prev_small = cv2.resize(prev_frame, (new_w, new_h))
        curr_small = cv2.resize(curr_frame, (new_w, new_h))

        # Absolute difference between frames
        diff = cv2.absdiff(prev_small, curr_small)

        # Mean absolute difference (faster than L2 norm)
        diff_norm = np.mean(diff) / 255.0

        # Clip to [0, 1]
        diff_normalized = float(np.clip(diff_norm, 0, 1))

        return diff_normalized

    def normalize_B(self, B, B_min=6, B_max=20):
        """
        Normalize compression ratio to [0, 1]

        Args:
            B: Current compression ratio [6, 20]

        Returns:
            float: Normalized B [0, 1]
        """
        B_normalized = (B - B_min) / (B_max - B_min)
        return float(np.clip(B_normalized, 0, 1))


if __name__ == "__main__":
    # Test feature extraction
    import sys
    from video_loader import VideoLoader

    if len(sys.argv) > 1:
        video_path = sys.argv[1]
    else:
        video_path = "data/cure-tsd/data/01_01_00_00_00.mp4"

    print(f"🔬 Feature Extractor Test: {video_path}\n")

    # Load video
    loader = VideoLoader(video_path)
    frame0 = loader.get_frame(0)
    frame1 = loader.get_frame(1)
    frame10 = loader.get_frame(10)

    # Initialize extractor
    extractor = FeatureExtractor()

    # Test with dummy detections
    print("Test 1: With detections (high confidence)")
    detections = [
        {'confidence': 0.92, 'class_id': 6},
        {'confidence': 0.78, 'class_id': 11}
    ]
    state1 = extractor.extract_state(frame1, frame0, detections, current_B=10)
    print(f"  State: {state1}")
    print(f"  Shape: {state1.shape}, Dtype: {state1.dtype}")

    print("\nTest 2: No detections")
    state2 = extractor.extract_state(frame10, frame1, None, current_B=15)
    print(f"  State: {state2}")

    print("\nTest 3: Individual features (frames 0 → 1)")
    print(
        f"  1. Optical Flow: {extractor.compute_optical_flow(frame0, frame1):.2f} px/frame")
    print(f"  2. Edge Density: {extractor.compute_edge_density(frame1):.4f}")
    print(
        f"  3. Sign Confidence: {extractor.compute_sign_confidence(detections):.4f}")
    print(f"  4. Blur Score: {extractor.compute_blur_score(frame1):.2f}")
    print(f"  5. Brightness: {extractor.compute_brightness(frame1):.2f}")
    print(
        f"  6. Frame Difference: {extractor.compute_frame_difference(frame0, frame1):.4f}")
    print(f"  7. B Normalized (B=10): {extractor.normalize_B(10):.4f}")

    print("\n✅ Feature extractor working!")
