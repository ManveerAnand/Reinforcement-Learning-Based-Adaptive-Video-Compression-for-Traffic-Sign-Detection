"""
OpenAI Gym Environment for RL-based Video Compression
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from .video_loader import VideoLoader
from .label_parser import CURETSDLabelParser
from .feature_extractor import FeatureExtractor


class VideoCompressionEnv(gym.Env):
    """
    Custom Gym Environment for traffic sign detection with adaptive video compression

    State: 7D continuous vector
        [optical_flow, edge_density, sign_confidence, blur_score, 
         brightness, frame_difference, current_B_normalized]

    Action: Discrete (3 actions)
        0 = decrease_B (B = max(6, B-2))
        1 = keep_B (B unchanged)
        2 = increase_B (B = min(20, B+2))

    Reward: 0.7 × mAP + 0.3 × (B/20) - 2.0 × critical_sign_misses
    """

    metadata = {'render.modes': ['human']}

    def __init__(self, video_path, label_path, yolo_model=None,
                 B_min=6, B_max=20, B_step=2):
        """
        Initialize the environment

        Args:
            video_path: Path to video file
            label_path: Path to label file
            yolo_model: YOLOv8 model for detection (optional for now)
            B_min: Minimum compression ratio
            B_max: Maximum compression ratio
            B_step: Step size for B adjustment
        """
        super(VideoCompressionEnv, self).__init__()

        # Load video and labels
        self.video_loader = VideoLoader(video_path)
        self.label_parser = CURETSDLabelParser(label_path)
        self.yolo_model = yolo_model

        # Compression ratio constraints
        self.B_min = B_min
        self.B_max = B_max
        self.B_step = B_step

        # Feature extractor
        self.feature_extractor = FeatureExtractor()

        # Define action and observation spaces
        # Action space: {0: decrease_B, 1: keep_B, 2: increase_B}
        self.action_space = spaces.Discrete(3)

        # Observation space: 7D continuous [0, 1] (normalized)
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(7,),
            dtype=np.float32
        )

        # Episode state
        self.current_frame = 0
        self.current_B = 10  # Start with medium compression
        self.prev_frame_data = None
        self.total_reward = 0
        self.episode_detections = []

        # Cache frames for efficiency
        print(f"Loading video frames into memory...")
        self.all_frames = self.video_loader.load_all_frames(cache=True)
        print(f"✅ Loaded {len(self.all_frames)} frames")

    def reset(self, seed=None, options=None):
        """
        Reset the environment to initial state

        Returns:
            observation: Initial 7D state
            info: Additional info dict
        """
        super().reset(seed=seed)

        self.current_frame = 0
        self.current_B = 10  # Reset to medium compression
        self.total_reward = 0
        self.episode_detections = []

        # Get first frame
        frame = self.all_frames[0]
        self.prev_frame_data = frame.copy()

        # Initial state (no previous frame, so use same frame)
        # For first frame, optical flow will be 0 (same frame comparison)
        state = self._get_state(frame, frame, [], self.current_B)

        info = {
            'frame': self.current_frame,
            'B': self.current_B,
            'total_frames': len(self.all_frames)
        }

        return state, info

    def step(self, action):
        """
        Execute one step in the environment

        Args:
            action: 0 (decrease_B), 1 (keep_B), or 2 (increase_B)

        Returns:
            observation: Next state (7D)
            reward: Reward for this step
            terminated: Whether episode is done
            truncated: Whether episode was truncated
            info: Additional info dict
        """
        # Update compression ratio based on action
        if action == 0:  # Decrease B (less compression, more bandwidth)
            self.current_B = max(self.B_min, self.current_B - self.B_step)
        elif action == 1:  # Keep B (no change)
            pass
        elif action == 2:  # Increase B (more compression, less bandwidth)
            self.current_B = min(self.B_max, self.current_B + self.B_step)
        else:
            raise ValueError(f"Invalid action: {action}")

        # Move to next frame
        self.current_frame += 1

        # Check if episode is done
        terminated = self.current_frame >= len(self.all_frames)

        if terminated:
            # Return final state
            state = self._get_state(
                self.prev_frame_data,
                self.prev_frame_data,
                [],
                self.current_B
            )
            reward = 0.0
            info = {
                'frame': self.current_frame,
                'B': self.current_B,
                'total_reward': self.total_reward,
                'episode_end': True
            }
            return state, reward, terminated, False, info

        # Get current frame
        curr_frame = self.all_frames[self.current_frame]

        # Run detection (placeholder - will use YOLOv8 later)
        detections = self._run_detection(curr_frame)

        # Get state
        state = self._get_state(curr_frame, self.prev_frame_data,
                                detections, self.current_B)

        # Calculate reward
        reward = self._calculate_reward(detections)
        self.total_reward += reward

        # Update previous frame
        self.prev_frame_data = curr_frame.copy()

        # Info
        info = {
            'frame': self.current_frame,
            'B': self.current_B,
            'detections': len(detections),
            'reward_components': self._get_reward_components(detections)
        }

        return state, reward, terminated, False, info

    def _get_state(self, curr_frame, prev_frame, detections, B):
        """
        Extract 7D state vector

        Returns:
            np.array: [flow, edges, conf, blur, brightness, diff, B_norm]
        """
        state = self.feature_extractor.extract_state(
            curr_frame, prev_frame, detections, B
        )

        # Normalize state to [0, 1] for neural network
        state_normalized = self._normalize_state(state)

        return state_normalized

    def _normalize_state(self, state):
        """
        Normalize state vector to [0, 1] range

        State components:
        0. Optical flow: [0, 50] → [0, 1]
        1. Edge density: [0, 1] → [0, 1] (already normalized)
        2. Sign confidence: [0, 1] → [0, 1] (already normalized)
        3. Blur score: [0, 1000] → [0, 1]
        4. Brightness: [0, 255] → [0, 1]
        5. Frame difference: [0, 1] → [0, 1] (already normalized)
        6. B normalized: [0, 1] → [0, 1] (already normalized)
        """
        state_norm = state.copy()
        state_norm[0] = state[0] / 50.0      # Optical flow
        state_norm[3] = state[3] / 1000.0    # Blur score
        state_norm[4] = state[4] / 255.0     # Brightness

        return np.clip(state_norm, 0, 1).astype(np.float32)

    def _run_detection(self, frame):
        """
        Run YOLOv8 detection on frame (placeholder for now)

        Returns:
            list: Detection dicts with 'confidence', 'class_id', 'bbox'
        """
        if self.yolo_model is None:
            # Use ground truth labels as placeholder
            gt_labels = self.label_parser.get_frame_labels(self.current_frame)

            # Convert to detection format with perfect confidence
            detections = [
                {
                    'confidence': 0.95,  # Placeholder confidence
                    'class_id': label['class_id'],
                    'bbox': [label['x_min'], label['y_min'],
                             label['x_max'], label['y_max']]
                }
                for label in gt_labels
            ]

            return detections
        else:
            # TODO: Run actual YOLOv8 inference
            results = self.yolo_model.predict(frame, verbose=False)
            # Parse results...
            return []

    def _calculate_reward(self, detections):
        """
        Dynamic Scene-Adaptive Reward Function

        Adapts reward weights based on scene complexity to encourage:
        - Low B (high quality) for complex/high-motion scenes
        - High B (high compression) for simple/low-motion scenes

        This forces the agent to learn feature-dependent policies.
        """
        # Get ground truth for current frame
        gt_labels = self.label_parser.get_frame_labels(self.current_frame)

        # Calculate detection score (binary match: 1.0 if counts match, 0.8 otherwise)
        detection_score = 1.0 if len(detections) == len(gt_labels) else 0.8

        # Bandwidth factor (higher B = more compression)
        bandwidth_factor = self.current_B / self.B_max

        # Critical sign misses penalty
        critical_gt = [l for l in gt_labels if l['is_critical']]
        critical_detected = [
            d for d in detections if d.get('is_critical', False)]
        critical_misses = max(0, len(critical_gt) - len(critical_detected))

        # === SCENE-ADAPTIVE REWARD WEIGHTING ===
        # Extract scene complexity from current state features
        curr_frame = self.all_frames[self.current_frame]
        prev_frame = self.prev_frame_data if self.prev_frame_data is not None else curr_frame

        # Get scene features
        features = self.feature_extractor.extract_state(
            curr_frame, prev_frame, detections, self.current_B
        )

        # Scene complexity indicators:
        optical_flow = features[0]      # Motion intensity [0-50+]
        edge_density = features[1]      # Scene complexity [0-1]
        blur_score = features[3]        # Image quality [0-1000+]

        # Normalize to [0, 1]
        motion_intensity = min(1.0, optical_flow / 50.0)
        scene_complexity = edge_density
        image_quality = min(1.0, blur_score / 500.0)

        # Composite complexity score [0, 1]
        # High complexity = needs low B (less compression)
        complexity_score = (
            0.5 * motion_intensity +      # Motion is primary indicator
            0.3 * scene_complexity +      # Edge density matters
            0.2 * (1 - image_quality)     # Blur = needs better quality
        )
        complexity_score = np.clip(complexity_score, 0, 1)

        # Dynamic weight adjustment:
        # Complex scenes (high motion/edges): Prefer accuracy (detection_score weight high)
        # Simple scenes (low motion/edges): Prefer bandwidth (B weight high)
        detection_weight = 0.4 + 0.4 * complexity_score      # Range: [0.4, 0.8]
        bandwidth_weight = 0.6 - 0.4 * complexity_score  # Range: [0.2, 0.6]

        # Additional penalty for using extreme B values inappropriately
        B_penalty = 0.0
        if complexity_score > 0.6 and self.current_B > 12:
            # Complex scene but high B → penalize
            B_penalty = 0.2 * (self.current_B - 12) / 8.0
        elif complexity_score < 0.3 and self.current_B < 10:
            # Simple scene but low B → penalize (wasting bandwidth)
            B_penalty = 0.2 * (10 - self.current_B) / 4.0

        # Final reward with dynamic weighting
        reward = (
            detection_weight * detection_score +
            bandwidth_weight * bandwidth_factor -
            2.0 * critical_misses -
            B_penalty
        )

        return float(reward)

    def _get_reward_components(self, detections):
        """Get individual reward components for logging"""
        gt_labels = self.label_parser.get_frame_labels(self.current_frame)
        detection_score = 1.0 if len(detections) == len(gt_labels) else 0.8

        critical_gt = [l for l in gt_labels if l['is_critical']]
        critical_detected = [
            d for d in detections if d.get('is_critical', False)]
        critical_misses = max(0, len(critical_gt) - len(critical_detected))

        # Get scene complexity
        curr_frame = self.all_frames[self.current_frame]
        prev_frame = self.prev_frame_data if self.prev_frame_data is not None else curr_frame
        features = self.feature_extractor.extract_state(
            curr_frame, prev_frame, detections, self.current_B
        )

        optical_flow = features[0]
        edge_density = features[1]
        blur_score = features[3]

        motion_intensity = min(1.0, optical_flow / 50.0)
        complexity_score = (
            0.5 * motion_intensity +
            0.3 * edge_density +
            0.2 * (1 - min(1.0, blur_score / 500.0))
        )

        detection_weight = 0.4 + 0.4 * complexity_score
        bandwidth_weight = 0.6 - 0.4 * complexity_score

        return {
            'detection_score': detection_score,
            'bandwidth_factor': self.current_B / self.B_max,
            'critical_misses': critical_misses,
            'complexity_score': complexity_score,
            'detection_weight': detection_weight,
            'bandwidth_weight': bandwidth_weight,
            'optical_flow': optical_flow,
            'edge_density': edge_density
        }

    def render(self, mode='human'):
        """Render the environment (optional)"""
        pass

    def close(self):
        """Clean up resources"""
        pass


if __name__ == "__main__":
    # Test the Gym environment
    import sys

    video_path = "data/cure-tsd/data/01_01_00_00_00.mp4"
    label_path = "data/cure-tsd/labels/01_01.txt"

    print(f"🏋️ Testing VideoCompressionEnv\n")
    print(f"Video: {video_path}")
    print(f"Labels: {label_path}\n")

    # Create environment
    env = VideoCompressionEnv(video_path, label_path)

    print(f"\n📊 Environment Info:")
    print(f"  Action space: {env.action_space}")
    print(f"  Observation space: {env.observation_space}")
    print(f"  Total frames: {len(env.all_frames)}")

    # Test reset
    print(f"\n🔄 Testing reset()...")
    obs, info = env.reset()
    print(f"  Initial observation: {obs}")
    print(f"  Info: {info}")

    # Test a few steps
    print(f"\n🚶 Testing step()...")
    actions = [2, 2, 1, 0, 1]  # increase, increase, keep, decrease, keep

    for i, action in enumerate(actions, 1):
        obs, reward, terminated, truncated, info = env.step(action)
        action_names = ['decrease_B', 'keep_B', 'increase_B']
        print(f"  Step {i}: action={action_names[action]}, "
              f"B={info['B']}, reward={reward:.4f}, "
              f"detections={info['detections']}")

        if terminated:
            break

    print(f"\n✅ Gym environment working!")
    print(f"   Total reward after {i} steps: {env.total_reward:.4f}")
