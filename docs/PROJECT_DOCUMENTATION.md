# Reinforcement Learning-Based Adaptive Video Compression
## Technical Documentation

**Project**: RL-Based Adaptive Compression for Traffic Sign Detection  
**Course**: CS307 - Advanced Topics in Computer Vision  
**Date**: November 2025 - January 2026

---

## Table of Contents
1. [Problem Statement](#problem-statement)
2. [Motivation and Background](#motivation-and-background)
3. [Proposed Methodology](#proposed-methodology)
4. [Dataset: CURE-TSD](#dataset-cure-tsd)
5. [Implementation Architecture](#implementation-architecture)
6. [Experimental Setup](#experimental-setup)
7. [Evaluation Metrics](#evaluation-metrics)
8. [Results](#results)
9. [References](#references)

---

## 1. Problem Statement

**Objective**: Design an adaptive video compression system that dynamically adjusts compression ratios based on scene characteristics to maintain traffic sign detection accuracy while minimizing bandwidth consumption on edge devices.

**Key Challenges**:
- Traditional fixed compression ratios cannot adapt to varying scene complexity
- Safety-critical traffic signs (Stop, Yield, No Entry) require different treatment than informational signs
- Edge devices have limited computational resources and bandwidth
- Need to balance compression efficiency with detection accuracy

**Novel Contributions**:
- Safety-aware reward function prioritizing critical traffic sign detection
- DQN-based adaptive compression policy learning frame-level adjustments
- Edge case failure prevention while maintaining average performance
- Comprehensive evaluation on challenging weather conditions

---

## 2. Motivation and Background

### 2.1 Real-World Application

Autonomous vehicles require real-time video transmission to cloud infrastructure or edge servers for processing, but face severe bandwidth constraints:

- **Edge Device Constraints**: Limited bandwidth (LTE/5G), computational resources
- **Safety Requirements**: Missing critical signs can cause accidents
- **Environmental Challenges**: Weather conditions (rain, snow, haze) affect detection

### 2.2 Prior Work Limitations

- **Fixed Compression (JPEG, HEVC)**: Static quality parameters cannot adapt to scene changes
- **Learned Compression (Ballé et al.)**: Heavy neural codecs unsuitable for edge devices (>1s latency)
- **CACTI (Paper 3608479.pdf)**: Compressive sensing for video, but no adaptive control
- **Existing RL for Compression**: Ignore task-specific objectives (sign detection)

### 2.3 Our Approach
Combine:
1. **Reinforcement Learning** (DQN) for adaptive decision-making
2. **Optical Flow** for speed estimation (no GPS required)
3. **YOLOv8** for traffic sign detection (task-aware objective)
4. **Multi-factor State Space** (speed, complexity, lighting, blur)

---

## 3. Proposed Methodology

### 3.1 System Architecture

```
Video Frame(t) 
    ↓
Feature Extraction (7D State)
    ↓
RL Agent (DQN) → Action: {decrease_B, keep_B, increase_B}
    ↓
Compression Simulator (B-frame sampling)
    ↓
YOLOv8 Detection → mAP, Critical Misses
    ↓
Reward Calculation
    ↓
Update RL Policy
```

### 3.2 State Space (7 Dimensions)

| Feature | Description | Extraction Method | Range |
|---------|-------------|-------------------|-------|
| **Optical Flow Magnitude** | Vehicle speed proxy | Farneback dense flow | [0, 50] pixels/frame |
| **Edge Density** | Scene complexity | Canny edge detection | [0, 1] |
| **Sign Confidence** | Detection certainty | YOLOv8 max confidence | [0, 1] |
| **Blur Score** | Motion/focus blur | Laplacian variance | [0, 1000] |
| **Brightness** | Lighting condition | Mean pixel intensity | [0, 255] |
| **Frame Difference** | Temporal change | L2 norm prev-curr | [0, 1] |
| **Current B** | Compression ratio | Normalized | [0, 1] |

**Rationale**:
- **Speed (flow)**: High speed → tolerate more compression
- **Complexity (edges)**: Complex scenes → need less compression
- **Confidence**: Low confidence → reduce compression to improve detection
- **Blur**: Blurry frames → already degraded, can compress more
- **Brightness**: Dark frames → need less compression (signs harder to see)
- **Temporal**: High change → reduce compression (important event)

### 3.3 Action Space

**Discrete Actions**: 3 choices per frame
1. `decrease_B`: B = B - 2 (less compression, better quality)
2. `keep_B`: B = B (maintain current)
3. `increase_B`: B = B + 2 (more compression, save bandwidth)

**Constraints**: B ∈ [6, 20]
- B=6: Minimal compression (urban, critical signs)
- B=20: Maximum compression (highway, clean conditions)

### 3.4 Reward Function

```python
reward = α * mAP + β * (B/B_max) - γ * critical_sign_misses

where:
  α = 0.7  # Weight for detection accuracy
  β = 0.3  # Weight for compression efficiency
  γ = 2.0  # Penalty for missing STOP/Yield signs
  
  mAP = Mean Average Precision across all sign classes
  B/B_max = Normalized compression ratio (higher B → more savings)
  critical_sign_misses = Count of missed STOP, Yield, Speed Limit signs
```

**Design Rationale**:
- **Accuracy-first**: 70% weight on mAP (safety critical)
- **Efficiency bonus**: 30% weight on compression (bandwidth savings)
- **Critical penalty**: Heavy penalty for missing important signs

### 3.5 Compression Simulation

Since we don't have CACTI hardware, we simulate compression via:

**Method 1: B-Frame Sampling** (Primary)
- B=5: Keep every 5th frame (80% compression)
- B=10: Keep every 10th frame (90% compression)
- B=20: Keep every 20th frame (95% compression)

**Method 2: JPEG Quality** (Alternative)
- B=6 → Quality=95 (minimal loss)
- B=20 → Quality=50 (aggressive compression)

**Reconstruction**: For detection, use only sampled frames (mimics CACTI measurement constraint)

---

## 4. Dataset: CURE-TSD

### 4.1 Overview
- **Full Name**: Challenging Unreal and Real Environments for Traffic Sign Detection
- **Source**: IEEE DataPort
- **Size**: 75.87 GB (our subset)
- **Videos**: 1,805 sequences
- **Frames**: ~541,500 total (300 per video @ 30 FPS)
- **Resolution**: 640×480
- **Classes**: 14 traffic sign types

### 4.2 Our Subset Selection

| Subset | Sequences | Videos | Size | Purpose |
|--------|-----------|--------|------|---------|
| Real-world (01_*) | 25 | 1,525 | ~64 GB | Training |
| Unreal (02_*) | 5 | 280 | ~12 GB | Validation |
| **Total** | **30** | **1,805** | **~76 GB** | Full dataset |

### 4.3 Challenge Types (CURE-TSD)

| Code | Challenge | Description | Relevance |
|------|-----------|-------------|-----------|
| 00 | Clean | Baseline (no degradation) | Reference performance |
| 01 | Decolorization | Grayscale conversion | Low relevance |
| 02 | Lens Blur | Out-of-focus | Tests blur handling |
| 03 | Codec Error | Compression artifacts | Directly relevant |
| 04 | Darkening | Low light | Tests lighting robustness |
| 05 | Dirty Lens | Sensor occlusion | Low relevance |
| 06 | Exposure | Overexposure | Tests lighting |
| 07 | Gaussian Blur | Motion blur | **High relevance** (speed proxy) |
| 08 | Noise | Sensor noise | Low relevance |
| 09 | Rain | Weather | Tests complexity |
| 10 | Shadow | Lighting variation | Tests brightness |
| 11 | Snow | Weather | Tests complexity |
| 12 | Haze | Visibility reduction | Tests complexity |

**Key Challenges for Our Work**: 
- **07 (Gaussian Blur)**: Simulates high-speed motion
- **04/06 (Lighting)**: Tests brightness feature
- **09/11 (Weather)**: Tests scene complexity

### 4.4 Annotation Format

Label files (`01_01.txt` format):
```
frame_number x_center y_center width height class_id
```

Example:
```
0 0.487 0.345 0.123 0.089 6    # Frame 0: STOP sign (class 6)
1 0.489 0.347 0.125 0.091 6    # Frame 1: Same sign (slight movement)
...
```

**Class Mapping** (most important):
- Class 1: Speed Limit 30
- Class 6: STOP sign (critical)
- Class 13: Yield sign (critical)

### 4.5 Directory Structure

```
data/cure-tsd/
├── data/
│   ├── 01_01_00_00_00.mp4   # Sequence 01, Sign 01, Clean
│   ├── 01_01_01_07_03.mp4   # Sequence 01, Sign 01, Blur Level 3
│   ├── ...
│   └── 02_05_01_11_05.mp4   # Sequence 02, Sign 05, Snow Level 5
└── labels/
    ├── 01_01.txt             # Annotations for all 01_01_* videos
    ├── ...
    └── 02_05.txt             # Annotations for all 02_05_* videos
```

**File Naming**: `AA_BB_CC_DD_EE.mp4`
- AA: Sequence type (01=Real, 02=Unreal)
- BB: Sequence number (01-49)
- CC: Source type (00=original, 01=synthetic challenge)
- DD: Challenge type (00=clean, 01-12=degradation)
- EE: Challenge level (00=none, 01-05=severity)

---

## 5. Implementation Architecture

### 5.1 Module Breakdown

```
RL_Video_Compression/
├── src/
│   ├── data_loader.py          # Parse CURE-TSD labels, load videos
│   ├── feature_extractor.py    # Extract 7D state from frames
│   ├── compression_simulator.py # B-frame sampling simulation
│   ├── yolo_detector.py        # YOLOv8 fine-tuning & inference
│   ├── rl_agent.py             # DQN network & training
│   ├── reward_calculator.py    # Compute mAP + reward
│   └── train.py                # Main training loop
├── data/cure-tsd/              # Dataset (75 GB)
├── results/                    # Checkpoints, logs, plots
├── docs/                       # Documentation
└── requirements.txt
```

### 5.2 Key Algorithms

#### 5.2.1 Optical Flow (Speed Estimation)
```python
import cv2
import numpy as np

def estimate_speed(prev_frame, curr_frame):
    # Convert to grayscale
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
    
    # Dense optical flow
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, curr_gray, None,
        pyr_scale=0.5, levels=3, winsize=15,
        iterations=3, poly_n=5, poly_sigma=1.2, flags=0
    )
    
    # Magnitude (pixels/frame)
    magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
    avg_speed = np.mean(magnitude)
    
    return avg_speed  # Range: [0, 50] typically
```

#### 5.2.2 Scene Complexity (Edge Density)
```python
def calculate_complexity(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, threshold1=50, threshold2=150)
    edge_density = edges.sum() / (frame.shape[0] * frame.shape[1])
    return edge_density  # Range: [0, 1]
```

#### 5.2.3 DQN Network
```python
import torch
import torch.nn as nn

class DQN(nn.Module):
    def __init__(self, state_dim=7, action_dim=3):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )
    
    def forward(self, state):
        return self.network(state)
```

---

## 6. Experimental Setup

### 6.1 Training Configuration

| Parameter | Value | Justification |
|-----------|-------|---------------|
| **RL Algorithm** | DQN | Stable, proven for discrete actions |
| **Replay Buffer** | 10,000 | Balance memory & diversity |
| **Batch Size** | 64 | Standard for DQN |
| **Learning Rate** | 1e-4 | Conservative (avoid instability) |
| **Discount Factor (γ)** | 0.99 | Long-term reward focus |
| **Epsilon (ε-greedy)** | 0.1 → 0.01 | Exploration decay |
| **Target Update** | Every 100 steps | Stabilize Q-learning |
| **Episodes** | 1,525 (all training videos) | Full dataset coverage |

### 6.2 Train/Validation Split

- **Training**: 25 sequences (01_01 to 01_25) = 1,525 videos
- **Validation**: 5 sequences (02_01 to 02_05) = 280 videos

**Rationale**: 
- Train on real-world data (01_*)
- Test on Unreal data (02_*) to check generalization
- Unreal sequences have different lighting/rendering (good diversity test)

### 6.3 Baseline Comparisons

| Baseline | Description | B Value |
|----------|-------------|---------|
| **Fixed-Low** | Always low compression | B = 6 |
| **Fixed-High** | Always high compression | B = 20 |
| **Random** | Random B ∈ [6, 20] | Uniform |
| **Our RL Agent** | Adaptive based on 7D state | Dynamic [6, 20] |

### 6.4 Ablation Studies

Test contribution of each feature by removing one at a time:

1. **No Speed**: Remove optical flow from state (6D)
2. **No Complexity**: Remove edge density
3. **No Blur**: Remove Laplacian variance
4. **No Lighting**: Remove brightness
5. **Speed Only**: Use only optical flow (1D state)
6. **Full Model**: All 7 features (baseline)

**Hypothesis**: Speed feature should show significant improvement (professor's suggestion).

---

## 7. Evaluation Metrics

### 7.1 Detection Accuracy
- **mAP (Mean Average Precision)**: Standard object detection metric
- **Critical Sign Accuracy**: Precision/Recall for STOP, Yield, Speed Limit
- **Per-Class AP**: Breakdown across 14 sign types

### 7.2 Compression Efficiency
- **Average B**: Mean compression ratio over all frames
- **Bandwidth Savings**: `(1 - 1/B_avg) × 100%`
- **Example**: B=10 → 90% bandwidth savings

### 7.3 Trade-off Metrics
- **mAP vs B**: Plot detection accuracy against compression
- **Reward Curve**: Training progress
- **Critical Miss Rate**: Percentage of STOP/Yield signs missed

### 7.4 Success Criteria

| Metric | Target | Justification |
|--------|--------|---------------|
| **mAP** | ≥ 0.75 | Acceptable detection performance |
| **Bandwidth Savings** | ≥ 85% | Significant compression (B ≥ 10) |
| **Critical Miss Rate** | ≤ 5% | Safety requirement |
| **Speed Improvement** | +10% mAP vs No-Speed | Validate professor's hypothesis |

---

## 8. Expected Results

### 8.1 Hypotheses

**H1: Speed-aware compression improves detection accuracy**
- Urban (low speed) → RL assigns low B (B=6-8)
- Highway (high speed) → RL assigns high B (B=15-20)
- **Result**: Higher mAP than fixed baselines

**H2: Multi-factor state outperforms single-factor**
- Full 7D state > Speed-only state
- **Result**: Ablation study shows each feature contributes

**H3: Task-aware compression outperforms generic compression**
- Our RL agent > Standard JPEG/H.264
- **Result**: Better mAP at same bandwidth

### 8.2 Expected Performance

| Method | Avg B | mAP | Bandwidth Savings | Critical Miss Rate |
|--------|-------|-----|-------------------|-------------------|
| Fixed-Low (B=6) | 6 | 0.85 | 83% | 2% |
| Fixed-High (B=20) | 20 | 0.60 | 95% | 15% |
| Random | 13 | 0.70 | 92% | 10% |
| **Our RL Agent** | **12** | **0.78** | **92%** | **4%** |

**Key Result**: RL achieves 92% compression with 78% mAP (vs 60% mAP for fixed B=20).

### 8.3 Visualization Plans

1. **mAP vs B Scatter Plot**: Show RL dynamically adjusts
2. **Reward Curve**: Training convergence over episodes
3. **Speed Distribution**: Histogram of optical flow magnitudes
4. **Confusion Matrix**: Sign detection accuracy per class
5. **Qualitative Examples**: Side-by-side frames (original vs compressed)

---

## 9. References

### 10.1 Key Papers
1. **CACTI**: "Video from a Single Coded Exposure Photograph" (3608479.pdf)
2. **CURE-TSD**: Temel et al., "CURE-TSD: Challenging Unreal and Real Environments Traffic Sign Detection Dataset", IEEE DataPort
3. **DQN**: Mnih et al., "Human-level control through deep reinforcement learning", Nature 2015
4. **YOLOv8**: Ultralytics YOLOv8 documentation

### 10.2 Code Dependencies
```
torch==2.0.0          # DQN training
opencv-python==4.8.0  # Optical flow, preprocessing
ultralytics==8.0.0    # YOLOv8
numpy==1.24.0
matplotlib==3.7.0     # Plotting
tqdm==4.65.0          # Progress bars
```

---

## 10. Risk Mitigation

| Risk | Impact | Mitigation |
|------|--------|------------|
| **RL doesn't converge** | High | Use proven DQN, tune hyperparameters carefully |
| **YOLOv8 poor accuracy** | High | Fine-tune on CURE-TSD, use pretrained weights |
| **Time constraint** | High | Focus on core features first, skip ablations if needed |
| **GPU unavailable** | Medium | Use Google Colab / AWS free tier |
| **Dataset too large** | Low | Already extracted subset (75 GB manageable) |

---

## 11. Delta Contribution (Novelty)

**What's New?**
1. ✅ **Speed-aware compression** (optical flow without GPS)
2. ✅ **Multi-factor RL state** (7D combining speed, complexity, blur, lighting)
3. ✅ **Task-aware objective** (detection mAP instead of PSNR/SSIM)
4. ✅ **Real-world dataset** (CURE-TSD with challenging conditions)

**vs Existing Work**:
- CACTI paper: No adaptive control (fixed compression)
- Generic RL compression: Uses PSNR (not task-aware)
- Traffic sign detection: Assumes raw video (no compression)

---

## 12. OpenAI Gym / Gymnasium Integration

### Framework Choice: Gymnasium + Stable-Baselines3

We will use **Gymnasium** (successor to OpenAI Gym) with **Stable-Baselines3** for RL training.

**Rationale**:
1. **Standard Interface**: Industry-standard RL environment API
2. **Proven Algorithms**: Stable-Baselines3 provides battle-tested DQN implementation
3. **Less Code**: No need to write replay buffer, target networks manually
4. **Hyperparameter Tuning**: Easy to experiment with different algorithms (PPO, A2C, SAC)
5. **Academic Recognition**: Expected in research papers

### Custom Environment Structure

```python
import gymnasium as gym
from gymnasium import spaces

class VideoCompressionEnv(gym.Env):
    """
    Custom Gym environment for speed-aware video compression
    
    Observation: 7D Box [optical_flow, edge_density, sign_conf, blur, brightness, diff, B_norm]
    Action: Discrete(3) - {0: decrease_B, 1: keep_B, 2: increase_B}
    Reward: 0.7*mAP + 0.3*(B/20) - 2.0*critical_misses
    """
    
    def __init__(self, video_path, label_path, yolo_detector):
        super().__init__()
        
        # Define observation space (7D state)
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0, 0, 0, 0]),
            high=np.array([50, 1, 1, 1000, 255, 1, 1]),
            dtype=np.float32
        )
        
        # Define action space (3 discrete actions)
        self.action_space = spaces.Discrete(3)
        
        # Load video and annotations
        self.video_frames = load_video(video_path)
        self.labels = parse_labels(label_path)
        self.yolo = yolo_detector
        
    def reset(self, seed=None):
        """Reset to initial state"""
        super().reset(seed=seed)
        self.frame_idx = 0
        self.current_B = 10
        return self._get_obs(), {}
    
    def step(self, action):
        """Execute action, return (obs, reward, terminated, truncated, info)"""
        # Update B based on action
        if action == 0:
            self.current_B = max(6, self.current_B - 2)
        elif action == 2:
            self.current_B = min(20, self.current_B + 2)
        
        # Compress and detect
        compressed = compress(self.video_frames[self.frame_idx], self.current_B)
        detections = self.yolo.detect(compressed)
        
        # Calculate reward
        reward = self._calc_reward(detections)
        
        # Update state
        self.frame_idx += 1
        terminated = (self.frame_idx >= len(self.video_frames))
        
        return self._get_obs(), reward, terminated, False, {}
```

### Training with Stable-Baselines3

```python
from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env

# Create and validate environment
env = VideoCompressionEnv(video_path, label_path, yolo_model)
check_env(env)  # Validates Gym API compliance

# Create DQN agent
model = DQN(
    'MlpPolicy',
    env,
    learning_rate=1e-4,
    buffer_size=10000,
    batch_size=64,
    gamma=0.99,
    exploration_fraction=0.3,
    tensorboard_log='./results/logs/'
)

# Train
model.learn(total_timesteps=100000)

# Save and evaluate
model.save('results/dqn_model')
```

### Key Benefits

1. **No Manual DQN**: SB3 handles experience replay, target networks, epsilon decay
2. **TensorBoard**: Built-in logging for rewards, losses, B-values
3. **Algorithm Flexibility**: Can easily switch to PPO, A2C if DQN doesn't work
4. **Environment Validation**: `check_env()` catches API issues early
5. **Reproducibility**: Set random seeds for consistent results

---

## Notes
- GPU: RTX 3060/3070 recommended (YOLOv8 + RL training)
- Storage: 100 GB free (dataset + results)
- Dependencies: `gymnasium`, `stable-baselines3`, `torch`, `ultralytics`

---

**Last Updated**: November 12, 2025  
**Status**: Phase 1 - Dataset Ready ✅, Gym strategy defined ✅
