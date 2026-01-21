# Reinforcement Learning-Based Adaptive Video Compression for Autonomous Driving

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5.1-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**Safety-Aware Adaptive Compression for Traffic Sign Detection on Edge Devices**

---

## Abstract

This repository contains the implementation of a reinforcement learning-based adaptive video compression framework designed for bandwidth-constrained edge devices in autonomous vehicles. The system dynamically adjusts compression ratios using Deep Q-Networks (DQN) to optimize the trade-off between bandwidth consumption and traffic sign detection accuracy. By employing Snapshot Compressive Imaging (SCI) and a safety-aware reward function that prioritizes critical traffic signs (Stop, Yield, No Entry), our approach achieves up to 91.6% bandwidth savings while maintaining robust detection performance and preventing catastrophic failures in challenging scenarios.

**Key Contributions:**
- Safety-aware adaptive compression with critical sign prioritization
- DQN-based policy for frame-level compression ratio selection
- Edge case failure prevention (28.6% improvement in challenging conditions)
- Comprehensive evaluation on CURE-TSD dataset with 280 validation videos

---

## Table of Contents

- [Installation](#installation)
- [Dataset Preparation](#dataset-preparation)
- [Training](#training)
- [Evaluation](#evaluation)
- [Results](#results)
- [Project Structure](#project-structure)
- [Citation](#citation)
- [License](#license)
- [Acknowledgments](#acknowledgments)

---

## Installation

### System Requirements

- **GPU**: NVIDIA GPU with CUDA support (8GB+ VRAM recommended)
- **CUDA**: Version 12.1 or higher
- **Python**: 3.12
- **Storage**: Minimum 50GB free space
- **Operating System**: Linux, macOS, or Windows with WSL

### Environment Setup

1. **Clone the repository:**
```bash
git clone https://github.com/ManveerAnand/Adaptive_video_compression.git
cd Adaptive_video_compression
```

2. **Create and activate conda environment:**
```bash
conda create -n rl_video_compression python=3.12
conda activate rl_video_compression
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Verify installation:**
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
```

---

## Dataset Preparation

### CURE-TSD Dataset

This work uses the **CURE-TSD** (Challenging Unreal and Real Environments for Traffic Sign Detection) dataset, which contains 1,805 videos with 14 traffic sign classes under various challenging conditions.

**Dataset Specifications:**
- **Resolution**: 1628 × 1236 pixels
- **Frame Rate**: 10 FPS
- **Classes**: 14 traffic sign types (Stop, Speed limits, Yield, etc.)
- **Conditions**: Rain, Snow, Haze, Decolorization, and other challenges
- **Source**: [Georgia Tech OLIVES Lab](https://github.com/olivesgatech/CURE-TSD)

### Dataset Generation

1. **Download CURE-TSD dataset** and place in `data/cure-tsd/`

2. **Generate SCI compressed measurements:**
```bash
python scripts/generate_full_dataset.py
```

This creates SCI-compressed measurements for B ∈ {6, 8, 10, 12, 15, 20} and converts labels to YOLO format.

**Expected Output:**
```
data/yolo_dataset_full/
├── images/
│   ├── train/     # ~20,000 compressed measurements
│   └── val/       # ~8,700 compressed measurements
├── labels/
│   ├── train/
│   └── val/
└── data.yaml      # Dataset configuration
```

3. **Verify dataset integrity:**
```bash
python scripts/check_dataset_progress.py
```

---
## Training

### 1. Train YOLOv8n Detector

Train the YOLOv8 Nano model on SCI-compressed measurements:

```bash
cd training
python train_yolo_local.py
```

**Training Configuration:**
- **Architecture**: YOLOv8 Nano (3.0M parameters)
- **Batch Size**: 16
- **Epochs**: 100
- **Image Size**: 640×640
- **Optimizer**: AdamW
- **Hardware**: Single NVIDIA GPU (8GB VRAM)

**Checkpointing**: Training automatically resumes from `runs/train/yolo_cure_tsd/weights/last.pt` if interrupted.

**Expected Training Time**: ~16 hours on RTX 4060 Laptop

### 2. Train RL Agent

Train the DQN agent for adaptive compression ratio selection:

```bash
cd training
python train_rl_agent_adaptive.py
```

**RL Configuration:**
- **Algorithm**: Deep Q-Network (DQN)
- **State Space**: 7-dimensional (motion, edge density, blur, brightness, previous B, detections, misses)
- **Action Space**: 3 discrete actions (decrease B, keep B, increase B)
- **Network Architecture**: 7 → 128 → 128 → 3
- **Replay Buffer**: 10,000 transitions
- **Episodes**: 500
- **Exploration**: ε-greedy (1.0 → 0.01, decay 0.995)

**Reward Function:**
```
R = 0.7 × Detection_Score + 0.3 × B_norm - 2.0 × Critical_Misses - 0.1
```

where `Critical_Misses` counts missed critical signs (Stop, Yield, No Entry).

**Expected Training Time**: ~3-4 hours on RTX 4060 Laptop

### 3. Validate Trained Models

```bash
cd training
python validate_yolo.py
```

---

## Evaluation

### Benchmark Experiments

We evaluate the system using four experiments on 280 validation videos:

**1. Fixed Baseline Compression:**
```bash
python scripts/evaluate_fixed_baselines.py
```
Evaluates all fixed B-values {6, 8, 10, 12, 15, 20} across 280 videos (1,680 total evaluations).

**2. Random Policy Baseline:**
```bash
python scripts/evaluate_random_policy.py
```
Random B-value selection for comparison (includes checkpoint resume capability).

**3. RL Agent Evaluation:**
```bash
python scripts/evaluate_rl_agent.py --model runs/rl_training_adaptive/best_model_adaptive.pth
```
Evaluates trained DQN agent on validation set.

**4. Statistical Analysis:**
```bash
python scripts/statistical_tests.py
```
Performs statistical significance tests comparing RL vs baselines.

### Output Files

All results are saved in `outputs/`:
- `fixed_baseline_results.csv` - Fixed B-value results (1,680 rows)
- `random_policy_results.csv` - Random policy results (280 rows)
- `rl_agent_results.csv` - RL agent results (280 rows)
- `statistical_tests_results.json` - Statistical test outcomes

---

## Results

### YOLOv8n Detection Performance

Training on SCI-compressed measurements (28,727 images):

| Metric | Value |
|--------|-------|
| mAP50 | 83.29% |
| mAP50-95 | 47.44% |
| Precision | 86.73% |
| Recall | 74.94% |

**Training Details**: 100 epochs, RTX 4060 Laptop, 16-hour duration

### Fixed Baseline Compression

Evaluation across 280 validation videos with fixed B-values:

| B-value | Detections (avg) | Bandwidth Savings |
|---------|------------------|-------------------|
| 6 | 94.996 | 83.33% |
| 8 | 78.436 | 87.33% |
| 10 | 66.218 | 90.00% |
| 12 | 57.979 | 91.67% |
| 15 | 47.957 | 93.33% |
| 20 | 39.504 | 95.00% |

Clear bandwidth-accuracy trade-off: lower B preserves more information but requires higher bandwidth.

### RL Agent Performance

**Average Performance (280 videos):**
- Average B-value: 11.92 (adaptive)
- Average Detections: 57.64
- Bandwidth Savings: 91.56%
- Performance vs Fixed B=12: -0.59% (statistically similar)

**Edge Case Prevention:**
While average performance matches fixed compression, the RL agent prevents catastrophic failures in challenging scenarios:

| Video ID | Condition | RL Detections | Fixed B=12 | Improvement |
|----------|-----------|---------------|------------|-------------|
| 02_01_01_06_05 | Rain + Low Light | 9.0 | 7.0 | +28.6% |
| 02_02_01_02_02 | Rain + Challenge | 67.7 | 55.0 | +23.0% |
| 02_04_01_09_04 | Rain + Artifact | 51.7 | 43.0 | +20.2% |

**Key Finding**: RL achieves similar average performance but excels in 6.8% of videos with >10% improvement, crucial for safety-critical autonomous driving applications.

### Safety-Aware Behavior

The reward function successfully prioritizes critical traffic signs:
- Critical sign classes: Stop, Yield, No Entry
- Penalty weight: 2.0× for critical sign misses
- Result: Agent learns to preserve compression quality when critical signs are present

---

## Project Structure

```
RL_Video_Compression/
├── data/
│   ├── cure-tsd/              # Original CURE-TSD dataset
│   ├── masks/                 # Binary SCI masks (B=6,8,10,12,15,20)
│   └── yolo_dataset_full/     # Generated YOLO dataset
│       ├── images/            # SCI compressed measurements
│       ├── labels/            # YOLO format labels
│       └── data.yaml          # Dataset configuration
│
├── src/
│   ├── phase1/                # Core compression & environment
│   │   ├── video_compression_env.py  # RL environment
│   │   ├── sci_compressor.py         # SCI implementation
│   │   └── feature_extractor.py      # State extraction
│   └── phase5/                # Dataset generation
│       ├── dataset_builder.py
│       ├── label_converter.py
│       └── measurement_generator.py
│
├── training/                  # Training scripts
│   ├── train_yolo_local.py
│   ├── train_rl_agent_adaptive.py
│   └── validate_yolo.py
│
├── scripts/                   # Evaluation & utilities
│   ├── evaluate_fixed_baselines.py
│   ├── evaluate_random_policy.py
│   ├── evaluate_rl_agent.py
│   ├── statistical_tests.py
│   └── generate_full_dataset.py
│
├── outputs/                   # Experimental results
│   ├── fixed_baseline_results.csv
│   ├── random_policy_results.csv
│   └── benchmarks/
│
├── runs/                      # Training outputs
│   ├── rl_training_adaptive/  # RL agent checkpoints
│   └── train/yolo_cure_tsd/   # YOLO training logs
│
└── docs/                      # Documentation
    ├── PROJECT_DOCUMENTATION.md
    ├── BENCHMARKING_RESULTS.md
    └── RL_FOUNDATIONS_AND_PROJECT_GUIDE.md
```

---

## Citation

If you use this code or methodology in your research, please cite:

```bibtex
@misc{anand2026rl_adaptive_compression,
  author = {Anand, Manveer},
  title = {Reinforcement Learning-Based Adaptive Video Compression for Autonomous Driving},
  year = {2026},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/ManveerAnand/Adaptive_video_compression}},
  note = {CS307 - Advanced Topics in Computer Vision}
}
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- **CURE-TSD Dataset**: Temel, S., et al. "CURE-TSD: Challenging unreal and real environment traffic sign detection." arXiv preprint arXiv:1712.02463 (2017). [Link](https://github.com/olivesgatech/CURE-TSD)
- **YOLOv8**: Ultralytics. "YOLOv8 Documentation." (2023). [Link](https://github.com/ultralytics/ultralytics)
- **Snapshot Compressive Imaging**: Yuan, X. "Generalized alternating projection based total variation minimization for compressive sensing." In 2016 IEEE International Conference on Image Processing (ICIP), pp. 2539-2543. IEEE, 2016.
- **Deep Q-Network**: Mnih, V., et al. "Human-level control through deep reinforcement learning." Nature 518.7540 (2015): 529-533.

---

## Contact

**Manveer Anand**  
CS307 - Advanced Topics in Computer Vision  
GitHub: [@ManveerAnand](https://github.com/ManveerAnand)

For questions or collaboration inquiries, please open an issue on the GitHub repository.

---

## References

1. Temel, S., Kwon, G., Prabhushankar, M., & AlRegib, G. (2017). CURE-TSD: Challenging unreal and real environment traffic sign detection. arXiv preprint arXiv:1712.02463.

2. Ultralytics. (2023). YOLOv8: State-of-the-art object detection. GitHub repository.

3. Yuan, X., Liu, Y., Suo, J., & Dai, Q. (2016). Plug-and-play algorithms for large-scale snapshot compressive imaging. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition.

4. Mnih, V., Kavukcuoglu, K., Silver, D., et al. (2015). Human-level control through deep reinforcement learning. Nature, 518(7540), 529-533.

---

**Last Updated**: January 21, 2026
