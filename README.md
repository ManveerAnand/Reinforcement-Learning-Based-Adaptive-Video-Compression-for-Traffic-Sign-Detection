# 🎯 RL-Based Adaptive Video Compression for Traffic Sign Detection

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5.1-red.svg)](https://pytorch.org/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8n-Ultralytics-green.svg)](https://github.com/ultralytics/ultralytics)
[![CUDA](https://img.shields.io/badge/CUDA-12.1-green.svg)](https://developer.nvidia.com/cuda-toolkit)

## 📋 Overview

This project implements an **RL-based adaptive video compression system** using **Snapshot Compressive Imaging (SCI)** for efficient traffic sign detection. The goal is to dynamically adjust compression ratios (B values) based on video content to maximize detection accuracy while minimizing bandwidth.

### Key Components
- **SCI Compression**: Binary mask-based snapshot compressive imaging (B ∈ {6,8,10,12,15,20})
- **YOLOv8n Detection**: Fine-tuned traffic sign detector on compressed measurements
- **RL Agent**: Adaptive B-value selection based on video characteristics (planned)

### Current Status
✅ **Phase 1-5 Complete**: Dataset generation, YOLO training finished  
✅ **RL Training**: 500 episodes, +9.0% improvement (45.48→49.58)  
✅ **Experiment 1 Complete**: Fixed baseline (280 videos, 1682 results)  
🔄 **Experiment 2 Running**: Random policy with checkpoint system  
⏳ **Experiment 3-4**: Package created for parallel execution  
📋 **Next**: Complete all benchmarking, statistical analysis, write paper

---

## 🚀 Quick Start

### Prerequisites
- **GPU**: NVIDIA GPU with 8GB+ VRAM (RTX 4060/3060 or better)
- **CUDA**: 12.1+
- **Python**: 3.12
- **Storage**: 50GB+ free space

### Environment Setup

```bash
# Create conda environment
conda create -n rl_video_compression python=3.12
conda activate rl_video_compression

# Install dependencies
pip install -r requirements.txt
```

### Training YOLOv8

```bash
cd training
python train_yolo_local.py
```

### Validation

```bash
cd training
python validate_yolo.py
```

---

## 📁 Project Structure

```
RL_Video_Compression/
├── data/
│   ├── cure-tsd/              # Original CURE-TSD dataset
│   │   ├── data/              # Raw video frames
│   │   └── labels/            # Ground truth annotations
│   ├── masks/                 # Binary SCI masks (B=6,8,10,12,15,20)
│   └── yolo_dataset_full/     # Generated YOLO dataset
│       ├── images/            # SCI compressed measurements (28,727 images)
│       ├── labels/            # YOLO format labels
│       └── data.yaml          # Dataset config
│
├── src/
│   ├── phase1/                # Feature extraction & compression
│   │   ├── feature_extractor.py
│   │   ├── label_parser.py
│   │   ├── sci_compressor.py
│   │   └── video_loader.py
│   └── phase5/                # Dataset generation
│       ├── dataset_builder.py
│       ├── label_converter.py
│       └── measurement_generator.py
│
├── training/                  # Training scripts
│   ├── train_yolo_local.py   # YOLOv8 training
│   ├── validate_yolo.py      # Validation script
│   └── test_inference.py     # Inference testing
│
├── scripts/                   # Utility & experiment scripts
│   ├── evaluate_fixed_baselines.py  # Experiment 1 (complete)
│   ├── evaluate_random_policy.py    # Experiment 2 (running, with checkpoints)
│   ├── evaluate_rl_agent.py         # Experiment 3 (ready)
│   ├── statistical_tests.py         # Experiment 4 (ready)
│   ├── generate_full_dataset.py
│   ├── recover_dataset.py
│   ├── check_dataset_progress.py
│   └── validate_remaining_experiments.py
│
├── outputs/                   # Experiment results
│   ├── fixed_baseline_results.csv   # 1,682 rows (complete)
│   ├── random_policy_checkpoint.json # Resume point
│   └── benchmarks/            # (reserved for final results)
│
├── runs/                      # Training outputs
│   ├── rl_training/           # RL agent checkpoints
│   │   ├── best_model.pth     # Best RL agent (episode 500)
│   │   └── checkpoint_ep*.pth # Training checkpoints
│   └── train/
│       └── yolo_cure_tsd/     # YOLO training run
│           ├── weights/       # best.pt (83.29% mAP50)
│           └── results.csv    # Training metrics
│
├── docs/                      # Documentation
│   ├── PROJECT_STATUS.md     # Current progress (75% complete)
│   ├── PAPER_OUTLINE.md      # Research paper structure
│   └── RESEARCH_PLAN.md      # Full research roadmap
│
├── tests/                     # Unit tests
└── requirements.txt           # Python dependencies
```

---

## 🎓 Dataset

**CURE-TSD** (Challenging Unreal and Real Environments - Traffic Sign Detection)

- **Size**: 1,805 videos, 14 traffic sign classes
- **Resolution**: 1628×1236 pixels, 10 FPS
- **Current Progress**: 28,727 images from 1,561 videos (86.5% complete)
  - Training: 19,975 images (1,073 videos, 68.7%)
  - Validation: 8,752 images (488 videos, 31.3%)
- **Link**: [Georgia Tech OLIVES Lab](https://github.com/olivesgatech/CURE-TSD)

### Traffic Sign Classes (14 Total)
- Speed limits: 30, 60, 80, 100, 120 km/h
- Signs: Stop, Give way, No passing, Priority road, Priority at intersection, No passing trucks, End of restrictions, Roundabout, Crosswalk

---

## 🔬 Methodology

### 1. SCI Compression
Compresses B consecutive frames into a single measurement using binary masks:

```
Y = Σ(mask_i × frame_i) for i = 1 to B
```

- **Compression Ratios**: B ∈ {6,8,10,12,15,20} → 83-95% bandwidth savings
- **Masks**: Random binary patterns (1628×1236)

### 2. YOLOv8n Detection
- **Architecture**: YOLOv8 Nano (3.0M parameters)
- **Training**: 100 epochs, batch=16, img=640, AdamW optimizer
- **Hardware**: RTX 4060 Laptop (8GB VRAM)
- **Performance**: 83.29% mAP50, 86.73% precision, 74.94% recall

### 3. RL Agent (Planned)
- **State**: Video features (optical flow, edge density, blur, etc.)
- **Action**: Select B ∈ {6,8,10,12,15,20}
- **Reward**: 0.7×mAP + 0.3×(B/20) - 2.0×critical_misses
- **Algorithm**: DQN or PPO

---

## 📊 Results

### Benchmarking Progress (280 Validation Videos)

| Experiment | Status | Output | Notes |
|------------|--------|--------|-------|
| **1. Fixed Baselines** | ✅ Complete | `fixed_baseline_results.csv` (1,681 rows) | 6 B-values × 280 videos |
| **2. Random Policy** | ✅ Complete | `random_policy_results.csv` (281 rows) | 280 videos (1 trial each) |
| **3. RL Agent** | 🔄 Running | `rl_agent_results.csv` (280 rows) | DQN fixed, ready to execute |
| **4. Statistical Tests** | ⏳ Queued | `statistical_tests_results.json` | After Exp 3 complete |

**Experiment 1 Summary** (Fixed B-values):
- **B=6**: 85 detections avg, 83.3% bandwidth savings
- **B=10**: 63 detections avg, 90.0% bandwidth savings  
- **B=20**: 43 detections avg, 95.0% bandwidth savings
- **Trade-off**: Lower B = higher accuracy, Higher B = more savings

### RL Training Results (500 Episodes)

| Metric | Initial | Final | Improvement |
|--------|---------|-------|-------------|
| **Score** | 45.48 | 49.58 | +9.0% |
| **Avg B-value** | ~14 | 18.14 | Learned higher compression |
| **Training Time** | - | 3.35 hours | RTX 4060 Laptop |

**Key Features**:
- ✅ Checkpoint system implemented (auto-resume)
- ✅ Memory management (periodic garbage collection)
- ✅ Package system for distributed execution
- ✅ Statistical testing infrastructure ready

### YOLO Training Results (SCI-Compressed Data)

| Metric | Value |
|--------|-------|
| mAP50 | 83.29% |
| mAP50-95 | 47.44% |
| Precision | 86.73% |
| Recall | 74.94% |

### Per-Class Performance (Top 5)

| Class | mAP50 | Images |
|-------|-------|--------|
| No passing trucks | 93.1% | 241 |
| Speed limit 120 | 89.1% | 435 |
| Priority road | 85.6% | 1,254 |
| Speed limit 80 | 84.4% | 1,068 |
| Crosswalk | 83.7% | 1,028 |

### Training Details
- **Duration**: 100 epochs (~16 hours)
- **Best Epoch**: Epoch 84 (84.06% mAP50)
- **Final Loss**: box=1.18, cls=0.75, dfl=1.05
- **VRAM Usage**: 1.1-2.0 GB stable

---

## 🛠️ Usage

### Running Benchmarking Experiments

**Prerequisites**:
```bash
conda activate rl_video_compression
export PYTHONPATH="/path/to/RL_Video_Compression"  # Linux/Mac
$env:PYTHONPATH = "D:\path\to\RL_Video_Compression"  # Windows
```

**Experiment 2: Random Policy** (with checkpoint resume):
```bash
python scripts/evaluate_random_policy.py
# Auto-resumes from checkpoint if interrupted
# Output: outputs/random_policy_results.csv (840 rows)
```

**Experiment 3: RL Agent**:
```bash
python scripts/evaluate_rl_agent.py
# Output: outputs/rl_agent_results.csv (280 rows)
```

**Experiment 4: Statistical Tests**:
```bash
python scripts/statistical_tests.py
# Compares RL vs Fixed B=10, generates significance tests
# Output: outputs/statistical_tests_results.json
```

**Create Experiment Package** (for distributed execution):
```bash
.\create_experiment_package.ps1
# Creates experiment_3_4_package.zip with all dependencies
# Transfer to another computer and run Exp 3-4 in parallel
```

### Generate Dataset

```bash
python scripts/generate_full_dataset.py
```

### Train YOLOv8

```bash
cd training
python train_yolo_local.py  # Auto-resumes from checkpoint if exists
```

### Validate Model

```bash
cd training
python validate_yolo.py
```

### Check Progress

```bash
python scripts/check_dataset_progress.py
```

---

## 🎯 Research Objectives

1. ✅ **Dataset Generation**: Create SCI compressed measurements from CURE-TSD (28,727 images)
2. ✅ **YOLO Fine-tuning**: Train detector on compressed data (83.29% mAP50)
3. ✅ **RL Agent Training**: DQN agent for adaptive B-selection (500 episodes, +9.0%)
4. 🔄 **Benchmarking**: Full evaluation on 280 validation videos
   - ✅ Experiment 1: Fixed baselines complete
   - 🔄 Experiment 2: Random policy running
   - ⏳ Experiment 3: RL agent queued
   - ⏳ Experiment 4: Statistical analysis queued
5. 📋 **Paper Writing**: Document methodology, results, and analysis

**Progress**: ~85% complete (benchmarking in progress)

---

## 📈 Expected Impact

- **Bandwidth Savings**: 15-20% vs fixed compression schemes
- **Detection Accuracy**: Near-optimal (within 5% of uncompressed)
- **Real-time Capable**: Adaptive selection <33ms latency
- **Safety-Aware**: Prioritizes critical sign detection

---

## 📚 Key References

1. CURE-TSD Dataset: [Temel et al. (2017)](https://github.com/olivesgatech/CURE-TSD)
2. YOLOv8: [Ultralytics (2023)](https://github.com/ultralytics/ultralytics)
3. Snapshot Compressive Imaging: [Yuan et al. (2016)](https://opg.optica.org/oe/fulltext.cfm?uri=oe-24-17-18829)
4. Deep Q-Network: [Mnih et al. (2015)](https://www.nature.com/articles/nature14236)

---

## 🐛 Troubleshooting

### CUDA Out of Memory
```python
# Reduce batch size in train_yolo_local.py
'batch': 8,  # Default is 16
```

### Dataset Not Found
```bash
# Verify data.yaml path
data: ../data/yolo_dataset_full/data.yaml
```

### Training Not Resuming
```bash
# Check checkpoint exists
ls ../runs/train/yolo_cure_tsd/weights/last.pt
```

---

## 👥 Authors

**Manveer Anand**  
CS307 - Advanced Topics in Computer Vision  
[GitHub](https://github.com/ManveerAnand/Adaptive_video_compression)

---

## 📝 License

MIT License - See LICENSE for details

---

## 🔗 Links

- **Documentation**: [docs/PROJECT_STATUS.md](docs/PROJECT_STATUS.md)
- **Repository**: [github.com/ManveerAnand/Adaptive_video_compression](https://github.com/ManveerAnand/Adaptive_video_compression)

---

**Last Updated**: November 15, 2025  
**Status**: 🔄 **Benchmarking In Progress** - Exp 1 complete (1682 results), Exp 2-4 running
