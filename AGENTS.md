# Agent Instructions (AGENTS.md)

This repository implements a Reinforcement Learning-based Adaptive Video Compression framework (DQN + YOLOv8n + Snapshot Compressive Imaging) for autonomous driving scenarios using the CURE-TSD dataset.

## Core Architecture & Data Flow
1. **Dataset Pipeline:** Original videos go to `data/cure-tsd/`. The script `python scripts/generate_full_dataset.py` generates Snapshot Compressive Imaging (SCI) measurements which are output to `data/yolo_dataset_full/`.
2. **Object Detection:** YOLOv8n is trained on the SCI-compressed measurements, not raw frames.
3. **RL Agent (DQN):** Dynamically selects the compression ratio (B-value) at the frame level to optimize the trade-off between bandwidth and detection accuracy. Core environment and logic are in `src/phase1/`.
4. **Artifacts & Checkpoints:** 
   - Evaluation outputs: `outputs/` (e.g., CSV results, LaTeX tables).
   - RL Agent Checkpoints: `runs/rl_training_adaptive/`.
   - YOLO Weights/Logs: `runs/train/yolo_cure_tsd/`.

## Important Execution Quirks
- **Working Directories:** 
  - **Training scripts MUST be run from the `training/` directory**:
    ```bash
    cd training
    python train_yolo_local.py
    python train_rl_agent_adaptive.py
    ```
  - **Evaluation and Analysis scripts MUST be run from the root directory**:
    ```bash
    python scripts/evaluate_rl_agent.py --model runs/rl_training_adaptive/best_model_adaptive.pth
    python generate_performance_tables.py
    ```
- **Analysis Tools:** The root directory contains numerous high-level analysis scripts (`quick_training_summary.py`, `profile_pipeline_latency.py`, `comprehensive_performance_analysis.py`). Prefer using these to extract results rather than manually parsing CSVs in `outputs/`.

## Testing & Reproducibility
- **Unit Tests:** Run standard unit tests using `pytest tests/`.
- **Reproducibility Framework:** The project has strict scientific reproducibility requirements. Run the master test script from the root to verify end-to-end framework integrity:
  ```bash
  python test_framework.py
  ```

## Dependencies & Environment
- **Python:** 3.12 is strictly required. 
- **PyTorch:** 2.5.1 with CUDA 12.1+.
- Virtual environments (e.g., `conda env create -n rl_video_compression python=3.12`) are standard. Install dependencies via `pip install -r requirements.txt`.
