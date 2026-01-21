# Quick Reproducibility Guide

> **For Journal Reviewers**: This is a condensed version of [REPRODUCIBILITY.md](REPRODUCIBILITY.md)

## One-Command Reproduction

```bash
# Clone, setup, and reproduce everything
git clone https://github.com/yourusername/RL_Video_Compression.git
cd RL_Video_Compression
conda create -n rl_video_compression python=3.12 -y
conda activate rl_video_compression
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
python scripts/verify_installation.py
python scripts/download_dataset.py
python scripts/download_models.py
python scripts/reproduce_all.py  # 6-8 hours
python scripts/verify_results.py
```

## Expected Results

| Method | Mean mAP@0.5 | Improvement |
|--------|--------------|-------------|
| Fixed B=6 (Best) | 50.23% | - |
| Fixed B=20 (Worst) | 40.45% | - |
| Random Policy | 45.87% | +5.42% vs worst |
| **RL Agent** | **49.58%** | **+9.13% vs worst** |

**Statistical Significance**: RL vs Random: p < 0.001 ✓

## System Requirements

- **GPU**: 8GB+ VRAM (RTX 3060/4060 or better)
- **RAM**: 16GB+
- **Storage**: 50GB+ free
- **OS**: Windows/Linux/macOS
- **Time**: 6-8 hours for full reproduction

## File Checklist

After running reproduction, you should have:

```
✓ outputs/reproduction/fixed_baseline_results.csv (1,682 results)
✓ outputs/reproduction/random_policy_results.csv (280 results)
✓ outputs/reproduction/rl_agent_results.csv (280 results)
✓ outputs/reproduction/statistical_analysis.json
✓ outputs/reproduction/figures/*.png (5 plots)
✓ outputs/reproduction/reproduction_report.md
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| CUDA out of memory | Reduce batch size or use CPU mode |
| Dataset not found | Run `python scripts/download_dataset.py` |
| Models not found | Run `python scripts/download_models.py` |
| Results differ by >2% | Check random seed (use --seed 42) |

## Docker Alternative

For maximum reproducibility:

```bash
docker-compose build
docker-compose up -d rl_video_compression
docker exec -it rl_video_compression bash
python scripts/reproduce_all.py
```

See [DOCKER.md](DOCKER.md) for details.

## Getting Help

1. Check [FAQ.md](FAQ.md)
2. Read full guide: [REPRODUCIBILITY.md](REPRODUCIBILITY.md)
3. GitHub Issues: https://github.com/yourusername/RL_Video_Compression/issues
4. Email: your.email@university.edu

## Citation

```bibtex
@article{yourname2026rl,
  title={RL-Based Adaptive Video Compression for Traffic Sign Detection},
  author={Your Name and Co-authors},
  journal={Journal Name},
  year={2026}
}
```

---

**Reproducibility Status**: ✅ Fully Reproducible  
**Last Verified**: January 2026  
**Full Documentation**: [REPRODUCIBILITY.md](REPRODUCIBILITY.md)
