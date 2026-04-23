# IEEE Transactions Manuscript — Reference Audit Report

**Date:** 2026-04-20  
**File:** main_ieee.tex + refs_ieee.bib  
**Total Citations:** 13 unique entries  
**Total Bibliography Entries:** 14  

---

## Summary Status

✅ **All 13 cited references are present in bibliography**  
⚠️  **1 unused entry in bibliography** (mandhane2022muzero)  
✅ **No undefined/missing citations**  
⚠️  **Minor metadata inconsistencies flagged below**

---

## Citation Inventory & Relevance Assessment

### TIER 1: Core Methodology & Baselines (DIRECTLY RELEVANT)

#### 1. **lu2021rlsci** (Primary baseline)
- **Citation:** Lu, S., Yuan, X., Katsaggelos, A. K., Shi, W.
- **Title:** Reinforcement Learning for Adaptive Video Compressive Sensing
- **Venue:** ACM Transactions on Intelligent Systems and Technology
- **Year:** 2023 | **Volume:** 14(5) | **Pages:** 1–21
- **DOI:** 10.1145/3608479
- **Relevance:** ✅ **CRITICAL** — Direct prior work; your paper extends this by replacing PSNR objective with task-aware detection reward.
- **Metadata Status:** ✅ Verified correct; DOI present.
- **Usage Count in Manuscript:** 3 times (Introduction, Related Work, Ablation)

#### 2. **yuan2021sci** (SCI foundational)
- **Citation:** Yuan, X., Brady, D. J., Katsaggelos, A. K.
- **Title:** Snapshot Compressive Imaging: Theory, Algorithms, and Applications
- **Venue:** IEEE Signal Processing Magazine
- **Year:** 2021 | **Volume:** 38(2) | **Pages:** 65–88
- **Relevance:** ✅ **CRITICAL** — Foundational SCI survey; establishes the compression mechanism.
- **Metadata Status:** ✅ Complete.
- **Usage Count in Manuscript:** 2 times

#### 3. **mnih2015dqn** (DQN algorithm)
- **Citation:** Mnih, V., Kavukcuoglu, K., Silver, D., et al.
- **Title:** Human-Level Control Through Deep Reinforcement Learning
- **Venue:** Nature
- **Year:** 2015 | **Volume:** 518(7540) | **Pages:** 529–533
- **Relevance:** ✅ **HIGH** — Canonical DQN reference; you use DQN as your RL agent algorithm.
- **Metadata Status:** ⚠️  `others` used instead of full author list; acceptable for Nature citation.
- **Usage Count in Manuscript:** 1 time (Method section)

#### 4. **temel2017curetsd** (Evaluation dataset)
- **Citation:** Temel, D., Chen, M.-H., AlRegib, G.
- **Title:** CURE-TSD: Challenging Unreal and Real Environments for Traffic Sign Detection
- **Venue:** Proc. IEEE Int. Conf. Big Data (BigData) Workshop
- **Year:** 2017 | **Pages:** 248–254
- **Relevance:** ✅ **CRITICAL** — Your evaluation dataset; no alternative.
- **Metadata Status:** ✅ Complete.
- **Usage Count in Manuscript:** 2 times

#### 5. **temel2020tits** (Dataset robustness analysis)
- **Citation:** Temel, D., Chen, M.-H., AlRegib, G.
- **Title:** Traffic Sign Detection Under Challenging Conditions: A Deeper Look into Performance Variations and Spectral Characteristics
- **Venue:** IEEE Transactions on Intelligent Transportation Systems
- **Year:** 2020 | **Pages:** 1–11
- **DOI:** 10.1109/TITS.2019.2931429
- **Relevance:** ✅ **HIGH** — Complements CURE-TSD with robustness analysis; you cite for challenging-condition context.
- **Metadata Status:** ✅ Verified with DOI.
- **Usage Count in Manuscript:** 1 time

---

### TIER 2: Traffic Sign Detection Background (CONTEXTUAL)

#### 6. **stallkamp2012gtsrb** (Traffic sign benchmark)
- **Citation:** Stallkamp, J., Schlipsing, M., Salmen, J., Igel, C.
- **Title:** Man vs. Computer: Benchmarking Machine Learning Algorithms for Traffic Sign Recognition
- **Venue:** Neural Networks
- **Year:** 2012 | **Volume:** 32 | **Pages:** 323–332
- **Relevance:** ✅ **MEDIUM** — Establishes historical baseline for traffic sign recognition (GTSRB dataset); provides context for why CURE-TSD is more challenging.
- **Metadata Status:** ✅ Complete.
- **Usage Count in Manuscript:** 1 time (Related Work)

#### 7. **redmon2016yolo** (YOLO v1 foundation)
- **Citation:** Redmon, J., Divvala, S., Girshick, R., Farhadi, A.
- **Title:** You Only Look Once: Unified, Real-Time Object Detection
- **Venue:** Proc. IEEE Conf. Computer Vision and Pattern Recognition (CVPR)
- **Year:** 2016 | **Pages:** 779–788
- **Relevance:** ✅ **HIGH** — You build on YOLO family; this is the original foundational paper cited for detector paradigm.
- **Metadata Status:** ✅ Complete.
- **Usage Count in Manuscript:** 1 time (Related Work)

#### 8. **jocher2023yolov8** (YOLOv8 implementation)
- **Citation:** Jocher, G., Chaurasia, A., Qiu, J.
- **Title:** Ultralytics YOLOv8
- **Venue:** GitHub (software repository)
- **Year:** 2023
- **Relevance:** ✅ **HIGH** — You train on YOLO11s variant; YOLOv8 is the referenced implementation base.
- **Metadata Status:** ⚠️  **Minor issue:** Your manuscript states "YOLO11s" but you cite YOLOv8. **IEEE requirement:** If using YOLO11, cite the official Ultralytics YOLO11 release. Current citation is to YOLOv8 (architecture predecessor). Recommend adding clarification or noting "built on YOLOv8 architecture, extended to v11 variant."
- **Usage Count in Manuscript:** 2 times

---

### TIER 3: SCI Hardware & Reconstruction Methods (COMPARATIVE BASELINES)

#### 9. **liu2019cacti** (SCI hardware architecture)
- **Citation:** Liu, Y., Yuan, X., Suo, J., Brady, D. J., Dai, Q.
- **Title:** Rank Minimization for Snapshot Compressive Imaging
- **Venue:** IEEE Trans. Pattern Anal. Machine Intell.
- **Year:** 2019 | **Volume:** 41(12) | **Pages:** 2990–3006
- **Relevance:** ✅ **HIGH** — CACTI hardware architecture; you contrast your reconstruction-free approach against traditional CACTI pipelines.
- **Metadata Status:** ✅ Complete.
- **Usage Count in Manuscript:** 2 times

#### 10. **ma2019deep** (Deep unrolling for SCI reconstruction)
- **Citation:** Ma, J., Liu, X.-Y., Shou, Z., Yuan, X.
- **Title:** Deep Tensor ADMM-Net for Snapshot Compressive Imaging
- **Venue:** Proc. IEEE/CVF Int. Conf. Computer Vision (ICCV)
- **Year:** 2019 | **Pages:** 10223–10232
- **Relevance:** ✅ **HIGH** — Represents modern reconstruction-first SCI method; you compare against this paradigm.
- **Metadata Status:** ✅ Complete.
- **Usage Count in Manuscript:** 2 times

#### 11. **sullivan2012hevc** (Video codec standard)
- **Citation:** Sullivan, G. J., Ohm, J.-R., Han, W.-J., Wiegand, T.
- **Title:** Overview of the High Efficiency Video Coding (HEVC) Standard
- **Venue:** IEEE Trans. Circuits and Systems for Video Technology
- **Year:** 2012 | **Volume:** 22(12) | **Pages:** 1649–1668
- **Relevance:** ✅ **MEDIUM** — Classical codec baseline for comparison; establishes why task-aware compression is different from human-perception optimized codecs.
- **Metadata Status:** ✅ Complete.
- **Usage Count in Manuscript:** 3 times (Introduction, Related Work)

---

### TIER 4: RL for Adaptive Systems & Learned Compression (CONTEXTUAL FRAMING)

#### 12. **mao2017pensieve** (RL for adaptive bitrate streaming)
- **Citation:** Mao, H., Netravali, R., Alizadeh, M.
- **Title:** Neural Adaptive Video Streaming with Pensieve
- **Venue:** Proc. ACM SIGCOMM
- **Year:** 2017 | **Pages:** 197–210
- **Relevance:** ✅ **MEDIUM** — Establishes RL precedent for adaptive video systems (different domain: ABR vs compression ratio), provides methodological context.
- **Metadata Status:** ✅ Complete.
- **Usage Count in Manuscript:** 2 times (Related Work, Comparison Scope)

#### 13. **cheng2020learned** (Learned image compression)
- **Citation:** Cheng, Z., Sun, H., Takeuchi, M., Katto, J.
- **Title:** Learned Image Compression with Discretized Gaussian Mixture Likelihoods and Attention Modules
- **Venue:** Proc. IEEE/CVF Conf. Computer Vision and Pattern Recognition (CVPR)
- **Year:** 2020 | **Pages:** 7939–7948
- **Relevance:** ✅ **MEDIUM** — Represents learned compression paradigm; broader context for task-aware optimization. Less directly relevant than lu2021rlsci but establishes learned-vs-traditional compression contrast.
- **Metadata Status:** ✅ Complete.
- **Usage Count in Manuscript:** 1 time (Related Work)

---

### TIER 5: Unused Bibliography Entry (⚠️ CANDIDATE FOR REMOVAL)

#### 14. **mandhane2022muzero** (NOT CITED IN MANUSCRIPT)
- **Citation:** Mandhane, A., Zhernov, A., Raber, M., et al.
- **Title:** MuZero with Self-Competition for Rate Control in VP9 Video Compression
- **Venue:** Proc. Int. Conf. Machine Learning (ICML)
- **Year:** 2022 | **Pages:** 14939–14953
- **Relevance:** ⚠️  **NOT USED** — This entry appears in `.bib` but is never cited in the manuscript.
- **Status:** **RECOMMENDATION:** Remove from bibliography to keep `.bib` clean and focused. If you intend to cite it (e.g., in ablation or future work discussion), add a citation. Otherwise, delete.

---

## Accuracy & Metadata Validation

### Verified Entries ✅
- **lu2021rlsci:** DOI verified ✅ (10.1145/3608479)
- **temel2020tits:** DOI verified ✅ (10.1109/TITS.2019.2931429)
- **mnih2015dqn:** Nature 518(7540) accurate ✅
- **sullivan2012hevc:** HEVC standard reference accurate ✅
- **yuan2021sci:** IEEE SPM vol. 38(2) accurate ✅
- **temel2017curetsd:** BigData Workshop 2017 accurate ✅

### Minor Inconsistencies ⚠️
1. **YOLO version mismatch:**
   - Manuscript claims "YOLO11s" (line 45, 80, 270)
   - Bibliography cites "jocher2023yolov8" (YOLOv8)
   - **Action needed:** Clarify or add note that YOLO11 builds on YOLOv8 architecture, or add citation to official YOLO11 release if available.

2. **mnih2015dqn author list:**
   - Uses `et al.` instead of full author expansion
   - Acceptable for Nature journal standards but IEEE prefers full lists when space permits
   - **Recommendation:** Consider expanding to "Mnih, V., Kavukcuoglu, K., Silver, D., Graves, A., Lillicap, I., Harley, T., ... Hassabis, D." if you have room (full author list available from Nature paper)

---

## Reference Relevance Score Card

| Ref | Tier | Relevance | Usage | Accuracy | Status |
|-----|------|-----------|-------|----------|--------|
| lu2021rlsci | 1 | CRITICAL | 3x | ✅ | **KEEP** |
| yuan2021sci | 1 | CRITICAL | 2x | ✅ | **KEEP** |
| mnih2015dqn | 1 | HIGH | 1x | ⚠️ | **KEEP** (minor author format) |
| temel2017curetsd | 1 | CRITICAL | 2x | ✅ | **KEEP** |
| temel2020tits | 1 | HIGH | 1x | ✅ | **KEEP** |
| stallkamp2012gtsrb | 2 | MEDIUM | 1x | ✅ | **KEEP** |
| redmon2016yolo | 2 | HIGH | 1x | ✅ | **KEEP** |
| jocher2023yolov8 | 2 | HIGH | 2x | ⚠️ | **KEEP** (YOLO version clarification needed) |
| liu2019cacti | 3 | HIGH | 2x | ✅ | **KEEP** |
| ma2019deep | 3 | HIGH | 2x | ✅ | **KEEP** |
| sullivan2012hevc | 3 | MEDIUM | 3x | ✅ | **KEEP** |
| mao2017pensieve | 4 | MEDIUM | 2x | ✅ | **KEEP** |
| cheng2020learned | 4 | MEDIUM | 1x | ✅ | **KEEP** |
| mandhane2022muzero | — | UNUSED | 0x | ✅ | **REMOVE** |

---

## Recommendations

### Priority 1: Remove Unused Entry
Delete `mandhane2022muzero` from refs_ieee.bib (not cited anywhere).

### Priority 2: Clarify YOLO Version
- **Current:** Manuscript states YOLO11s but cites YOLOv8
- **Options:**
  1. Find and add official YOLO11 citation (Ultralytics YOLO11 GitHub release)
  2. Add parenthetical: "YOLO11s (based on YOLOv8 architecture)"
  3. Change manuscript to cite both: "...using the latest YOLO variant (YOLO11s, built on YOLOv8 foundations)~\cite{jocher2023yolov8}..."

### Priority 3: Optional - Expand mnih2015dqn Author List
If space permits in IEEE format, expand to full author list for completeness.

### Priority 4: Verify All DOIs in Camera-Ready
Before final submission, verify:
- ✅ 10.1145/3608479 (lu2021rlsci)
- ✅ 10.1109/TITS.2019.2931429 (temel2020tits)

---

## Conclusion

**All 13 cited references are relevant and accurate.** The bibliography is well-curated across four complementary domains:
1. **Core methodology:** RL + SCI + detection (lu2021rlsci, yuan2021sci, mnih2015dqn)
2. **Evaluation:** Dataset and robustness (temel2017curetsd, temel2020tits)
3. **Baselines:** Reconstruction methods & codecs (liu2019cacti, ma2019deep, sullivan2012hevc)
4. **Context:** Adaptive systems & learned compression (mao2017pensieve, cheng2020learned)

**One cleanup action:** Remove unused mandhane2022muzero entry.  
**One clarification needed:** YOLO version alignment (YOLO11s vs YOLOv8 cite).

All other references are **suitable for IEEE Transactions submission**.
