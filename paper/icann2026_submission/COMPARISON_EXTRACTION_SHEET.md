# Comparison Extraction Sheet (Verified-First)

Purpose: Build a reviewer-safe comparison section using only verified bibliographic metadata and explicitly extracted technical parameters.

Rule: If a field is not explicitly confirmed from the source paper, mark it as NR (Not Reported / Not Yet Extracted).

## Parameters We Are Analyzing Across Papers

1. Bibliographic truth
- Title
- Authors
- Venue
- Year
- DOI

2. Problem framing
- Task domain
- Primary objective

3. Method design
- Adaptivity type (none, fixed, dynamic)
- Decision granularity (frame, chunk, sequence)
- Learning paradigm (RL, supervised, optimization, codec standard)
- Action/control space

4. Pipeline assumptions
- Requires SCI reconstruction (Yes/No)
- Detector in-the-loop (Yes/No)
- Safety-aware objective term (Yes/No)

5. Experimental protocol
- Dataset(s)
- Split protocol
- Hardware profile

6. Evaluation metrics
- Accuracy metric(s)
- Compression metric(s)
- Latency/FPS metric(s)
- Statistical testing (Yes/No)

7. Limits and transferability
- Reported limitations
- Closest comparable claim to our work

## Paper-by-Paper Extraction (Current Status)

### A) Lu et al. (closest direct comparator)
- Title: Reinforcement Learning for Adaptive Video Compressive Sensing
- Venue: ACM Transactions on Intelligent Systems and Technology
- Year: 2023
- DOI: 10.1145/3608479
- Task domain: Adaptive video compressive sensing
- Primary objective: NR
- Adaptivity type: Dynamic
- Decision granularity: NR
- Learning paradigm: RL
- Action/control space: NR
- Requires SCI reconstruction: NR
- Detector in-the-loop: NR
- Safety-aware objective term: NR
- Dataset(s): NR
- Accuracy metric(s): NR
- Compression metric(s): NR
- Latency/FPS metric(s): NR
- Key comparable claim versus our paper: RL-driven adaptive sensing/compression

### B) SCI survey (theory baseline)
- Title: Snapshot Compressive Imaging: Theory, Algorithms, and Applications
- Venue: IEEE Signal Processing Magazine
- Year: 2021
- DOI: 10.1109/MSP.2020.3023869
- Task domain: SCI foundations and applications
- Primary objective: SCI theory/algorithms survey
- Adaptivity type: NR
- Decision granularity: NR
- Learning paradigm: Mixed (survey)
- Requires SCI reconstruction: NR
- Detector in-the-loop: NR
- Safety-aware objective term: NR
- Dataset(s): NR
- Accuracy metric(s): NR
- Compression metric(s): NR
- Latency/FPS metric(s): NR
- Key comparable claim versus our paper: SCI operating model and trade-offs

### C) SCI optimization baseline
- Title: Rank Minimization for Snapshot Compressive Imaging
- Venue: IEEE Transactions on Pattern Analysis and Machine Intelligence
- Year: 2019
- DOI: 10.1109/TPAMI.2018.2873587
- Task domain: SCI reconstruction optimization
- Primary objective: Reconstruction quality via rank minimization
- Adaptivity type: None (as comparator category)
- Decision granularity: N/A
- Learning paradigm: Optimization
- Action/control space: N/A
- Requires SCI reconstruction: Yes
- Detector in-the-loop: No
- Safety-aware objective term: No
- Dataset(s): NR
- Accuracy metric(s): NR
- Compression metric(s): NR
- Latency/FPS metric(s): NR
- Key comparable claim versus our paper: Reconstruction-first vs detection-first pipeline

### D) SCI deep reconstruction baseline
- Title: Deep Tensor ADMM-Net for Snapshot Compressive Imaging
- Venue: ICCV 2019
- Year: 2019
- DOI: 10.1109/ICCV.2019.01032
- Task domain: SCI reconstruction with deep unrolling
- Primary objective: Better reconstruction from SCI measurements
- Adaptivity type: None (as comparator category)
- Decision granularity: N/A
- Learning paradigm: Deep unrolled optimization
- Action/control space: N/A
- Requires SCI reconstruction: Yes
- Detector in-the-loop: No
- Safety-aware objective term: No
- Dataset(s): NR
- Accuracy metric(s): NR
- Compression metric(s): NR
- Latency/FPS metric(s): NR
- Key comparable claim versus our paper: Reconstruction-first vs reconstruction-free detection

### E) Classical codec baselines

#### E1) H.264/AVC overview
- Title: Overview of the H.264/AVC Video Coding Standard
- Venue: IEEE Transactions on Circuits and Systems for Video Technology
- Year: 2003
- DOI: 10.1109/TCSVT.2003.815165
- Task domain: Standard video coding
- Primary objective: Rate-distortion efficient video coding
- Adaptivity type: Codec control, not RL scene policy
- Decision granularity: NR
- Learning paradigm: Hand-engineered codec standard
- Action/control space: NR
- Requires SCI reconstruction: No
- Detector in-the-loop: No
- Safety-aware objective term: No
- Dataset(s): N/A (standard overview)
- Accuracy metric(s): NR
- Compression metric(s): NR
- Latency/FPS metric(s): NR
- Key comparable claim versus our paper: Classical compression baseline family

#### E2) HEVC overview
- Title: Overview of the High Efficiency Video Coding (HEVC) Standard
- Venue: IEEE Transactions on Circuits and Systems for Video Technology
- Year: 2012
- DOI: 10.1109/TCSVT.2012.2221191
- Task domain: Standard video coding
- Primary objective: Higher coding efficiency than AVC
- Adaptivity type: Codec control, not RL scene policy
- Decision granularity: NR
- Learning paradigm: Hand-engineered codec standard
- Action/control space: NR
- Requires SCI reconstruction: No
- Detector in-the-loop: No
- Safety-aware objective term: No
- Dataset(s): N/A (standard overview)
- Accuracy metric(s): NR
- Compression metric(s): NR
- Latency/FPS metric(s): NR
- Key comparable claim versus our paper: Strong non-RL compression standard baseline

#### E3) VVC overview
- Title: Overview of the Versatile Video Coding (VVC) Standard and its Applications
- Venue: IEEE Transactions on Circuits and Systems for Video Technology
- Year: 2021
- DOI: 10.1109/TCSVT.2021.3101953
- Task domain: Next-gen standard video coding
- Primary objective: Better coding efficiency than HEVC
- Adaptivity type: Codec control, not RL scene policy
- Decision granularity: NR
- Learning paradigm: Hand-engineered codec standard
- Action/control space: NR
- Requires SCI reconstruction: No
- Detector in-the-loop: No
- Safety-aware objective term: No
- Dataset(s): N/A (standard overview)
- Accuracy metric(s): NR
- Compression metric(s): NR
- Latency/FPS metric(s): NR
- Key comparable claim versus our paper: Modern non-RL standard baseline

### F) RL context papers

#### F1) Pensieve
- Title: Neural Adaptive Video Streaming with Pensieve
- Venue: ACM SIGCOMM
- Year: 2017
- DOI: 10.1145/3098822.3098843
- Task domain: Adaptive bitrate streaming (QoE)
- Primary objective: QoE optimization via RL
- Adaptivity type: Dynamic
- Decision granularity: Chunk/session-level streaming decisions
- Learning paradigm: RL
- Action/control space: ABR decisions
- Requires SCI reconstruction: No
- Detector in-the-loop: No
- Safety-aware objective term: No
- Dataset(s): NR
- Accuracy metric(s): QoE-oriented, not detection mAP
- Compression metric(s): Bitrate adaptation
- Latency/FPS metric(s): NR
- Key comparable claim versus our paper: RL for adaptive compression control, different downstream objective

#### F2) DQN foundation
- Title: Human-level control through deep reinforcement learning
- Venue: Nature
- Year: 2015
- DOI: 10.1038/nature14236
- Task domain: General RL control
- Primary objective: Learn action policy from state observations
- Adaptivity type: Dynamic
- Decision granularity: Step-wise actions
- Learning paradigm: DQN
- Action/control space: Discrete action policy
- Requires SCI reconstruction: N/A
- Detector in-the-loop: N/A
- Safety-aware objective term: N/A
- Dataset(s): N/A
- Accuracy metric(s): N/A
- Compression metric(s): N/A
- Latency/FPS metric(s): N/A
- Key comparable claim versus our paper: Algorithmic foundation for our DQN policy

### G) Detection and dataset references

#### G1) YOLO v1 foundational detector
- Title: You Only Look Once: Unified, Real-Time Object Detection
- Venue: CVPR
- Year: 2016
- DOI: 10.1109/CVPR.2016.91
- Task domain: Real-time object detection
- Primary objective: End-to-end detection speed/accuracy
- Adaptivity type: N/A
- Decision granularity: N/A
- Learning paradigm: Supervised deep detection
- Action/control space: N/A
- Requires SCI reconstruction: No
- Detector in-the-loop: Yes (detector method itself)
- Safety-aware objective term: No
- Dataset(s): NR
- Accuracy metric(s): Detection metrics
- Compression metric(s): N/A
- Latency/FPS metric(s): Real-time detection framing
- Key comparable claim versus our paper: Detector lineage context for downstream task

#### G2) CURE-TSD related validated dataset paper
- Title: Traffic Sign Detection Under Challenging Conditions: A Deeper Look into Performance Variations and Spectral Characteristics
- Venue: IEEE Transactions on Intelligent Transportation Systems
- Year: 2020
- DOI: 10.1109/TITS.2019.2931429
- Task domain: Traffic sign detection under challenging conditions
- Primary objective: Robustness analysis under challenging conditions
- Adaptivity type: NR
- Decision granularity: NR
- Learning paradigm: Detection evaluation study
- Action/control space: N/A
- Requires SCI reconstruction: No
- Detector in-the-loop: Yes
- Safety-aware objective term: NR
- Dataset(s): CURE-TSD
- Accuracy metric(s): Detection performance metrics
- Compression metric(s): NR
- Latency/FPS metric(s): NR
- Key comparable claim versus our paper: Dataset/challenge baseline for traffic sign robustness

## Next Extraction Pass (to do)

1. Extract exact objective functions and explicit metrics from full texts for A, C, D, F1, and G2.
2. Add one strict comparison row: "Our method vs Lu 2023" with matched dimensions only.
3. Add one strict comparison row: "Our method vs fixed codec standards" as qualitative (non-matched protocol) comparison.
4. Keep NR where evidence is absent.

## Priority-5 Extraction Completed (Verifiable Fields Only)

Scope: A, C, D, E2, F1 from this sheet.

Evidence policy:
- DOI, venue, year, pages: Crossref-verified.
- Objective/method labels: from title and, where available, abstract snippet.
- Quantitative benchmark values are kept NR unless explicitly extracted from full text.

### P1) Lu et al. (A)
- DOI: 10.1145/3608479
- Type: Journal article
- Venue details: ACM TIST, 2023, Vol. 14, No. 5, pp. 1-21
- Task domain: Adaptive video compressive sensing (SCI)
- Primary objective: Adapt compression ratio B in video SCI using RL
- Adaptivity type: Dynamic
- Learning paradigm: RL
- Requires SCI reconstruction: Yes (explicitly stated in abstract context: multiple frames reconstructed from one snapshot)
- Detector in-the-loop: Yes (abstract notes object detection network performance is considered)
- Safety-aware objective term: NR
- Accuracy metric(s): NR (paper-specific full-text extraction pending)
- Compression metric(s): Compression ratio B (qualitative extraction confirmed)
- Latency/FPS metric(s): NR

### P2) Rank Minimization for SCI (C)
- DOI: 10.1109/TPAMI.2018.2873587
- Type: Journal article
- Venue details: IEEE TPAMI, 2019, Vol. 41, No. 12, pp. 2990-3006
- Task domain: SCI reconstruction optimization
- Primary objective: Reconstruction quality via rank-minimization formulation
- Adaptivity type: None
- Learning paradigm: Optimization
- Requires SCI reconstruction: Yes
- Detector in-the-loop: No
- Safety-aware objective term: No
- Accuracy metric(s): NR
- Compression metric(s): NR
- Latency/FPS metric(s): NR

### P3) Deep Tensor ADMM-Net for SCI (D)
- DOI: 10.1109/ICCV.2019.01032
- Type: Proceedings article
- Venue details: ICCV 2019, pp. 10222-10231
- Task domain: SCI reconstruction with deep unrolling
- Primary objective: Improve SCI reconstruction quality using deep tensor ADMM-Net
- Adaptivity type: None
- Learning paradigm: Deep unrolled optimization
- Requires SCI reconstruction: Yes
- Detector in-the-loop: No
- Safety-aware objective term: No
- Accuracy metric(s): NR
- Compression metric(s): NR
- Latency/FPS metric(s): NR

### P4) HEVC standard overview (E2)
- DOI: 10.1109/TCSVT.2012.2221191
- Type: Journal article
- Venue details: IEEE TCSVT, 2012, Vol. 22, No. 12, pp. 1649-1668
- Task domain: Standard video coding
- Primary objective: Coding-efficiency improvements over prior standards
- Adaptivity type: Codec control (not RL scene policy)
- Learning paradigm: Hand-engineered standard
- Requires SCI reconstruction: No
- Detector in-the-loop: No
- Safety-aware objective term: No
- Accuracy metric(s): NR (non-detection paper)
- Compression metric(s): Rate-distortion coding efficiency (qualitative)
- Latency/FPS metric(s): NR

### P5) Pensieve (F1)
- DOI: 10.1145/3098822.3098843
- Type: Proceedings article
- Venue details: ACM SIGCOMM 2017, pp. 197-210
- Task domain: Adaptive video streaming
- Primary objective: Neural policy for adaptive streaming decisions
- Adaptivity type: Dynamic
- Decision granularity: Chunk/session-level
- Learning paradigm: RL
- Requires SCI reconstruction: No
- Detector in-the-loop: No
- Safety-aware objective term: No
- Accuracy metric(s): NR (not object detection)
- Compression metric(s): Bitrate adaptation (qualitative)
- Latency/FPS metric(s): NR

## Paper-Ready Comparison Table Draft (for manuscript)

Use this table text in Related Work / Comparison section.

| Work | Objective | Adaptivity | Reconstruction Required | Detector in Loop | Safety-Aware Term | Direct Numeric Comparison to Ours | Claim Boundary |
|---|---|---|---|---|---|---|---|
| Lu et al., ACM TIST 2023 | RL-based adaptive B selection in SCI | Dynamic | Yes | Yes | NR | Partial (closest method family) | Compare method logic and operating principle; report numeric deltas only if protocol and metrics are matched |
| Liu et al., TPAMI 2019 | Rank-minimization SCI reconstruction | None | Yes | No | No | No | Qualitative family comparison only (reconstruction-first vs detection-first) |
| Ma et al., ICCV 2019 | Deep unrolled SCI reconstruction | None | Yes | No | No | No | Qualitative family comparison only |
| HEVC overview, IEEE TCSVT 2012 | Standard codec efficiency | Codec control | No | No | No | No | Standards-context comparison only; no cross-dataset numeric superiority claim |
| Pensieve, SIGCOMM 2017 | RL adaptive streaming policy | Dynamic | No | No | No | No | RL-control context only (different task objective) |

## Comparison Paragraph Starter (manuscript-ready)

Our comparison covers five established baselines spanning adaptive SCI control, reconstruction-first SCI methods, classical codec standards, and RL-based adaptive streaming. The closest methodological comparator is Lu et al. (ACM TIST 2023), which applies RL to adapt the SCI compression ratio B. In contrast to reconstruction-centric SCI methods (TPAMI 2019; ICCV 2019), our pipeline targets reconstruction-free detection with safety-aware reward shaping for critical-sign preservation. We include HEVC and Pensieve as standards and RL-control context, respectively; however, these are used for qualitative positioning due to non-matched downstream objectives and evaluation protocols. Accordingly, we only claim direct numerical superiority when dataset, metric definition, and operating protocol are explicitly aligned.
