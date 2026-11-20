# Survey of Excellent Domestic Open-Source Perception AI Solutions in China

> Research date: 2026-11-20
> Scope: the conceptual framework of Perception AI, plus domestic open-source solutions across three directions — Sensor AI / Video AI / Sound AI — with fusion directions such as omni-modal large models and embodied perception.
> Note: this report focuses on solutions **led by Chinese institutions/teams and publicly available (GitHub / Hugging Face / ModelScope / Gitee)**. Performance data come from public papers, official documentation, and authoritative media, with evaluation settings and versions labeled. Some metrics change as versions iterate — please verify against the latest releases when citing.

---

## Table of Contents

1. [Conceptual Framework: What is Perception AI](#1-conceptual-framework)
2. [Domestic Open-Source Ecosystem Overview](#2-domestic-open-source-ecosystem-overview)
3. [Sensor AI: Sensor Perception](#3-sensor-ai-sensor-perception)
4. [Video AI: Video Perception](#4-video-ai-video-perception)
5. [Sound AI: Audio / Speech Perception](#5-sound-ai-audio--speech-perception)
6. [Fusion Directions: Omni-Modal Large Models and Embodied Perception](#6-fusion-directions-omni-modal-large-models-and-embodied-perception)
7. [Cross-Comparison and Selection Guidance](#7-cross-comparison-and-selection-guidance)
8. [Trend Assessment and Gap Analysis](#8-trend-assessment-and-gap-analysis)
9. [Appendix: Quick-Reference Project List and Sources](#appendix)

---

## 1. Conceptual Framework

### 1.1 Definition and Positioning

**Perception AI** refers to the ability of machines, using artificial intelligence (especially machine learning, computer vision, speech processing, and related techniques), to "sense—understand—respond" to the external environment through sensors, simulating human senses such as sight, hearing, touch, smell, and taste[^1].

A widely cited evolution framework comes from NVIDIA CEO Jensen Huang's division of AI stages, which clearly positions Perception AI in its historical context[^2]:

| Stage | Meaning | Examples |
|-------|---------|----------|
| **① Perception AI** | Recognizing and interpreting world information: images, sound, vibration, temperature, and other sensor data | AlexNet (2012), defect detection, lane-line recognition |
| **② Generative AI** | Understanding semantics and generating new content: text, images, code | GPT, text-to-image/video (still in the software world) |
| **③ Agentic AI** | Reasoning, planning, and autonomously executing multi-step tasks in software environments | Tool calling, workflow orchestration |
| **④ Physical AI** | Combining models + sensors + control + hardware to perceive, reason, and act in the physical world | Robotics, autonomous driving, VLA models |

> Perception AI is the "sensory foundation" of all subsequent stages — without high-quality perception, higher-level generation, decision-making, and execution are impossible ("perception quality determines the ceiling of everything downstream")[^2].

### 1.2 Classification by Perceptual Modality

Classified by the standard of simulating human senses[^1][^3]:

| Modality | English | Typical Signals | Corresponding Section |
|----------|---------|-----------------|-----------------------|
| Visual perception | Visual | Images / video / multispectral | Section 4, Video AI |
| Auditory perception | Auditory | Speech / audio / voiceprint / abnormal sound | Section 5, Sound AI |
| Tactile perception | Tactile | Pressure / texture / force / slip | Section 3, Sensor AI |
| Environmental / sensor perception | Sensor | IMU / point cloud / radar / vibration / temperature | Section 3, Sensor AI |
| Text / language perception | Textual | Text / semantics | (belongs to NLP, not a focus of this report) |

### 1.3 Key Technology Trends

Industry research identifies three major trends in perception intelligence[^1]:

1. **Beyond-human perceptual range**: multispectral imaging brings ultraviolet, infrared, and terahertz into acquisition; AI parses superhuman sensory data (e.g., infrared thermography detecting early mechanical failures).
2. **Multimodal fusion**: a single modality struggles with complex scenarios; fusing infrared, visible light, acoustic spectra, and electrical signals builds cross-modal models (e.g., industrial monitoring analyzing spectral images and acoustic features together).
3. **On-device closed loop**: AI algorithms are burned into on-device modules, shifting from "collect—cloud analyze" to "local real-time closed loop," cutting latency by up to 90%.

Market-size corroboration: the global perception-intelligence market grew from about RMB 200.1 billion in 2020 to about RMB 292.9 billion in 2024 (CAGR 10%), and is forecast to reach RMB 589.1 billion in 2029 (CAGR 15%), with visual perception accounting for about 62% of the 2024 market[^1].

---

## 2. Domestic Open-Source Ecosystem Overview

China's Perception AI open-source strength can be grouped into three kinds of actors, forming a complete stack from underlying frameworks to upper-layer models:

### 2.1 Foundational Deep-Learning Frameworks (the base)

| Framework | Institution | Features | Perception-Related Suites |
|-----------|-------------|----------|---------------------------|
| **PaddlePaddle** | Baidu | Domestic full-featured framework, industrial-grade validation, adapted to domestic chips | PaddleCV / PaddleVideo / PaddleSpeech / Paddle3D / PaddleOCR |
| **MindSpore** | Huawei | Full-stack solution, paired with Ascend chips | Device-edge-cloud collaboration, autonomous driving / vision |
| **MegEngine** | Megvii | Integrated training and inference | Industrial deployment of detection / classification |

### 2.2 Representative Open-Source Institutions and Their "Full Suites"

| Institution | Open-Source Brand / System | Directions Covered |
|-------------|----------------------------|--------------------|
| **Shanghai AI Laboratory** | Intern (Shusheng) series | InternVL multimodal, Intern-Robotics embodied, OpenCompass evaluation, LMDeploy deployment |
| **OpenMMLab** (initiated by MMLab, CUHK) | 30+ MM-series algorithm libraries | Detection / segmentation / pose / video / 3D / deployment |
| **Alibaba Tongyi / DAMO Academy** | Qwen, FunASR, CosyVoice | Omni-modal large models, ASR, TTS |
| **Xiaohongshu FireRed team** | FireRedASR / FireRedTTS / FireRedChat | Speech recognition / synthesis / full-duplex dialogue |
| **Tencent Hunyuan** | HunyuanVideo, etc. | Video generation, omni-modality |
| **Baidu PaddlePaddle** | Paddle series | Full pipeline for CV / Speech / Video / 3D |
| **Ant LingBot** | LingBot series | Tactile foundation models, VLA |

> A notable phenomenon: open sourcing has become a long-term strategy for domestic vendors. In just July–August 2025, Chinese vendors consecutively open-sourced 33 and then 31 various large models on the Hugging Face Chinese community; the number of Qwen-derived models surpassed 100,000, ranking first among global open-source model families[^4][^5].

---

## 3. Sensor AI: Sensor Perception

Sensor AI refers to intelligent analysis of non-visual-image sensor signals (tactile, point cloud / LiDAR, IMU, vibration, industrial acoustics, multimodal sensor fusion). This is the direction where domestic breakthroughs have been most concentrated in recent years and which carries the strongest "embodied intelligence" character.

### 3.1 Tactile Perception (Tactile AI) — the new explosion point in 2025

| Project | Institution | Core Capability | License / Link |
|---------|-------------|-----------------|----------------|
| **LingBot-Touch** | Ant LingBot | General tactile foundation model, pretrained on nearly 5,000 hours of real tactile data, unifies **magneto-tactile / visuo-tactile / piezoresistive** signals into one representation; integrated into LingBot-VLA; boosts success rates on 26 real-robot tasks by 14–26 percentage points | Apache-2.0; GitHub `Robbyant/lingbot-touch`[^6] |
| **TipSim** | Huawei Kechuang (Huawiko) | Industry's first multimodal tactile simulation platform, based on MuJoCo, outputs contact-location / force-distribution array data and feeds them into the control loop, solving the "tactile data shortage" | Open core interfaces; GitHub `tiplab/TipSim`[^7] |

> Significance: LingBot-Touch extends embodied perception from "vision" to the "fingertips," solving the problem that in fine manipulation vision is occluded and force/slip cannot be judged; TipSim enables tactile data to be collected at scale in simulation, compressing the sim-to-real gap.

### 3.2 LiDAR / Point Cloud / 3D Perception

| Project | Institution | Core Capability | Notes |
|---------|-------------|-----------------|-------|
| **MMDetection3D** | OpenMMLab | 3D detection platform supporting PointPillars / SECOND / BEVFusion / Point Transformer, etc., covering KITTI / nuScenes | Apache-2.0, official BEVFormer implementation[^8] |
| **Paddle3D** | Baidu | Paddle 3D perception suite, covering monocular / point-cloud, detection / segmentation, adapted to domestic chips, full-process deployment | Chinese documentation friendly, TensorRT / OpenVINO acceleration[^8] |
| **OpenPCDet** | OpenMMLab | Focused on LiDAR 3D detection | Mainstream library for point-cloud detection[^8] |
| **lidar-slam-detection (LSD)** | Tongji University alumni team | One-stop: SLAM with FAST-LIO + G2O + PointPillars/CenterPoint detection + AB3DMOT tracking; CenterPoint achieves 77.55% AMOTA on the Waymo validation set | GitHub 750+ stars, multimodal fusion in practice[^9] |
| **BEVFormer / BEVFusion** | Tsinghua + SenseTime, etc. / MIT + Tsinghua | Multi-camera BEV perception, multimodal sensor fusion; BEVFusion reduces BEV pooling latency by 40×+ | Academic source of China's BEV mass-production wave[^10] |

### 3.3 Industrial Acoustics / Abnormal Sound Detection (Sensor + Sound crossover)

| Project / Dataset | Institution | Core Capability | Notes |
|-------------------|-------------|-----------------|-------|
| **IMPACT (Industrial Acoustic Transformer)** | (open-sourced 2025) | World's first pretrained large model dedicated to industrial acoustics; DINOS dataset (74,149 samples, 1,093 hours, covering CNC / machine tools / motors / pumps), robust to strong noise, cross-device generalization | MIT, commercially usable; ~300M–1.2B[^11] |
| **Qwen3-ASR-Industrial** | Alibaba | Industrial voiceprint-specific ASR + anomaly-detection foundation, three-stage denoising adapted to 100dB+ workshops, device-edge-cloud architecture (0.6B edge / 1.7B server), voiceprint + vibration + PLC fusion diagnosis | Apache-2.0[^11] |
| **MIMII / MIMII-DG / DCASE2025-Task2** | International datasets (widely used domestically) | Benchmarks for normal / abnormal machine sounds, unsupervised anomaly detection and domain generalization | Standard training / evaluation datasets[^12] |

> ⚠️ Note: some "2025 large-model" information in the industrial voiceprint area comes from technical blogs; before deployment, verify the official GitHub / HF repositories and licenses.

---

## 4. Video AI: Video Perception

Video AI splits into two main lines — "video understanding / analysis" and "video generation" — and China has world-class open-source solutions on both.

### 4.1 Video Understanding and Analysis

| Project | Institution | Tasks Covered | Features |
|---------|-------------|---------------|----------|
| **OpenMMLab video family** | MMLab, CUHK | MMAction2 (action recognition / spatiotemporal detection), MMTracking (VID video object detection / MOT multi-object tracking / SOT single-object tracking / VIS video instance segmentation), MMDetection | Modular, rich Model Zoo, Apache-2.0; MMTracking 3.9k+ stars, unifies four tasks[^13] |
| **PaddleVideo** | Baidu | Video classification, temporal action detection, behavior recognition | PP-TSMv2 reaches 75.16% on Kinetics-400 (22M, CPU 456ms/10s); 2025 adds TokenShift, YOWO spatiotemporal detection, lightweight models[^14] |
| **ByteTrack** | ByteDance | Multi-object tracking | Efficient tracker widely used in MOT[^13] |
| **InternVideo series** | Shanghai AI Lab | Video foundation models / video understanding | Multimodal video backbone in the Intern ecosystem |

### 4.2 Video Generation (World Models / Content Generation)

| Model | Institution | Open Source? | Params / Specs | Highlights |
|-------|-------------|--------------|----------------|------------|
| **HunyuanVideo** | Tencent Hunyuan | ✅ | 13B | DiT architecture, dual-stream-to-single-stream hybrid, 3D-VAE, native shot transitions[^15] |
| **HunyuanVideo 1.5** | Tencent Hunyuan | ✅ | 8.3B | Lightweight, runs on 14G VRAM, SSTA sparse attention, 5–10s 480p/720p → upscaled 1080p[^15] |
| **Wan2.1 (Tongyi Wanxiang)** | Alibaba | ✅ | 14B / 1.3B | 720p, open source[^16] |
| **CogVideoX 1.5** | Zhipu | ✅ | 5B | 1360×768, 5/10s[^16] |
| **Open-Sora / Open-Sora-v2** | Colossal-AI | ✅ | 11B | Open-source reproduction of the Sora route[^16] |
| **HunyuanVideo-Foley** | Tencent Hunyuan | ✅ | — | End-to-end video sound-effect generation, synchronized with video[^15] |
| Kling / Vidu / Jimeng / MiniMax Hailuo | Kuaishou / Shengshu / ByteDance / MiniMax | ❌ (closed-source commercial) | — | China's top-tier commercial video generation, not open source[^16] |

> Domestic open-source video generation has formed a landscape of "multiple strong open players — Tencent / Alibaba / Zhipu / Colossal-AI — plus closed-source commercial players — Kuaishou / ByteDance." Open models have made notable progress in parameter efficiency (e.g., 8.3B reaching flagship experience).

---

## 5. Sound AI: Audio / Speech Perception

This is the most active open-source direction in China and the most mature for industrial deployment, having formed a complete layered ecosystem of **model layer — toolkit layer — deployment runtime layer**[^17].

### 5.1 Three-Layer Ecosystem Structure

```
Model layer (recognition / synthesis): Paraformer, SenseVoice, Fun-ASR-Nano, FireRedASR, Qwen3-ASR, CosyVoice, Fish-Speech...
Toolkit layer (managing / composing models): FunASR, PaddleSpeech, WeNet...
Deployment runtime layer (running models on devices): sherpa-onnx (on-device, 10.8k stars), FunASR Docker, vLLM...
```

> In practice these are often combined, e.g., "SenseVoice model + FunASR API call + sherpa-onnx deployed to phone / HarmonyOS"[^17].

### 5.2 Automatic Speech Recognition (ASR)

| Model | Institution | Time | Positioning and Metrics |
|-------|-------------|------|-------------------------|
| **Paraformer** | Alibaba DAMO Academy | 2022 | FunASR's earliest self-developed model, most mature integration, has a streaming version, trained on 60,000 hours[^17][^18] |
| **SenseVoice-Small / SenseVoice-L** | Alibaba Tongyi | 2024/07 | Not only recognizes content but also **emotion and sound events** (applause / laughter); fastest processing; CER 4.47 (L)[^17][^19] |
| **Fun-ASR-Nano** | Alibaba Tongyi | 2025/12 | Tens of millions of hours of data, streaming transcription, 7 dialects / 26 accents, 93%+ accuracy in workshops, first-character latency <160ms[^17][^11] |
| **FireRedASR** | Xiaohongshu | 2025/01 (v2: 2026/02) | **#1 on public benchmarks for Chinese recognition accuracy**: LLM version CER 3.05%, AED version 3.18%, beating Seed-ASR (3.33) / Whisper (9.86) / Paraformer (4.56); supports lyric recognition, dialects, English; industrially commercially usable[^19][^20] |
| **Qwen3-ASR** | Alibaba Cloud Qwen | 2026/01 | 22 Chinese dialects, provides standalone install package qwen-asr[^17] |
| **WeNet** | Mobvoi + NWPU, etc. | — | "Production ready" recognition engine, widely used in industry[^18] |
| **PaddleSpeech** | Baidu | — | Integrated ASR + TTS + translation, Chinese ASR error rate 6.8%, supports domestic chips, 5 lines of code to start[^21] |

### 5.3 Text-to-Speech (TTS)

| Model | Institution | Stars / License | Positioning |
|-------|-------------|-----------------|-------------|
| **CosyVoice 3.0** | Alibaba Tongyi | 23.9k / Apache-2.0 | De facto standard for industrial Chinese TTS; 9 languages + 18 dialects; bidirectional streaming 150ms; zero-shot similarity 90%+; emotional instruction control[^22] |
| **Fish Speech S2 Pro** | Fish Audio | 33k / Research License (authorization needed for commercial use) | Dual-AR architecture, 10M hours / 80+ languages, emotion tags, 44.1kHz high fidelity, Chinese WER 0.54%[^22] |
| **GPT-SoVITS** | RVC-Boss | 59k / MIT | Few-shot powerhouse: 5-second cloning, 1-minute fine-tuning, Chinese/English/Japanese/Korean/Cantonese; overall commercially usable[^22] |
| **F5-TTS** | SWivid | 15.3k / code MIT, weights CC-BY-NC | Pure flow matching + DiT, pip-installable, Apple Silicon support[^22] |
| **ChatTTS** | 2noise | Non-commercial restriction | Dialogue scenarios, actively adds colloquial features like "um" / "you know," high realism[^22] |
| **IndexTTS** | Bilibili | 24.3k / custom license | Contact official for commercial use[^22] |
| **FireRedTTS / FireRedTTS-2** | Xiaohongshu | Open source, commercially usable | Zero-shot voice cloning; TTS-2 focuses on multi-turn dialogue / podcast generation, 4 speakers, open-source SOTA[^4] |

### 5.4 Full-Duplex Speech Interaction and Audio Understanding

| Project | Institution | Highlights |
|---------|-------------|------------|
| **FireRedChat** | Xiaohongshu | Industry's first fully open-source full-duplex speech dialogue system, end-to-end approaching industrial grade; one-stop opening of VAD / ASR / TTS / context-aware TTS / audio LLM / Dify support, private deployment supported[^4] |
| **SenseVoice** | Alibaba | Audio event detection + emotion recognition ("understanding what's between the lines")[^17] |

---

## 6. Fusion Directions: Omni-Modal Large Models and Embodied Perception

Single perceptual modalities are evolving toward "unified multimodal perception," the highest form of Perception AI.

### 6.1 End-to-End Omni-Modal Large Models (unified text / image / audio / video)

| Model | Institution | Release | Modalities and Highlights |
|-------|-------------|---------|---------------------------|
| **Qwen2.5-Omni-7B** | Alibaba Tongyi | 2025-03-27 | First end-to-end omni-modal large model, text / image / audio / video input, real-time text + speech output; Thinker-Talker dual-core architecture, TMRoPE; OmniBench record, beating Gemini-1.5-Pro; 7B deployable on phones, open for commercial use[^5] |
| **InternVL3.5** | Shanghai AI Lab | 2025-09 | General multimodal large model, 9 sizes (1B–241B), flagship MMMU 77.7 highest among open models; general multimodal perception surpasses GPT-5; GUI agents and embodied spatial perception stand out; 23M+ downloads across the series[^23] |
| **Kimi-Audio** | Moonshot AI | 2025 | Open-source audio foundation model[^4] |
| **Step-Audio 2** | StepFun | 2025 | End-to-end multimodal speech large model, 1.8B, real-time streaming <150ms[^4][^11] |

### 6.2 Embodied Perception

A review by Zhang Weinan's team at Harbin Institute of Technology divides embodied intelligence into three modules — "embodied perception — embodied decision — embodied execution" — of which **embodied perception** focuses on object perception (geometry / articulation / physical properties) and scene perception (metric maps / topological maps), and proposes bidirectional enhancement of "perception for action" and "action for perception"[^24].

| Project | Institution | Content |
|---------|-------------|---------|
| **Intern-Robotics full-stack engine** | Shanghai AI Lab | Navigation large model InternVLA·N1, manipulation large models InternVLA·M1/A1, reward model VLAC, humanoid control InternHumanoid; datasets InternScenes (40k+ indoor scenes), OmniWorld (300M+ video frames); 1.1M+ model/dataset downloads, repeatedly topped HF Robotics trending[^25] |
| **LingBot-VLA + LingBot-Touch** | Ant LingBot | Vision-language-action model + tactile foundation model, extending perception to the "fingertips"[^6] |

---

## 7. Cross-Comparison and Selection Guidance

### 7.1 Maturity Assessment of Domestic Open Source by Direction

| Direction | Open-Source Maturity | International Competitiveness | Representative Projects | Notes |
|-----------|----------------------|-------------------------------|-------------------------|-------|
| **Sound AI (ASR/TTS)** | ⭐⭐⭐⭐⭐ | World leading | FireRedASR, CosyVoice, SenseVoice, Qwen2.5-Omni | Chinese-scenario accuracy SOTA globally, complete layered ecosystem |
| **Video AI (understanding/detection)** | ⭐⭐⭐⭐⭐ | World leading | OpenMMLab, PaddleVideo, ByteTrack | Systematic toolchains, mature engineering deployment |
| **Video AI (generation)** | ⭐⭐⭐⭐☆ | First tier | HunyuanVideo, Wan2.1, CogVideoX | Active open source, high parameter efficiency; strong closed-source commercial too |
| **Omni-modal large models** | ⭐⭐⭐⭐⭐ | World leading | Qwen2.5-Omni, InternVL3.5 | 7B on-device deployable, surpassing same-tier closed models |
| **Sensor AI (3D/LiDAR)** | ⭐⭐⭐⭐☆ | Leading | MMDetection3D, Paddle3D, BEVFusion | Driven by autonomous-driving mass production, complete toolchains |
| **Sensor AI (tactile)** | ⭐⭐⭐☆☆ | Rapidly catching up | LingBot-Touch, TipSim | New explosion in 2025, datasets still a bottleneck |
| **Sensor AI (industrial acoustics)** | ⭐⭐⭐☆☆ | Catching up | IMPACT, Qwen3-ASR-Industrial | Datasets / benchmarks still mostly international |

### 7.2 Selection Guidance by Typical Scenario

| Need / Scenario | Recommended Solution | Reason |
|-----------------|----------------------|--------|
| High-precision Chinese transcription | FireRedASR-LLM / Fun-ASR-Nano | #1 on public CER benchmarks / industrial-grade streaming + dialects |
| Speech emotion + event understanding | SenseVoice | Recognizes emotion / applause / laughter, fast |
| Chinese TTS (real-time interaction) | CosyVoice 3.0 | 150ms streaming, dialects, Apache-2.0 commercially usable |
| Few-shot voice cloning | GPT-SoVITS (MIT, commercially usable) | 5-second cloning, mature ecosystem |
| On-device offline speech | sherpa-onnx + SenseVoice/Paraformer | Offline on phones / Raspberry Pi / HarmonyOS |
| Video detection / tracking / action | OpenMMLab (MMDetection + MMTracking + MMAction2) | Integrated toolchain, Model Zoo |
| Video action recognition (lightweight deployment) | PaddleVideo PP-TSMv2 | 22M small model, adapted to domestic chips |
| Autonomous driving / robot 3D perception | MMDetection3D / Paddle3D / BEVFusion | BEV multimodal fusion, full-process deployment |
| Unified vision + sound + text understanding | Qwen2.5-Omni-7B / InternVL3.5 | On-device deployable, multimodal SOTA |
| Robot tactile fine manipulation | LingBot-Touch + TipSim | Unified tactile representation, simulation supplements data |
| Unified text/image/audio/video generation + dialogue | FireRedChat / Qwen2.5-Omni | Full duplex, full-chain open source, privately deployable |

### 7.3 Three Key Judgments When Selecting

1. **Model vs toolkit vs runtime**: the three are not on the same layer and must be combined (e.g., SenseVoice + FunASR + sherpa-onnx). First identify which layer you are missing.
2. **License**: always verify before commercial use. Commercially friendly: Apache-2.0 (CosyVoice, Qwen2.5-Omni, InternVL, LingBot-Touch), MIT (GPT-SoVITS, IMPACT); authorization required / non-commercial: Fish Speech S2 Pro, ChatTTS, F5-TTS weights, IndexTTS.
3. **Data sources and evaluation settings**: accuracy metrics (CER / mAP / NDS) depend on test sets and versions; cross-model comparison needs a unified benchmark (e.g., AISHELL / WenetSpeech for Chinese ASR).

---

## 8. Trend Assessment and Gap Analysis

### 8.1 Trends

1. **From single modality to omni-modal unification**: Qwen2.5-Omni and InternVL3.5 represent "one model perceives all modalities" becoming mainstream; 7B on-device deployment brings Perception AI into phones and robots.
2. **From "recognition" to "understanding + interaction"**: SenseVoice's emotion / event detection and FireRedChat's full-duplex dialogue mean perception no longer just outputs labels but understands context and responds with emotion.
3. **From cloud to on-device closed loop**: sherpa-onnx, Qwen2.5-Omni-7B, and edge versions of industrial voiceprint models (0.6B) drive "local real-time perception—decision," meeting low-latency and privacy needs.
4. **From vision to multisensory embodiment**: tactile (LingBot-Touch), 3D space (InternVLA), vibration / acoustics and other multimodal sensor fusion make Perception AI the entry point to embodied intelligence / Physical AI.
5. **Open source as strategy**: domestic vendors build ecosystem influence through open source — Xiaohongshu in audio, OpenMMLab in vision, Shanghai AI Lab / Qwen in multimodal each forming their own systems.

### 8.2 Gaps and Risks

| Dimension | Current State | Suggested Focus |
|-----------|---------------|-----------------|
| **Tactile data** | High collection cost, low standardization, still reliant on simulation to supplement data | Simulation platforms like TipSim + unified representation are the way forward |
| **Industrial acoustics benchmarks** | Mainstream datasets (MIMII / DCASE) dominated by foreign sources | China needs its own industrial voiceprint benchmarks and cross-device generalization evaluation |
| **Long-tail / open-environment robustness** | Still strongly affected by lighting, noise, domain shift | Multimodal fusion and domain generalization are key |
| **Interpretability and safety** | Hard to attribute perception errors (high-risk scenarios such as medical / autonomous driving) | Need transparent interpretable models and liability definition |
| **License compliance** | Some strong models' weights are non-commercial | Verify licenses item by item before commercial use; note voice / data authorization |

---

## Appendix

### A. Quick-Reference Project List (by Direction)

**Sensor AI**
- LingBot-Touch (Ant LingBot, tactile foundation model, Apache-2.0)
- TipSim (Huawiko, tactile simulation, GitHub `tiplab/TipSim`)
- MMDetection3D / OpenPCDet (OpenMMLab, 3D detection)
- Paddle3D (Baidu, full 3D perception pipeline)
- lidar-slam-detection (Tongji team, SLAM + detection + tracking)
- BEVFormer / BEVFusion (BEV perception)
- IMPACT / Qwen3-ASR-Industrial (industrial acoustics)

**Video AI**
- MMAction2 / MMTracking / MMDetection (OpenMMLab)
- PaddleVideo (Baidu)
- ByteTrack (ByteDance)
- HunyuanVideo / HunyuanVideo 1.5 / HunyuanVideo-Foley (Tencent Hunyuan)
- Wan2.1 (Alibaba), CogVideoX (Zhipu), Open-Sora (Colossal-AI)

**Sound AI**
- FunASR / Paraformer / SenseVoice / Fun-ASR-Nano (Alibaba)
- FireRedASR / FireRedTTS / FireRedChat (Xiaohongshu)
- Qwen3-ASR / Qwen2.5-Omni (Alibaba)
- CosyVoice (Alibaba), GPT-SoVITS, Fish Speech, F5-TTS, ChatTTS, IndexTTS
- PaddleSpeech (Baidu), WeNet (Mobvoi + NWPU)
- sherpa-onnx (on-device deployment runtime)

**Fusion / Embodied**
- InternVL3.5, Intern-Robotics (Shanghai AI Lab)
- LingBot-VLA (Ant LingBot)
- Kimi-Audio (Moonshot AI), Step-Audio 2 (StepFun)

### B. Main Sources

[^1]: Sihan Industry Research Institute — definition, classification, market size, and trends of perception intelligence: http://chinasihan.com/news/cykj/22159.html
[^2]: CTai Labs / ISMR — four-stage evolution framework from Perception AI to Physical AI: https://ctailabs.ai/?p=12965 ; https://ismr.net/ai-perception-physical-reality
[^3]: Enterprise Nova — types and principles of AI Perception: https://enterprisenova.com/what-is-ai-perception-and-how-it-work
[^4]: Jiqizhixin — Xiaohongshu FireRed series open-source layout (FireRedTTS/ASR/Chat): http://mp.weixin.qq.com/s?__biz=MzA3MzI4MjgzMw==&mid=2650991374&idx=1&sn=879d00ba6ba313975f2627778d759d4b
[^5]: Tencent News / Shanghai Securities News — Alibaba Qwen2.5-Omni omni-modal open source: https://news.qq.com/rain/a/20250327A02DP300 ; https://www.cnstock.com/commonDetail/382802
[^6]: Ant LingBot LingBot-Touch open source (including GitHub / technical report): https://www.infoq.cn/article/W7VPoeNBPX4y2u72WAGM ; https://github.com/Robbyant/lingbot-touch
[^7]: Huawiko TipSim tactile simulation platform open source: https://m.chinadevelopment.com.cn/checklist.php?s=index/detail/id/2024398
[^8]: Survey of 3D point-cloud deep learning algorithms (MMDetection3D / Paddle3D / OpenPCDet, etc.): https://blog.csdn.net/qin_student/article/details/148524763
[^9]: lidar-slam-detection project analysis: https://www.pomoai.com/hubs/w111liang222/lidar-slam-detection
[^10]: BEV perception principles and industry ecosystem (BEVFormer / BEVFusion): https://myhqh.blog.csdn.net/article/details/161729886
[^11]: Recommended open-source industrial voiceprint models for local edge computing (IMPACT / Qwen3-ASR-Industrial, etc., technical blog, needs verification): https://www.cnblogs.com/dylancao/p/19629143
[^12]: MIMII-DG / DCASE2025-Task2 abnormal sound datasets: https://www.selectdataset.com/dataset/90d9b7343a950924ecd3122c32eb7aa3/mimii-dg
[^13]: MMTracking / OpenMMLab video perception tool ecosystem: https://onegen.ai/project/revolutionizing-video-object-detection-with-sequence-level-semantics-aggregation-selsa ; https://ai.hqyman.cn/post/9311.html
[^14]: PaddleVideo new version and PP-TSMv2 performance: https://cloud.baidu.com/article/5323752 ; https://xuyubao.blog.csdn.net/article/details/152039653
[^15]: Tencent Hunyuan HunyuanVideo / 1.5 / Foley open source: https://baike.baidu.com/item/HunyuanVideo ; https://www.cnstock.com/commonDetail/591479
[^16]: Overview of open-source text-to-video models (Wan2.1 / CogVideoX / Open-Sora, etc.): https://developer.cloud.tencent.com/article/2473768
[^17]: Three-layer ecosystem of 6 Chinese speech-recognition models + 2 tools: https://blog.csdn.net/CooVally_AI/article/details/159282240
[^18]: Comparison of FunASR / WeNet / PaddleSpeech: https://ai-bio.cn/funasr/
[^19]: FireRedASR Chinese SOTA and CER comparison table: https://thepaper.cn/newsDetail_forward_30116639
[^20]: FireRedASR official README (metrics and usage): https://github.com/FireRedTeam/FireRedASR
[^21]: PaddleSpeech ASR + TTS integrated framework: https://adg.csdn.net/6952436a5b9f5f31781b47e9.html
[^22]: Open-source Chinese TTS selection guide (CosyVoice / FishSpeech / GPT-SoVITS, etc.): https://m.toutiao.com/article/7654145155607298586 ; https://alishangtian.com/ainews/article/user-4685b75e5b
[^23]: Shanghai AI Lab InternVL3.5 open source: https://baike.baidu.com/item/书生·万象3.5/66727181 ; https://hub.baai.ac.cn/view/48595
[^24]: HIT Zhang Weinan team's embodied intelligence survey (three modules of embodied perception): http://mp.weixin.qq.com/s?__biz=MzIxMjAzNDY5Mg==&mid=2650815210&idx=1&sn=2842a364a940b75520e80a235c7fc852
[^25]: Shanghai AI Lab Intern-Robotics embodied intelligence open-source week: https://www.shlab.org.cn/news/5444213

---

> **Disclaimer**: this report is compiled from public materials available before 2026-11-20. Open-source projects iterate quickly; specific performance metrics, licenses, and repository addresses should follow each project's latest official release. Some data are quoted from technical blogs / media reports and are marked for reference only — please double-check before commercial or academic citation.
