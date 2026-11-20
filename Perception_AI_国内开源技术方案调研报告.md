# Perception AI 国内优秀开源技术方案调研报告

> 调研日期：2026-11-20
> 调研范围：感知智能（Perception AI）概念框架，以及 Sensor AI / Video AI / Sound AI 三大方向的国内开源方案，附全模态大模型与具身感知等融合方向。
> 说明：本报告聚焦"中国机构/团队主导、可公开获取（GitHub / Hugging Face / ModelScope / Gitee）"的方案。性能数据来自公开论文、官方文档与权威媒体报道，标注口径与版本，部分指标随版本迭代变化，引用时请核对最新版本。

---

## 目录

1. [概念框架：什么是 Perception AI](#一概念框架)
2. [国内开源生态全景](#二国内开源生态全景)
3. [Sensor AI：传感器感知](#三sensor-ai传感器感知)
4. [Video AI：视频感知](#四video-ai视频感知)
5. [Sound AI：声音/语音感知](#五sound-ai声音语音感知)
6. [融合方向：全模态大模型与具身感知](#六融合方向全模态大模型与具身感知)
7. [横向对比与选型建议](#七横向对比与选型建议)
8. [趋势判断与差距分析](#八趋势判断与差距分析)
9. [附录：项目速查清单与来源](#附录)

---

## 一、概念框架

### 1.1 定义与定位

**感知智能（Perception AI）** 是指利用人工智能（尤其是机器学习、计算机视觉、语音处理等技术），使机器通过传感器"感知—理解—响应"外部环境的能力，模拟人类的视觉、听觉、触觉、嗅觉、味觉等感官[^1]。

一个被广泛引用的演进框架来自 NVIDIA CEO 黄仁勋对 AI 阶段的划分，清晰定位了 Perception AI 所处的历史位置[^2]：

| 阶段 | 内涵 | 代表 |
|------|------|------|
| **① Perception AI（感知AI）** | 识别和解释世界信息：图像、声音、振动、温度等传感器数据 | AlexNet(2012)、缺陷检测、车道线识别 |
| **② Generative AI（生成AI）** | 理解语义并生成新内容：文本、图像、代码 | GPT、文生图/视频（仍在软件世界） |
| **③ Agentic AI（智能体AI）** | 在软件环境中推理、规划、自主执行多步任务 | 工具调用、工作流编排 |
| **④ Physical AI（物理AI）** | 结合模型+传感器+控制+硬件，在物理世界感知、推理、行动 | 机器人、自动驾驶、VLA 模型 |

> Perception AI 是所有后续阶段的"感官底座"——没有高质量感知，上层的生成、决策、执行都无从谈起（"感知质量决定下游一切的上限"）[^2]。

### 1.2 感知模态分类

按模拟人类感官的标准分类[^1][^3]：

| 模态 | 英文 | 典型信号 | 本报告对应章节 |
|------|------|---------|--------------|
| 视觉感知 | Visual | 图像/视频/多光谱 | 第四章 Video AI |
| 听觉感知 | Auditory | 语音/音频/声纹/异常声 | 第五章 Sound AI |
| 触觉感知 | Tactile | 压力/纹理/力/滑移 | 第三章 Sensor AI |
| 环境/传感器感知 | Sensor | IMU/点云/雷达/振动/温度 | 第三章 Sensor AI |
| 文本/语言感知 | Textual | 文字/语义 | （属 NLP，不在本报告重点） |

### 1.3 关键技术趋势

行业研究指出感知智能有三大趋势[^1]：

1. **感知范围超人类化**：多光谱成像把紫外、红外、太赫兹纳入采集，AI 解析超人类感官数据（如红外热成像识别机械早期故障）。
2. **多模态融合**：单一模态难以应对复杂场景，融合红外、可见光、声谱、电信号构建跨模态模型（如工业监测同时分析光谱图像与声学特征）。
3. **端侧闭环**：AI 算法烧录至端侧模组，从"采集—云端分析"转向"本地实时闭环"，延迟降低可达 90%。

市场规模佐证：全球感知智能市场从 2020 年约 2001 亿元增至 2024 年约 2929 亿元（CAGR 10%），预计 2029 年达 5891 亿元（CAGR 15%），其中视觉感知占 2024 年市场约 62%[^1]。

---

## 二、国内开源生态全景

国内 Perception AI 开源力量可分三类主体，形成了从底层框架到上层模型的完整栈：

### 2.1 底层深度学习框架（基座）

| 框架 | 机构 | 特点 | 感知相关套件 |
|------|------|------|-------------|
| **飞桨 PaddlePaddle** | 百度 | 国产全功能框架，工业级验证，适配国产芯片 | PaddleCV / PaddleVideo / PaddleSpeech / Paddle3D / PaddleOCR |
| **昇思 MindSpore** | 华为 | 全栈方案，配合昇腾芯片 | 端边云协同，自动驾驶/视觉 |
| **MegEngine** | 旷视 | 训练推理一体 | 检测/分类工业落地 |

### 2.2 代表性开源机构与"全家桶"

| 机构 | 开源品牌/体系 | 覆盖方向 |
|------|-------------|---------|
| **上海人工智能实验室** | 书生（Intern）系列 | InternVL 多模态、Intern-Robotics 具身、OpenCompass 评测、LMDeploy 部署 |
| **OpenMMLab**（港中文 MMLab 发起） | MM 系列 30+ 算法库 | 检测/分割/姿态/视频/3D/部署 |
| **阿里巴巴通义 / 达摩院** | Qwen、FunASR、CosyVoice | 全模态大模型、ASR、TTS |
| **小红书 FireRed 团队** | FireRedASR / FireRedTTS / FireRedChat | 语音识别/合成/全双工对话 |
| **腾讯混元** | HunyuanVideo 等 | 视频生成、全模态 |
| **百度飞桨** | Paddle 系列 | CV/Speech/Video/3D 全流程 |
| **蚂蚁灵波** | LingBot 系列 | 触觉基础模型、VLA |

> 一个显著现象：开源已成国内厂商长期战略。仅 2025 年 7–8 月，国内厂商在 Hugging Face 中文社区连续开源 33 款、31 款各类型大模型；Qwen 衍生模型数量突破 10 万，居全球开源模型阵营第一[^4][^5]。

---

## 三、Sensor AI：传感器感知

Sensor AI 指对非视觉图像类传感器信号（触觉、点云/LiDAR、IMU、振动、工业声学、多传感器融合）的智能分析。这是国内近年突破最集中、也最具"具身智能"色彩的方向。

### 3.1 触觉感知（Tactile AI）—— 2025 新爆发点

| 项目 | 机构 | 核心能力 | 许可/地址 |
|------|------|---------|----------|
| **LingBot-Touch** | 蚂蚁灵波 | 通用触觉基础模型，近 5000 小时真实触觉数据预训练，统一**磁触觉/视触觉/压阻式**三类信号为同一表征；接入 LingBot-VLA；26 项真机任务成功率提升 14–26 个百分点 | Apache-2.0；GitHub `Robbyant/lingbot-touch`[^6] |
| **TipSim** | 华威科 | 行业首个多模态触觉仿真平台，基于 MuJoCo，输出接触位置/受力分布阵列数据并接入控制回路，破解"触觉数据荒" | 开源核心接口；GitHub `tiplab/TipSim`[^7] |

> 意义：LingBot-Touch 把具身感知从"视觉"延伸到"指尖"，解决精细操作中视觉被遮挡、无法判断受力/滑移的问题；TipSim 则让触觉数据可规模化仿真采集，压缩 sim-to-real 落差。

### 3.2 LiDAR / 点云 / 3D 感知

| 项目 | 机构 | 核心能力 | 说明 |
|------|------|---------|------|
| **MMDetection3D** | OpenMMLab | 3D 检测平台，支持 PointPillars/SECOND/BEVFusion/Point Transformer 等，覆盖 KITTI/nuScenes | Apache-2.0，BEVFormer 官方实现[^8] |
| **Paddle3D** | 百度 | 飞桨 3D 感知套件，覆盖单目/点云、检测/分割，适配国产芯片，全流程部署 | 中文文档友好，TensorRT/OpenVINO 加速[^8] |
| **OpenPCDet** | OpenMMLab | 专注 LiDAR 3D 检测 | 点云检测主流库[^8] |
| **lidar-slam-detection (LSD)** | 同济大学校友团队 | 一站式：FAST-LIO+G2O 的 SLAM + PointPillars/CenterPoint 检测 + AB3DMOT 追踪；CenterPoint 在 Waymo 验证集 AMOTA 77.55% | GitHub 750+ star，多传感器融合落地[^9] |
| **BEVFormer / BEVFusion** | 清华+商汤等 / MIT+清华 | 多相机 BEV 感知、多传感器融合；BEVFusion BEV pooling 延迟降低 40 倍+ | 国内 BEV 量产潮的学术源头[^10] |

### 3.3 工业声学 / 异常声音检测（Sensor+Sound 交叉）

| 项目/数据集 | 机构 | 核心能力 | 说明 |
|------------|------|---------|------|
| **IMPACT (Industrial Acoustic Transformer)** | （2025 开源） | 全球首个工业声学专用预训练大模型，DINOS 数据集（74149 条、1093 小时，覆盖 CNC/机床/电机/泵），强噪鲁棒、跨设备泛化 | MIT 可商用；~300M–1.2B[^11] |
| **Qwen3-ASR-Industrial** | 阿里 | 工业声纹专用 ASR+异常检测基座，三级降噪适配 100dB+ 车间，端边云架构（0.6B 边缘/1.7B 服务器），声纹+振动+PLC 融合诊断 | Apache-2.0[^11] |
| **MIMII / MIMII-DG / DCASE2025-Task2** | 国际数据集（国内广泛使用） | 工业机器正常/异常声音基准，无监督异常检测与域泛化 | 训练/评测标准数据集[^12] |

> ⚠️ 提示：工业声纹领域部分"2025 大模型"信息来源为技术博客，落地前建议核对 GitHub/HF 官方仓库与许可证。

---

## 四、Video AI：视频感知

Video AI 分为"视频理解/分析"与"视频生成"两条主线，国内在这两条线上均有世界级开源方案。

### 4.1 视频理解与分析

| 项目 | 机构 | 覆盖任务 | 特点 |
|------|------|---------|------|
| **OpenMMLab 视频族** | 港中文 MMLab | MMAction2（动作识别/时空检测）、MMTracking（VID 视频目标检测/MOT 多目标跟踪/SOT 单目标跟踪/VIS 视频实例分割）、MMDetection | 模块化、Model Zoo 丰富、Apache-2.0；MMTracking 3.9k+ star，统一四任务[^13] |
| **PaddleVideo** | 百度 | 视频分类、时序动作检测、行为识别 | PP-TSMv2 在 Kinetics-400 达 75.16%（22M，CPU 456ms/10s）；2025 新增 TokenShift、YOWO 时空检测、轻量化模型[^14] |
| **ByteTrack** | 字节跳动 | 多目标跟踪 | MOT 领域广泛使用的高效跟踪器[^13] |
| **InternVideo 系列** | 上海 AI Lab | 视频基础模型/视频理解 | 书生体系多模态视频骨干 |

### 4.2 视频生成（世界模型/内容生成）

| 模型 | 机构 | 是否开源 | 参数/规格 | 亮点 |
|------|------|---------|----------|------|
| **HunyuanVideo** | 腾讯混元 | ✅ | 13B | DiT 架构，双流到单流混合，3D-VAE，原生镜头转换[^15] |
| **HunyuanVideo 1.5** | 腾讯混元 | ✅ | 8.3B | 轻量，14G 显存可跑，SSTA 稀疏注意力，5–10s 480p/720p→超分 1080p[^15] |
| **Wan2.1（通义万相）** | 阿里 | ✅ | 14B/1.3B | 720p，开源[^16] |
| **CogVideoX 1.5** | 智谱 | ✅ | 5B | 1360×768，5/10s[^16] |
| **Open-Sora / Open-Sora-v2** | 潞晨科技 | ✅ | 11B | 开源复现 Sora 路线[^16] |
| **HunyuanVideo-Foley** | 腾讯混元 | ✅ | — | 端到端视频音效生成，视频与音效同步[^15] |
| 可灵 Kling / Vidu / 即梦 / MiniMax 海螺 | 快手/生数/字节/MiniMax | ❌（闭源商用） | — | 国内第一梯队商用视频生成，非开源[^16] |

> 国内开源视频生成已形成"腾讯/阿里/智谱/潞晨多强开源 + 快手/字节闭源商用"的格局，开源模型在参数效率（如 8.3B 达旗舰体验）上进展显著。

---

## 五、Sound AI：声音/语音感知

这是国内开源最活跃、工业落地最成熟的感知方向，已形成**模型层—工具包层—部署运行时层**的完整分层生态[^17]。

### 5.1 三层生态结构

```
模型层（做识别/合成）：Paraformer、SenseVoice、Fun-ASR-Nano、FireRedASR、Qwen3-ASR、CosyVoice、Fish-Speech...
工具包层（管理/组合模型）：FunASR、PaddleSpeech、WeNet...
部署运行时层（把模型跑在设备上）：sherpa-onnx（端侧，10.8k star）、FunASR Docker、vLLM...
```

> 实践中常组合使用，例如"SenseVoice 模型 + FunASR 接口调用 + sherpa-onnx 部署到手机/鸿蒙"[^17]。

### 5.2 语音识别（ASR）

| 模型 | 机构 | 时间 | 定位与指标 |
|------|------|------|-----------|
| **Paraformer** | 阿里达摩院 | 2022 | FunASR 最早自研模型，集成最成熟，有流式版，6 万小时数据训练[^17][^18] |
| **SenseVoice-Small / SenseVoice-L** | 阿里通义 | 2024/07 | 不只识别内容，还能识别**情绪与声音事件**（掌声/笑声）；处理速度最快；CER 4.47（L）[^17][^19] |
| **Fun-ASR-Nano** | 阿里通义 | 2025/12 | 数千万小时数据，流式转录，7 方言/26 口音，车间准确率 93%+，首字延迟<160ms[^17][^11] |
| **FireRedASR** | 小红书 | 2025/01（v2:2026/02） | **中文识别精度公开基准第一**：LLM 版 CER 3.05%、AED 版 3.18%，优于 Seed-ASR(3.33)/Whisper(9.86)/Paraformer(4.56)；支持歌词识别、方言、英语；工业级可商用[^19][^20] |
| **Qwen3-ASR** | 阿里云 Qwen | 2026/01 | 22 种中文方言，提供独立安装包 qwen-asr[^17] |
| **WeNet** | 出门问问+西工大等 | — | "production ready"识别引擎，工业界广泛使用[^18] |
| **PaddleSpeech** | 百度 | — | ASR+TTS+翻译一体化，中文 ASR 错误率 6.8%，支持国产芯片，5 行代码上手[^21] |

### 5.3 语音合成（TTS）

| 模型 | 机构 | Star/许可 | 定位 |
|------|------|----------|------|
| **CosyVoice 3.0** | 阿里通义 | 23.9k / Apache-2.0 | 中文 TTS 工业部署事实标准；9 语言+18 方言；双向流式 150ms；零样本相似度 90%+；情感指令控制[^22] |
| **Fish Speech S2 Pro** | Fish Audio | 33k / Research License（商用需授权） | Dual-AR 架构，1000 万小时/80+ 语言，情绪标签，44.1kHz 高保真，中文 WER 0.54%[^22] |
| **GPT-SoVITS** | RVC-Boss | 59k / MIT | 少样本神器：5 秒克隆、1 分钟微调，中英日韩粤；整体可商用[^22] |
| **F5-TTS** | SWivid | 15.3k / 代码 MIT、权重 CC-BY-NC | 纯流匹配+DiT，pip 即装，Apple Silicon 支持[^22] |
| **ChatTTS** | 2noise | 非商用限制 | 对话场景，主动加入"嗯/那个"口语特征，拟真度高[^22] |
| **IndexTTS** | B站 | 24.3k / 自定义许可 | 商用需联系官方[^22] |
| **FireRedTTS / FireRedTTS-2** | 小红书 | 开源可商用 | 零样本音色克隆；TTS-2 主打多轮对话/播客生成，4 说话人，开源 SOTA[^4] |

### 5.4 全双工语音交互与音频理解

| 项目 | 机构 | 亮点 |
|------|------|------|
| **FireRedChat** | 小红书 | 业内首个完全开源的全双工语音对话系统，端到端接近工业级；一站式开放 VAD/ASR/TTS/上下文 TTS/音频 LLM/Dify 支持，支持私有化部署[^4] |
| **SenseVoice** | 阿里 | 音频事件检测+情绪识别（"听懂弦外之音"）[^17] |

---

## 六、融合方向：全模态大模型与具身感知

单一感知模态正向"统一多模态感知"演进，这是 Perception AI 的最高形态。

### 6.1 端到端全模态大模型（文本/图像/音频/视频统一）

| 模型 | 机构 | 发布时间 | 模态与亮点 |
|------|------|---------|-----------|
| **Qwen2.5-Omni-7B** | 阿里通义 | 2025-03-27 | 首个端到端全模态大模型，文本/图像/音频/视频输入，实时文本+语音输出；Thinker-Talker 双核架构、TMRoPE；OmniBench 刷新纪录，超越 Gemini-1.5-Pro；7B 手机可部署，开源可商用[^5] |
| **InternVL3.5** | 上海 AI Lab | 2025-09 | 通用多模态大模型，9 种尺寸(1B–241B)，旗舰 MMMU 77.7 开源最高；多模态通用感知超越 GPT-5；GUI 智能体、具身空间感知突出；全系列下载超 2300 万次[^23] |
| **Kimi-Audio** | 月之暗面 | 2025 | 开源音频基座模型[^4] |
| **Step-Audio 2** | 阶跃星辰 | 2025 | 端到端多模态语音大模型，1.8B，实时流<150ms[^4][^11] |

### 6.2 具身感知（Embodied Perception）

哈工大张伟男团队综述将具身智能划分为"具身感知—具身决策—具身执行"三模块，其中**具身感知**关注物体感知（几何/铰接/物理属性）与场景感知（尺度地图/拓扑图），并提出"感知用于行为"与"行为用于感知"双向增强[^24]。

| 项目 | 机构 | 内容 |
|------|------|------|
| **Intern-Robotics 全栈引擎** | 上海 AI Lab | 导航大模型 InternVLA·N1、操作大模型 InternVLA·M1/A1、奖励模型 VLAC、人形控制 InternHumanoid；数据集 InternScenes(4万+室内场景)、OmniWorld(300M+视频帧)；模型/数据集下载超 110 万次，多次登顶 HF Robotics 趋势榜[^25] |
| **LingBot-VLA + LingBot-Touch** | 蚂蚁灵波 | 视觉-语言-动作模型 + 触觉基础模型，感知延伸到"指尖"[^6] |

---

## 七、横向对比与选型建议

### 7.1 各方向国内开源成熟度评估

| 方向 | 开源成熟度 | 国际竞争力 | 代表项目 | 备注 |
|------|-----------|-----------|---------|------|
| **Sound AI（ASR/TTS）** | ⭐⭐⭐⭐⭐ | 世界领先 | FireRedASR、CosyVoice、SenseVoice、Qwen2.5-Omni | 中文场景精度全球 SOTA，生态分层完整 |
| **Video AI（理解/检测）** | ⭐⭐⭐⭐⭐ | 世界领先 | OpenMMLab、PaddleVideo、ByteTrack | 工具链体系化，工程落地成熟 |
| **Video AI（生成）** | ⭐⭐⭐⭐☆ | 第一梯队 | HunyuanVideo、Wan2.1、CogVideoX | 开源活跃，参数效率高；商用闭源亦强 |
| **全模态大模型** | ⭐⭐⭐⭐⭐ | 世界领先 | Qwen2.5-Omni、InternVL3.5 | 7B 端侧可部署，超越同级闭源 |
| **Sensor AI（3D/LiDAR）** | ⭐⭐⭐⭐☆ | 领先 | MMDetection3D、Paddle3D、BEVFusion | 自动驾驶量产驱动，工具链完善 |
| **Sensor AI（触觉）** | ⭐⭐⭐☆☆ | 快速追赶 | LingBot-Touch、TipSim | 2025 新爆发，数据集仍是瓶颈 |
| **Sensor AI（工业声学）** | ⭐⭐⭐☆☆ | 追赶中 | IMPACT、Qwen3-ASR-Industrial | 数据集/基准仍以国际为主 |

### 7.2 典型场景选型建议

| 需求场景 | 推荐方案 | 理由 |
|---------|---------|------|
| 中文语音转写（高精度） | FireRedASR-LLM / Fun-ASR-Nano | CER 公开基准第一 / 工业级流式+方言 |
| 语音情绪+事件理解 | SenseVoice | 情绪/掌声/笑声识别，速度快 |
| 中文语音合成（实时交互） | CosyVoice 3.0 | 150ms 流式、方言、Apache-2.0 可商用 |
| 少样本音色克隆 | GPT-SoVITS（MIT，可商用） | 5 秒克隆、生态成熟 |
| 端侧离线语音 | sherpa-onnx + SenseVoice/Paraformer | 手机/树莓派/鸿蒙离线运行 |
| 视频检测/跟踪/动作 | OpenMMLab（MMDetection+MMTracking+MMAction2） | 一体化工具链、Model Zoo |
| 视频动作识别（轻量部署） | PaddleVideo PP-TSMv2 | 22M 小模型、国产芯片适配 |
| 自动驾驶/机器人 3D 感知 | MMDetection3D / Paddle3D / BEVFusion | BEV 多传感器融合、全流程部署 |
| 视觉+声音+文本统一理解 | Qwen2.5-Omni-7B / InternVL3.5 | 端侧可部署、多模态 SOTA |
| 机器人触觉精细操作 | LingBot-Touch + TipSim | 触觉表征统一、仿真补数据 |
| 文本/图/音/视统一生成+对话 | FireRedChat / Qwen2.5-Omni | 全双工、全链路开源、可私有化 |

### 7.3 选型时的三个关键判断

1. **模型 vs 工具包 vs 运行时**：三者不在同一层，需组合使用（如 SenseVoice+FunASR+sherpa-onnx）。先明确自己缺哪一层。
2. **许可证**：商用前务必核对。可商用友好：Apache-2.0（CosyVoice、Qwen2.5-Omni、InternVL、LingBot-Touch）、MIT（GPT-SoVITS、IMPACT）；需授权/非商用：Fish Speech S2 Pro、ChatTTS、F5-TTS 权重、IndexTTS。
3. **数据源与口径**：精度指标（CER/mAP/NDS）依赖测试集与版本，跨模型对比需统一基准（如中文 ASR 用 AISHELL/WenetSpeech）。

---

## 八、趋势判断与差距分析

### 8.1 趋势

1. **从单模态到全模态统一**：Qwen2.5-Omni、InternVL3.5 代表"一个模型感知所有模态"成为主流，端侧 7B 可部署让感知 AI 进入手机/机器人。
2. **从"识别"到"理解+交互"**：SenseVoice 的情绪/事件检测、FireRedChat 的全双工对话，感知不再只输出标签，而是理解语境、带情感反馈。
3. **从云端到端侧闭环**：sherpa-onnx、Qwen2.5-Omni-7B、工业声纹边缘版（0.6B）推动"本地实时感知—决策"，满足低延迟与隐私需求。
4. **从视觉到多感官具身**：触觉（LingBot-Touch）、3D 空间（InternVLA）、振动/声学等多传感器融合，感知 AI 成为具身智能/Physical AI 的入口。
5. **开源成为战略**：国内厂商以开源构建生态话语权，音频领域小红书、视觉领域 OpenMMLab、多模态领域上海 AI Lab/Qwen 各成体系。

### 8.2 差距与风险

| 维度 | 现状 | 建议关注 |
|------|------|---------|
| **触觉数据** | 采集成本高、标准化低，仍依赖仿真补数据 | TipSim 等仿真平台 + 统一表征是破局方向 |
| **工业声学基准** | 主流数据集（MIMII/DCASE）为国外主导 | 国内需建设自有工业声纹基准与跨设备泛化评测 |
| **长尾/开放环境鲁棒性** | 受光照、噪声、域漂移影响仍大 | 多模态融合与域泛化是关键 |
| **可解释性与安全** | 感知错误归因困难（医疗/自动驾驶等高风险场景） | 需透明可解释模型与责任界定 |
| **许可证合规** | 部分强模型权重非商用 | 商用前逐一核对许可，注意音色/数据授权 |

---

## 附录

### A. 项目速查清单（按方向）

**Sensor AI**
- LingBot-Touch（蚂蚁灵波，触觉基础模型，Apache-2.0）
- TipSim（华威科，触觉仿真，GitHub `tiplab/TipSim`）
- MMDetection3D / OpenPCDet（OpenMMLab，3D 检测）
- Paddle3D（百度，3D 感知全流程）
- lidar-slam-detection（同济团队，SLAM+检测+追踪）
- BEVFormer / BEVFusion（BEV 感知）
- IMPACT / Qwen3-ASR-Industrial（工业声学）

**Video AI**
- MMAction2 / MMTracking / MMDetection（OpenMMLab）
- PaddleVideo（百度）
- ByteTrack（字节）
- HunyuanVideo / HunyuanVideo 1.5 / HunyuanVideo-Foley（腾讯混元）
- Wan2.1（阿里）、CogVideoX（智谱）、Open-Sora（潞晨）

**Sound AI**
- FunASR / Paraformer / SenseVoice / Fun-ASR-Nano（阿里）
- FireRedASR / FireRedTTS / FireRedChat（小红书）
- Qwen3-ASR / Qwen2.5-Omni（阿里）
- CosyVoice（阿里）、GPT-SoVITS、Fish Speech、F5-TTS、ChatTTS、IndexTTS
- PaddleSpeech（百度）、WeNet（出门问问+西工大）
- sherpa-onnx（端侧部署运行时）

**融合/具身**
- InternVL3.5、Intern-Robotics（上海 AI Lab）
- LingBot-VLA（蚂蚁灵波）
- Kimi-Audio（月之暗面）、Step-Audio 2（阶跃星辰）

### B. 主要来源

[^1]: 思瀚产业研究院 — 感知智能定义、分类、市场规模与趋势：http://chinasihan.com/news/cykj/22159.html
[^2]: CTai Labs / ISMR — Perception AI → Physical AI 四阶段演进框架：https://ctailabs.ai/?p=12965 ；https://ismr.net/ai-perception-physical-reality
[^3]: Enterprise Nova — AI Perception 类型与原理：https://enterprisenova.com/what-is-ai-perception-and-how-it-work
[^4]: 机器之心 — 小红书 FireRed 系列开源布局（FireRedTTS/ASR/Chat）：http://mp.weixin.qq.com/s?__biz=MzA3MzI4MjgzMw==&mid=2650991374&idx=1&sn=879d00ba6ba313975f2627778d759d4b
[^5]: 腾讯新闻/上证报 — 阿里 Qwen2.5-Omni 全模态开源：https://news.qq.com/rain/a/20250327A02DP300 ；https://www.cnstock.com/commonDetail/382802
[^6]: 蚂蚁灵波 LingBot-Touch 开源（含 GitHub/技术报告）：https://www.infoq.cn/article/W7VPoeNBPX4y2u72WAGM ；https://github.com/Robbyant/lingbot-touch
[^7]: 华威科 TipSim 触觉仿真平台开源：https://m.chinadevelopment.com.cn/checklist.php?s=index/detail/id/2024398
[^8]: 三维点云深度学习算法综述（MMDetection3D/Paddle3D/OpenPCDet 等）：https://blog.csdn.net/qin_student/article/details/148524763
[^9]: lidar-slam-detection 项目解析：https://www.pomoai.com/hubs/w111liang222/lidar-slam-detection
[^10]: BEV 感知原理与产业生态（BEVFormer/BEVFusion）：https://myhqh.blog.csdn.net/article/details/161729886
[^11]: 本地边缘计算开源工业声纹模型推荐（IMPACT/Qwen3-ASR-Industrial 等，技术博客，需核对官方）：https://www.cnblogs.com/dylancao/p/19629143
[^12]: MIMII-DG / DCASE2025-Task2 异常声音数据集：https://www.selectdataset.com/dataset/90d9b7343a950924ecd3122c32eb7aa3/mimii-dg
[^13]: MMTracking / OpenMMLab 视频感知工具生态：https://onegen.ai/project/revolutionizing-video-object-detection-with-sequence-level-semantics-aggregation-selsa ；https://ai.hqyman.cn/post/9311.html
[^14]: PaddleVideo 新版本与 PP-TSMv2 性能：https://cloud.baidu.com/article/5323752 ；https://xuyubao.blog.csdn.net/article/details/152039653
[^15]: 腾讯混元 HunyuanVideo / 1.5 / Foley 开源：https://baike.baidu.com/item/HunyuanVideo ；https://www.cnstock.com/commonDetail/591479
[^16]: 开源文生视频模型一览（Wan2.1/CogVideoX/Open-Sora 等）：https://developer.cloud.tencent.com/article/2473768
[^17]: 中文语音识别 6 模型+2 工具三层生态梳理：https://blog.csdn.net/CooVally_AI/article/details/159282240
[^18]: FunASR / WeNet / PaddleSpeech 对比：https://ai-bio.cn/funasr/
[^19]: FireRedASR 中文 SOTA 与 CER 对比表：https://thepaper.cn/newsDetail_forward_30116639
[^20]: FireRedASR 官方 README（指标与用法）：https://github.com/FireRedTeam/FireRedASR
[^21]: PaddleSpeech ASR+TTS 一体化框架：https://adg.csdn.net/6952436a5b9f5f31781b47e9.html
[^22]: 开源中文 TTS 选型指南（CosyVoice/FishSpeech/GPT-SoVITS 等）：https://m.toutiao.com/article/7654145155607298586 ；https://alishangtian.com/ainews/article/user-4685b75e5b
[^23]: 上海 AI Lab InternVL3.5 开源：https://baike.baidu.com/item/书生·万象3.5/66727181 ；https://hub.baai.ac.cn/view/48595
[^24]: 哈工大张伟男团队具身智能综述（具身感知三模块）：http://mp.weixin.qq.com/s?__biz=MzIxMjAzNDY5Mg==&mid=2650815210&idx=1&sn=2842a364a940b75520e80a235c7fc852
[^25]: 上海 AI Lab Intern-Robotics 具身智能开源周：https://www.shlab.org.cn/news/5444213

---

> **免责声明**：本报告基于 2026-11-20 前的公开资料整理，开源项目迭代快，具体性能指标、许可证、仓库地址请以各项目官方最新发布为准。部分数据引自技术博客/媒体转述，标注供参考，商用与学术引用前请二次核验。
