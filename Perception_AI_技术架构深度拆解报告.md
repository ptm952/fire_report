# Perception AI 精选项目 · 技术架构深度拆解（中英对照）
# Curated Perception AI Projects — In-Depth Technical Architecture Deconstruction (Bilingual)

> 调研日期 / Research date：2026-11-21
> 定位 / Positioning：本报告是《精选开源项目中英对照调研报告》（速查版）的**深度补充**——不再只列简介/开源状态/地址/部署，而是逐项目拆解**架构范式、模块数据流、核心创新、参数规模与架构约束**，并配文字架构图。
> *This report is a **deep supplement** to the quick-reference survey: instead of merely listing intros/licenses/links/deployment, it deconstructs each project's **architecture paradigm, module-level data flow, key innovations, parameter scale and architectural constraints**, with text-based architecture diagrams.*
> 可靠性 / Reliability：全部架构细节于 2026-11-21 直接读官方 GitHub README、arXiv 论文全文与官方技术博客核实；凡官方未披露者明确标注，未采信二手博客。核实中纠正了多处流传的错误术语，见文末「术语校正表」。
> *All details verified on 2026-11-21 against official GitHub READMEs, full arXiv papers and official blogs; anything not officially disclosed is flagged. Several widely circulated wrong terms were corrected — see the "Terminology Corrections" table at the end.*

---

## 阅读指南 / How to read

每个项目统一用六个维度拆解，体例一致：
*Each project is deconstructed along six consistent dimensions:*

| 维度 / Dimension | 内容 / Content |
|---|---|
| ① 架构范式 / Paradigm | 模型属于哪一类（DiT / LLM-ASR / 非自回归 / 双核……） |
| ② 架构图 / Diagram | ASCII 文字图，展示模块与数据流 |
| ③ 核心模块与数据流 / Modules & flow | 每个模块职责、信号如何流转 |
| ④ 关键创新 / Innovations | 官方明确主张的创新点 |
| ⑤ 参数与指标 / Scale & metrics | 参数量分层、基准分数 |
| ⑥ 架构约束 / Constraints | 时长/分辨率/采样率等架构决定硬限制 |

---

## 术语校正（先看，避免带错概念读全文）/ Terminology Corrections (read first)

| 常见说法 / Common claim | 官方准确表述 / Official wording |
|---|---|
| Qwen2.5-Omni「Thinker-Talker **双核**」 | 官方只说 "Thinker-Talker architecture"，**未用 dual-core 一词** / Official says only "Thinker-Talker architecture"; no "dual-core" |
| HunyuanVideo「single-stream dual attention」 | 官方无此术语；正确是 **Dual-stream→Single-stream** + **unified Full Attention** + **3D RoPE** / No such term; correct: Dual-to-Single stream + unified Full Attention + 3D RoPE |
| InternVL3.5「ViR = Visual-consistency Reinforcement」 | **ViR = Visual Resolution Router**；视觉一致性叫 **ViCO**；**DvD = Decoupled Vision-Language Deployment** |
| FireRedTTS2「1.5B 参数」 | 官方 README 与论文**均未披露参数量**；1.5B 仅见二手来源 / Params **not disclosed** officially; 1.5B appears only in secondary sources |
| SenseVoice「非自回归多任务」泛指全系 | 仅 **Small（234M，encoder-only）**成立；**Large（1.6B）是自回归 encoder-decoder** |
| TipSim 当作「触觉基础模型」 | 实为 **MuJoCo 触觉仿真插件平台**，无神经网络权重 / It's a MuJoCo tactile **simulation plugin platform**, no NN weights |
| FireRedTTS2 延迟「140ms」 | README 写 140ms@L20，论文写 **<100ms**，两个官方来源口径不同 |

---

## 一、传感器 AI：触觉感知 / 1. Sensor AI: Tactile Perception

### 1.1 LingBot-Touch（蚂蚁灵波 / Ant LingBot）

**① 架构范式 / Paradigm**：基于 **JEPA（Joint Embedding Predictive Architecture）** 的触觉表征模型——围绕共享的 **HaT（Haptic Transformer）**，在 latent 空间做 masked prediction（预测被遮蔽的时空触觉表征），而非重建原始信号。
*A JEPA-based tactile representation model: a shared **Haptic Transformer (HaT)** predicts masked spatiotemporal tactile context in latent space, rather than reconstructing raw signals.*

**② 架构图 / Diagram**：

```
三类异构信号 / 3 heterogeneous modalities
  magnetic(磁)  piezoresistive(压阻)  visuotactile(视触觉)
        │              │                  │
  [modality-specific tokenizer] ×3  ← 各模态专属分词
        └──────────────┼──────────────────┘
                       ▼
        canonical geometric interface（统一几何接口）
        （对齐 time_id / finger_id / pos_xyz）
                       ▼
              ┌─────────────────┐
              │  shared HaT     │  ← JEPA: masked latent prediction
              │ (Haptic Transf.)│
              └────────┬────────┘
                       ▼
        统一 latent 表征 [B,N,C] + 时空地址
        (token-level features + addresses)
                       ▼
        下游 plug-and-play / downstream
   diffusion policy · π0.5 · LingBot-VLA
```

**③ 核心模块与数据流 / Modules & flow**：
- **模态专属 tokenizer**：三类传感器信号各自分词，再经「统一几何接口」对齐到共享坐标系。
*Modality-specific tokenizers encode each sensor type, then a canonical geometric interface aligns them into a shared coordinate frame.*
- **共享 HaT**：在带时空地址（时间/手指/三维位置）的 token 序列上做 masked latent prediction。
*Shared HaT performs masked latent prediction over tokens carrying time/finger/3D-position addresses.*
- **推理输出**：官方开放 frozen target encoder，输出 `latent [B,N,C]` + `time_id/finger_id/pos_xyz`；触觉 token 可直接喂给 diffusion policy 或 VLA。
*Inference exposes a frozen target encoder outputting `latent [B,N,C]` plus addresses; tactile tokens plug directly into diffusion policies / VLA.*

**④ 关键创新 / Innovations**：**Unified**（多源异构信号在统一 latent 空间互增强）、**Transferable**（跨未见传感器/真机灵巧手零训练迁移）、**Efficient**（最小骨干高频推理）、**Scalable**（数据规模持续增益）。
*Unified (heterogeneous signals reinforce each other in one latent space), Transferable (zero-shot transfer to unseen sensors/hands), Efficient (tiny backbone, high-frequency inference), Scalable (gains scale with data).*

**⑤ 参数与指标 / Scale & metrics**：HaT-**Tiny 仅 5.46M 参数**；Tiny/Small 两个 checkpoint；预训练 **4,983 小时 / 464,431 episodes / 约 13.3 亿帧**（压阻 2197h + 磁式 1136h + 视触觉 1650h）。
*HaT-Tiny only 5.46M params; pretrained on 4,983 h / 464,431 episodes / ~1.33B frames.*

**⑥ 架构约束 / Constraints**：新传感器须按 canonical 几何与标定参数归一化，**点数/布局需匹配已验证配置**（仅形状相同不够，要核对 3D 坐标与信号轴顺序）。
*New sensors must be normalized to the canonical geometry; point counts/layouts must match validated configs (shape alone is insufficient).*

---

### 1.2 TipSim（华威科 / Huaweike tiplab）

**① 架构范式 / Paradigm**：**不是神经网络模型，而是 MuJoCo 触觉仿真插件平台**——以预编译 `.so` 运行时插件形式提供 Loong / Universa 系列传感器，经 MuJoCo 标准 sensor 接口读取 taxel 阵列。
*Not a neural model but a **MuJoCo tactile simulation plugin platform**: Loong/Universa sensors ship as precompiled `.so` runtime plugins read via MuJoCo's standard sensor interface.*

**② 架构图 / Diagram**：

```
场景 XML (<plugin> 声明传感器阵列 / declare sensor array)
        │
  mujoco.mj_loadPluginLibrary("lib/libtip_*.so")  ← 加载闭源插件
        │
        ▼
  ┌──────────────────┐
  │ MuJoCo 求解器     │ ← 动力学 + 接触约束求解
  │ + 软垫 geom       │   (名字后缀 tip.tactile.* 自动识别)
  └────────┬─────────┘
           ▼ mj_step() / mj_forward()
  mjData.sensordata → reshape → per-taxel 力阵列
           │
           ▼
  抓取控制环（力达阈值→停闭合→抬升/放置）
  + ROS2 publishing
```

**③ 核心模块与数据流 / Modules & flow**：插件经 `mj_loadPluginLibrary()` 加载；`mj_step()` 后从 `mjData.sensordata` 按 `sensor_adr/sensor_dim` 索引读出并 reshape 成触觉阵列；软垫 geom 按名字后缀自动识别。
*Plugin loaded via `mj_loadPluginLibrary()`; after `mj_step()`, read `mjData.sensordata` by `sensor_adr/sensor_dim` and reshape into taxel arrays; soft-pad geoms auto-detected by name suffix.*

**④ 关键创新 / Innovations**：连续接触区/力分布感知；动力学求解与触觉反馈控制耦合；参数可配置（量程/精度/触发力/迟滞）+ 可复现实验；多 MuJoCo 版本支持。
*Continuous contact-area/force-distribution sensing; coupling dynamics solving with tactile feedback control; configurable + reproducible; multi-version MuJoCo support.*

**⑤ 参数与指标 / Scale & metrics**：**无神经网络参数量（官方未披露，因为不存在）**；预编译插件支持 MuJoCo **3.9.0–3.16.0 共 8 个 minor 版本**。
*No NN parameters (N/A); plugins support MuJoCo 3.9.0–3.16.0 (8 minor versions).*

**⑥ 架构约束 / Constraints**：插件内部实现**明确不在开源范围**（"out of scope for open-source contributions"）；自动几何检测依赖 geom 命名后缀；仅 Linux x86_64 预编译。
*Plugin internals explicitly out of open-source scope; auto-detection relies on geom name suffixes; Linux x86_64 precompiled only.*

---

## 二、视频感知：视频理解与分析 / 2. Video Perception: Understanding & Analysis

### 2.1 PaddleVideo · PP-TSM / PP-TSMv2（百度飞桨 / Baidu）

**① 架构范式 / Paradigm**：**2D CNN + TSM（Temporal Shift Module）**——底层机制是把部分通道沿时间维移位，**零参数、零 FLOPs** 实现时序建模，达到近似 3D CNN 的效果；并非 3D 卷积网络。
*2D CNN + TSM: shift part of channels along the temporal dimension — **zero-parameter, zero-FLOP** temporal modeling approximating 3D CNNs; NOT a 3D conv net.*

**② 架构图 / Diagram**：

```
视频 → 等分 N 段各取 1 帧（最终 16 帧 / 16 frames）
        │
  ┌─────▼──────────────────────────┐
  │ backbone: ResNet-50 (PP-TSM)   │
  │       或 PP-LCNetV2 (v2, CPU)  │
  │  ┌── Stage3 ────────────────┐  │
  │  │ [LTA 全局时序注意力]      │  │ ← v2 新增，置于 TSM 之前
  │  │ [temporal_shift 通道时移] │  │ ← 零参数时序建模
  │  └──────────────────────────┘  │
  └─────┬──────────────────────────┘
        ▼  GAP + FC → 动作类别
  数据增强 VideoMix · 蒸馏 DML · 解码 decord
```

**③ 核心模块与数据流 / Modules & flow**：PP-TSM 基于 ResNet-50 做 6 方面调优；PP-TSMv2 换用 CPU 端 PP-LCNetV2（PW 卷积+SE+重参数化），在 **Stage3 插 1 个 temporal_shift**（高层优于低层）；新增 **LTA（Lightweight Temporal Attention）** 用全局信息指导时移，加在 TSM 之前。
*PP-TSM tunes ResNet-50; v2 switches to CPU-friendly PP-LCNetV2, inserts one `temporal_shift` at Stage3, and adds **LTA** before TSM to guide shifting with global cues.*

**④ 关键创新 / Innovations**：TSM 零参数时序建模；LTA 全局时序注意力（+0.78%）；DML 互蒸馏（teacher 为 PP-TSM ResNet-50 时 +2.20%）；VideoMix 增强；CPU 端 456ms/10s 视频的产业级精度-速度平衡。
*Zero-param TSM; LTA (+0.78%); DML mutual distillation (+2.20%); VideoMix; industry-grade accuracy-speed balance at 456ms/10s on CPU.*

**⑤ 参数与指标 / Scale & metrics**：官方**未给出参数量/FLOPs**；Kinetics-400 中心采样 Top-1：PP-TSMv2 **75.16%**（16f+DML），PP-TSM ResNet101 Dense 77.15%、+KD 的 ResNet50 76.16%。
*Params/FLOPs not disclosed; Kinetics-400 top-1: PP-TSMv2 75.16%, PP-TSM ResNet101 Dense 77.15%.*

**⑥ 架构约束 / Constraints**：TSM 仅局部通道位移、缺全局时序建模（故需 LTA）；PP-LCNetV2 去掉部分残差连接使 TSM 原论文的残差内插入方式不再适用；CPU 端瓶颈在视频解码。
*TSM is local only (hence LTA); PP-LCNetV2's removed residual links invalidate TSM's original insertion; CPU bottleneck is decoding.*

---

### 2.2 ByteTrack（华中科大+港大+字节 / HUST+HKU+ByteDance）

**① 架构范式 / Paradigm**：**Tracking-by-detection**，核心 **BYTE 关联**——关联「几乎所有检测框」而非仅高分框；检测器为 **YOLOX-X**。
*Tracking-by-detection with **BYTE association** — associate "almost every detection box", not just high-score ones; detector is YOLOX-X.*

**② 架构图 / Diagram**：

```
每帧帧 f_k → YOLOX 检测 D_k
        │ 按阈值 τ=0.6 分组
   ┌────┴─────┐
 D_high     D_low   （高/低分框）
   │           │
 KalmanFilter 预测 tracks
   │
 [First association]  D_high ↔ 全部 tracks
   相似度#1 = IoU 或 ReID → Hungarian
   │ 未匹配: D_remain / T_remain
   ▼
 [Second association] D_low ↔ T_remain
   相似度#2 = 仅 IoU（低分框外观不可靠；比 ReID +1.0 MOTA）
   │ 未匹配低分框=背景删除; 仍未匹配 tracks → T_lost(保留30帧)
   ▼
 新轨迹只用 D_remain（第一次关联的未匹配高分框）初始化
```

**③ 核心模块与数据流 / Modules & flow**：两阶段关联——第一次高分框匹配全部 tracks；第二次低分框匹配剩余 tracks，**只用 IoU**（低分框常严重遮挡/运动模糊，外观特征不可靠，实测比 ReID +1.0 MOTA）；新轨迹只用第一次关联的未匹配高分框初始化，避免背景误建轨。
*Two-pass association: high-score boxes match all tracks; low-score boxes match remaining tracks using **IoU only** (+1.0 MOTA over ReID); new tracks initialized only from unmatched high-score boxes to avoid background false tracks.*

**④ 关键创新 / Innovations**：BYTE 把每个检测框视为 tracklet 基本单元；创新位于「检测与关联的交界处」；低分框既作恢复真目标的桥梁、又用于过滤背景；可迁移——套用到 9 个 SOTA tracker，IDF1 一致提升 1–10 点。
*BYTE treats every box as a tracklet unit; innovation at the "junction of detection and association"; low-score boxes both recover true objects and filter background; portable to 9 SOTA trackers with +1–10 IDF1.*

**⑤ 参数与指标 / Scale & metrics**：MOT17 **80.3 MOTA / 77.3 IDF1 / 63.1 HOTA @ 29.6 FPS**（单 V100）；MOT20 77.8 MOTA；轻量版 nano **0.90M 参数 / 3.99G FLOPs**（69.0 MOTA）、tiny 5.03M/24.45G（77.1 MOTA）。
*MOT17 80.3 MOTA @ 29.6 FPS (one V100); nano 0.90M params (69.0 MOTA), tiny 5.03M (77.1 MOTA).*

**⑥ 架构约束 / Constraints**：默认 τ=0.6；benchmark 只用 IoU，IoU<0.2 拒绝匹配；第二次关联不用外观；BDD100K 用 UniTrack ReID 且不用 Kalman。
*Default τ=0.6; benchmarks use IoU only, reject IoU<0.2; no appearance in second pass; BDD100K uses UniTrack ReID without Kalman.*

---

## 三、视频感知：视频生成 / 3. Video Perception: Video Generation

> 三家都是 **DiT（Diffusion Transformer）** 路线，但**训练目标与条件注入方式不同**——这是理解三者差异的主线。
> *All three use **DiT**, but differ in **training objective and conditioning** — the key axis for comparison.*
> 对照表见本节末 / comparison table at end of this section.

### 3.1 HunyuanVideo（腾讯混元 / Tencent, 13B）

**① 架构范式 / Paradigm**：**DiT + Flow Matching**，推理用一阶 Euler ODE；**双流→单流（Dual-stream → Single-stream）** 混合设计 + **统一全注意力（unified Full Attention）+ 3D RoPE**（非 2D+1D 分离注意力）。
*DiT + Flow Matching (Euler ODE); **Dual-stream→Single-stream** hybrid + **unified Full Attention** + **3D RoPE** (not separated 2D+1D).*

**② 架构图 / Diagram**：

```
text ─[MLLM decoder-only encoder]→ text tokens ─┐
                                    ┌───────────┘
CLIP-L → global guidance token ─────┤
                                    ▼
  ┌─ Dual-stream (20 blocks) ─────────────────┐
  │ video tokens ─┐  ┌─ text tokens           │  ← 各模态独立 modulation
  │ (3D VAE latent)│  │                        │
  └───────┬────────┴──┬────────────────────────┘
          │ concat    │
  ┌───────▼─ Single-stream (40 blocks) ───────┐
  │  unified Full Attention + 3D RoPE          │  ← 多模态融合
  └───────┬────────────────────────────────────┘
          ▼ flow-reverse 去噪 → 3D VAE decode → video
  3D VAE: CausalConv3D, 压缩 4×8×8, latent C=16
```

**③ 核心模块与数据流 / Modules & flow**：**Dual-stream 阶段**视频/文本 token 各自过 block（各学自己的 modulation，互不干扰，20 块）；**Single-stream 阶段** concat 后融合（40 块）；文本侧用 **MLLM 作为 text encoder**（非 T5），因 causal attention 另加 **bidirectional token refiner**，并用 **CLIP-Large** 末 token 作 global guidance。
*Dual-stream (20 blocks) lets video/text learn separate modulations; single-stream (40 blocks) concatenates for fusion; text encoder is an **MLLM** (not T5) plus a bidirectional token refiner and a CLIP-Large global token.*

**④ 关键创新 / Innovations**：MLLM 作 text encoder（更强图文对齐、image detail description、zero-shot 指令）；unified Full Attention + 3D RoPE；Dual→Single 兼顾模态独立与融合；timestep shifting（50 步 s=7，<20 步 s=17）；spatial-temporal tiling 支持任意分辨率。
*MLLM as text encoder (better alignment/reasoning); unified Full Attention + 3D RoPE; Dual→Single; timestep shifting; spatial-temporal tiling.*

**⑤ 参数与指标 / Scale & metrics**：总规模 **>13B**（论文称当时开源最大）；block 配置 dim 3072 / FFN 12288 / 24 heads；3D VAE 压缩 4×8×8、latent 16 通道。DiT/VAE 分项参数官方未披露。
*Total >13B; dim 3072 / FFN 12288 / 24 heads; 3D VAE 4×8×8. Per-component params not disclosed.*

**⑥ 架构约束 / Constraints**：默认 129 帧、540p/720p、50 steps（flow-shift 7.0、embedded CFG 6.0）；先在 256/512px 图像预训练，再 256→960px 图文视频联合训练（视频:图像=4:1）。
*Default 129 frames, 540p/720p, 50 steps; image pretraining then 256→960px joint training (video:image=4:1).*

---

### 3.2 Wan2.1（阿里通义万相 / Alibaba Wan）

**① 架构范式 / Paradigm**：**DiT + Flow Matching / Rectified Flow**（`x_t = t·x₁+(1-t)·x₀`，回归速度）；**单流 DiT + cross-attention 注入文本**；patchify 用 3D conv(1,2,2)，序列长 `L=(1+T/4)×H/16×W/16`。
*DiT + Flow Matching/Rectified Flow; **single-stream DiT with cross-attention** for text; 3D-conv patchify.*

**② 架构图 / Diagram**：

```
text ─[umT5 多语言编码]→ 512 token embedding ──┐
                                              │ cross-attention（每 block 注入）
  ┌───────────────────────────────────────────▼──┐
  │ DiT blocks (1.3B:30层 / 14B:40层)            │
  │  - 共享时序 MLP 预测 6 个 modulation 参数      │ ← 省约 25% 参数
  │  - patchify: 3D conv (1,2,2)                 │
  └──────────────────┬───────────────────────────┘
                     ▼ Flow Matching 去噪
  Wan-VAE decode: 3D causal VAE, 4×8×8, C=16, 127M
        （RMSNorm 替代 GroupNorm 保时序因果；
          feature cache 支持无限长；首帧仅空间压缩）
```

**③ 核心模块与数据流 / Modules & flow**：文本经 **umT5** 编码为 512 token embedding（选中英文/视觉文字的多语言能力），在每个 transformer block 内以 **cross-attention** 注入；时间嵌入用**共享 MLP 预测 6 个 modulation 参数**（所有 block 共享权重、各 block 学 bias，省约 25% 参数）。
*Text encoded by **umT5** (multilingual), injected via **cross-attention** in every block; a **shared temporal MLP predicts 6 modulation params**, saving ~25% parameters.*

**④ 关键创新 / Innovations**：**Wan-VAE**（3D causal VAE，用 RMSNorm 保因果、feature cache 支持无限长视频、空间上采样输入通道减半省 33% 显存、重建比 Hunyuan 快 2.5×，仅 127M）；共享 modulation MLP；中英文文字生成；三阶段 VAE 训练。
*Wan-VAE (3D causal, RMSNorm, feature cache for infinite length, 33% VRAM saving, 2.5× faster reconstruction, only 127M); shared modulation MLP; bilingual text rendering.*

**⑤ 参数与指标 / Scale & metrics**：DiT **1.3B**（dim1536/12 heads/30 layers）与 **14B**（dim5120/40 heads/40 layers）；Wan-VAE **127M**；1.3B 推理仅 **8.19GB 显存**。
*DiT 1.3B and 14B; Wan-VAE 127M; 1.3B inference needs only 8.19GB VRAM.*

**⑥ 架构约束 / Constraints**：典型 81 帧@16fps（约 5s）；~50 steps；1.3B 官方推荐 480P（720P 训练不足不稳）；长序列示例 81f@1280×720。
*Typically 81f@16fps (~5s), ~50 steps; 1.3B recommends 480P.*

---

### 3.3 CogVideoX 1.5（智谱 / Zhipu zai-org, 5B）

**① 架构范式 / Paradigm**：**DiT，但用 v-prediction + zero SNR（非 Flow Matching）**——这是与另两家的关键差异；文本与视频 embedding 在**序列维 concat**，送 **expert transformer blocks**；3D 全注意力。
*DiT but with **v-prediction + zero SNR (NOT Flow Matching)** — the key difference; text/video embeddings concatenated along sequence into **expert transformer blocks**; 3D full attention.*

**② 架构图 / Diagram**：

```
text ─[T5]→ z_text (1.5: ≤224 tokens, English) ─┐
video ─[3D causal VAE, 8×8×4, C=16]→ z_video ─┤
                                              │ concat（序列维）
  ┌───────────────────────────────────────────▼─────────┐
  │ Expert Transformer blocks (5B:42层/48heads)          │
  │  - Vision Expert AdaLN ┐ 由 timestep t 驱动          │ ← 模态专属 modulation
  │  - Text Expert AdaLN   ┘                            │
  │  - 3D Full Attention + 3D-RoPE (通道 3/8,3/8,2/8)    │
  └──────────────────┬───────────────────────────────────┘
                     ▼ v-prediction 去噪 → VAE decode → video
  帧数须 16N+1（默认81），5s/10s@16fps，1360×768，BF16
```

**③ 核心模块与数据流 / Modules & flow**：文本 T5 编码（1.5 限 224 tokens、英文）；3D causal VAE 压缩 8×8×4；文本与视频 embedding 在序列维拼接；**Expert Transformer** 用 timestep 驱动 **Vision/Text 两个 Expert AdaLN** 分别处理两模态（因特征空间/数值尺度差异大），促进对齐且少增参数；3D-RoPE 每维独立 1D-RoPE（通道占比 3/8、3/8、2/8）。
*T5 text (≤224 tokens for 1.5); 3D causal VAE 8×8×4; sequence-level concat; **Expert Transformer** with timestep-driven **Vision/Text Expert AdaLN**; 3D-RoPE per-dimension.*

**④ 关键创新 / Innovations**：3D 全注意力（反对分离式 spatial+temporal，保大运动物体一致性）；Expert Adaptive LayerNorm； Explicit Uniform Sampling 稳定 loss；多分辨率 frame pack。
*3D full attention (better object consistency); Expert Adaptive LayerNorm; Explicit Uniform Sampling; multi-resolution frame pack.*

**⑤ 参数与指标 / Scale & metrics**：DiT **5B / 2B**（5B:42层/48heads/hidden3072）；1.5 版本 CogVideoX1.5-5B（-I2V）；基础版 5B@480×720-6s 约 113s/26GB。
*DiT 5B/2B; 1.5-5B variant; base 5B ~113s/26GB at 480×720-6s.*

**⑥ 架构约束 / Constraints**：1.5 分辨率 1360×768、帧数 **16N+1（N≤10，默认 81）**、5s 或 10s@16fps、BF16；基础版帧数 8N+1（默认 49）、8fps。VAE/T5 参数量官方未披露。
*1.5: 1360×768, frames **16N+1** (default 81), 5s/10s@16fps, BF16; base: 8N+1, 8fps.*

---

### 三家视频生成架构对照 / Video Generation Architecture Comparison

| 维度 / Dimension | HunyuanVideo | Wan2.1 | CogVideoX 1.5 |
|---|---|---|---|
| 范式 / Paradigm | DiT + Flow Matching | DiT + Flow Matching/Rectified | DiT + **v-pred/zero SNR** |
| 条件注入 / Conditioning | Dual→Single stream + Full Attn | 单流 + cross-attention | 序列 concat + Expert AdaLN |
| 文本编码 / Text enc | MLLM + refiner + CLIP-L | umT5（多语言） | T5（英文 224 tok） |
| VAE 压缩 / VAE | CausalConv3D 4×8×8, C16 | 3D causal 4×8×8, C16, 127M | 3D causal 8×8×4, C16 |
| 位置编码 / Pos enc | 3D RoPE | —（时序 MLP modulation） | 3D-RoPE (3/8,3/8,2/8) |
| 规模 / Scale | >13B | DiT 1.3B/14B | DiT 2B/5B |
| 时序约束 / Frames | 129f, 50 steps | 81f@16fps, ~50 steps | 16N+1, 81f@16fps |

---

## 四、声音/语音感知：ASR / 4. Sound: Automatic Speech Recognition

### 4.1 Fun-ASR-Nano（阿里通义 / Alibaba, 0.8B）

**① 架构范式 / Paradigm**：**LLM-based ASR**，四组件级联：**audio encoder + audio adaptor + CTC decoder + LLM-based decoder**（非纯 Conformer+CTC，也非纯 AED）。
*LLM-based ASR with four components: **audio encoder + adaptor + CTC decoder + LLM decoder**.*

**② 架构图 / Diagram**：

```
audio → [audio encoder, Best-RQ 自监督预训练]
              │  (用 Qwen3 层权重初始化)
   ┌──────────┴──────────┐
   ▼                     ▼
[audio adaptor 2层Transf]  [CTC decoder]→greedy hypothesis
   │ 对齐到 LLM 语义空间      │ → RAG 取 context / hotword 检索
   └──────────┬──────────┘
              ▼
  [LLM-based decoder] ← audio condition + CTC pred + hotwords
              ▼         （SFT 5 阶段 + FunRL GRPO 强化）
        最终转写文本
```

**③ 核心模块与数据流 / Modules & flow**：audio encoder 经 Best-RQ 自监督 + AED 监督两阶段预训练（用 Qwen3 层权重初始化）；2 层 Transformer adaptor 对齐 LLM 语义空间；CTC decoder 出 initial hypothesis 供 RAG/hotword 检索；LLM decoder 融合 audio condition + CTC 预测 + hotword 生成最终文本。
*Encoder pretrained with Best-RQ + AED; a 2-layer adaptor aligns to LLM space; CTC hypothesis feeds RAG/hotword retrieval; LLM decoder fuses all for final text.*

**④ 关键创新 / Innovations**：**FunRL**（GRPO 强化学习，reward 含 ASR 准确率/关键词召回/噪声鲁棒/幻觉抑制，offline +4.1%、streaming +9.2% relative）；RAG 热词定制（name recall 0.75→1.0）；流式训练缩小训推失配。
*FunRL (GRPO, +4.1% offline / +9.2% streaming); RAG hotword customization; streaming training to close train/infer gap.*

**⑤ 参数与指标 / Scale & metrics**：Nano **0.8B（encoder 0.2B + LLM 0.6B）**；AIShell-1 WER 1.80；工业集平均 WER 16.72；vLLM RTFx 340×。adaptor/CTC 参数未披露。
*Nano 0.8B (enc 0.2B + LLM 0.6B); AIShell-1 WER 1.80; vLLM RTFx 340×.*

**⑥ 架构约束 / Constraints**：输入采样率、tokenizer Hz、推理最大时长**官方均未披露**；当前版本不支持 far-field / multi-channel。
*Input sample rate, tokenizer Hz, max inference duration all undisclosed; no far-field/multi-channel support.*

---

### 4.2 FireRedASR（小红书 / Xiaohongshu FireRed）

**① 架构范式 / Paradigm**：**两个变体共享前端**——AED 版（Conformer Encoder + Transformer Decoder）；LLM 版（**Encoder-Adapter-LLM**，LLM 初始化自 Qwen2-7B-Instruct）。
*Two variants sharing frontend: AED (Conformer Encoder + Transformer Decoder); LLM (Encoder-Adapter-LLM, LLM from Qwen2-7B-Instruct).*

**② 架构图 / Diagram**：

```
audio → 80-dim log-Mel Fbank (25ms窗/10ms移)
        │
  [Conformer Encoder] (2层 stride-2 conv: 10ms→40ms=25Hz)
        │ 输出连续 speech embedding（非离散 token）
   ┌────┴─────────────────────┐
   ▼ AED 版                    ▼ LLM 版
[Transformer Decoder]      [Adapter: Linear-ReLU-Linear
 (weight tying)             + frame splicing 40→80ms=12.5Hz]
   │                           │
   ▼                        [Qwen2-7B-Instruct + LoRA]
 文本                          ▼ next-token prediction → 文本
 参数: XS140M/S413M/M732M/L1.1B   LLM版: enc710M+adapter22M+总8.3B
```

**③ 核心模块与数据流 / Modules & flow**：共享 Conformer Encoder（输出 25Hz 连续特征）；AED 版接 Transformer Decoder；LLM 版经 Linear-ReLU-Linear adapter + frame splicing（降到 12.5Hz）映射到 LLM 输入维，LLM 主体冻结 + LoRA，用 (prompt, speech, transcript) 三元组训练。**注意：LLM 分支用的是连续 speech embedding，非离散声学 token。**
*Shared Conformer Encoder (25Hz continuous features); LLM branch uses a Linear-ReLU-Linear adapter + frame splicing (to 12.5Hz) → frozen LLM + LoRA. **Continuous embeddings, not discrete tokens.** *

**④ 关键创新 / Innovations**：LLM+AED 双变体分层（精度 vs 效率）；LLM encoder 复用 AED 权重、轻量 adapter 对齐；Progressive Regularization Training；歌声/歌词识别能力（自建集 CER 7.05）；scaling law（encoder 86M→710M，CER 3.29→3.05）。
*LLM+AED dual variants; encoder reused from AED; progressive regularization; singing lyric recognition; scaling law.*

**⑤ 参数与指标 / Scale & metrics**：AED XS/S/M/L = 140M/413M/732M/**1.1B**；LLM 版 encoder 710M + adapter 22M + 总 **8.3B**；普通话平均 CER **LLM 3.05 / AED 3.18**（优于 Seed-ASR 3.33、Whisper 9.86）。
*AED up to 1.1B; LLM variant total 8.3B; Mandarin avg CER 3.05/3.18.*

**⑥ 架构约束 / Constraints**：AED 支持至 60s（>200s 触发位置编码错误）；LLM 支持至 30s；输入须 16kHz/16bit 单声道。
*AED up to 60s, LLM up to 30s; 16kHz/16bit mono input.*

---

## 五、声音/语音感知：TTS / 5. Sound: Text-to-Speech

### 5.1 CosyVoice 3.0（阿里，官方名 Fun-CosyVoice3-0.5B）

**① 架构范式 / Paradigm**：**两阶段 coarse-to-fine**：**AR LLM（离散 speech token）→ conditional Flow Matching（DiT）→ vocoder → waveform**；非单阶段。
*Two-stage coarse-to-fine: **AR LLM (discrete tokens) → CFM (DiT) → vocoder → waveform**.*

**② 架构图 / Diagram**：

```
text (raw text, TN 靠 self-training)
  + 3s 参考音频 → [FSQ-MinMo speech tokenizer, 25Hz] → 离散 token
        │
  ┌─────▼──────────────┐
  │ TTS LLM (0.5B)      │ ← 自回归生成离散 speech token @25Hz
  │ BPE + word/phoneme  │   （DiffRO 强化：Token2Text reward）
  └─────┬──────────────┘
        ▼
  [CFM: Diffusion Transformer, 100M]  ← 移除复杂 text encoder/length reg.
        ▼
  [vocoder] → waveform（具体类型官方未披露）
```

**③ 核心模块与数据流 / Modules & flow**：speech tokenizer = **FSQ-MinMo**（把 Finite Scalar Quantization 插入 MinMo voice encoder，训练数据 53 万小时）；TTS LLM 自回归生成 **25Hz** 离散 token（支持 BPE + 拼音/CMU 音素混合，无需传统 frontend）；CFM 用 **DiT** backbone，移除复杂 text encoder 与 length regularization，用 interpolation 解决帧率失配。
*Tokenizer = FSQ-MinMo (530k h); TTS LLM generates 25Hz discrete tokens; CFM uses a DiT backbone, dropping complex text encoder/length reg.*

**④ 关键创新 / Innovations**：**DiffRO（Differentiable Reward Optimization）**——用 ASR-like Token2Text posterior 作 reward，Gumbel-Softmax + token-level KL 直接对 speech token 反传，WER 相对降 20–50%；in-the-wild 零样本克隆（抗噪/长静音 reference）；9 语言 + 18 种中文方言。
*DiffRO (differentiable reward on speech tokens, -20–50% WER); in-the-wild zero-shot cloning; 9 languages + 18 Chinese dialects.*

**⑤ 参数与指标 / Scale & metrics**：TTS LM **0.5B（CFM 100M）**，另有 1.5B 版（CFM 300M）；token rate 25Hz；test-zh CER 1.21（RL 版 0.81）。tokenizer/vocoder 参数未披露。
*TTS LM 0.5B (CFM 100M); 25Hz; test-zh CER 1.21.*

**⑥ 架构约束 / Constraints**：vocoder 类型/采样率、Mel 维度/hop 官方未披露；**150ms 流式延迟仅见 README（Bi-Streaming），论文未披露**；训练片段 <30s。
*Vocoder type/sample rate undisclosed; **150ms streaming latency only in README**; training clips <30s.*

---

### 5.2 FireRedTTS2（小红书 / Xiaohongshu FireRed, 1.5B*）

**① 架构范式 / Paradigm**：**token-based 自回归 TTS（非 flow matching）**——**12.5Hz 流式 tokenizer + 文本-语音交错序列 + dual-transformer + Vocos 声码器**。
*Token-based **autoregressive TTS (not flow matching)**: 12.5Hz streaming tokenizer + text-speech interleaved sequence + dual-transformer + Vocos decoder.*

**② 架构图 / Diagram**：

```
序列: [S1]<text><audio>[S2]<text><audio>… （说话人 tag + 时间对齐 token）
  audio(16kHz) → Whisper encoder(语义) + acoustic encoder → adapter
        → 50Hz 降采样 4× → 12.5Hz → RVQ(16层×2048, BPS2200)
        → 升回 50Hz → semantic decoder → Vocos → waveform
        │
  ┌─────▼─────────────────────────────────┐
  │ dual-transformer（均基于 Qwen2.5 结构）│
  │  - large decoder-only: 整条混排序列 →  │
  │    预测 first-layer token + hidden     │
  │  - small decoder: 补齐剩余 15 层 token  │
  │    （1+(N-1) 步生成 16 层，降首包延迟） │
  └───────────────────────────────────────┘
```

**③ 核心模块与数据流 / Modules & flow**：**dual-transformer**——大 backbone 处理整条混排序列、预测第一层 token 并输出 hidden states，小 decoder 同时吃第一层 token + hidden 补齐剩余 15 层（相比 delay-pattern 降低首包延迟）；tokenizer 用 Whisper encoder 提语义 + RVQ（16 层×2048），从 50Hz 降采样到 12.5Hz，Vocos 重建波形。
*Dual-transformer: large backbone predicts first-layer tokens + hidden states, smaller decoder fills remaining 15 layers; tokenizer uses Whisper features + RVQ (16×2048) downsampled 50→12.5Hz, Vocos reconstructs waveform.*

**④ 关键创新 / Innovations**：长对话多说话人（当前 3 分钟/4 说话人）；dual-transformer 更低首包延迟；句级流式生成；零样本跨语/语码混合克隆（Test-ZH CER 1.14/SIM 0.736）；交互式情感控制（6 类情绪）。
*Long multi-speaker dialogue (3 min / 4 speakers); lower first-packet latency; sentence-level streaming; zero-shot cross-lingual cloning; emotion control.*

**⑤ 参数与指标 / Scale & metrics**：**参数量官方 README 与论文均未披露**（*1.5B 仅见二手来源，不能作一手结论）；tokenizer 重建 WER 2.16 / SIM 0.87 / UTMOS 3.88。
***Params not disclosed officially** (*1.5B only in secondary sources); tokenizer WER 2.16 / SIM 0.87.*

**⑥ 架构约束 / Constraints**：**延迟两官方口径冲突——README「140ms @ L20」、论文「<100ms」**；tokenizer 输入 16kHz，Stage-2 全流式 decoder 输出 24kHz；推理硬时长上限未披露。
**Two conflicting official figures — README "140ms @ L20" vs paper "<100ms"**; 16kHz input, 24kHz streaming output.

---

## 六、全双工语音交互与音频理解 / 6. Full-Duplex Interaction & Audio Understanding

### 6.1 FireRedChat（小红书 / Xiaohongshu FireRed）

**① 架构范式 / Paradigm**：**模块化可插拔全双工系统**（非端到端模型）——**turn-taking controller + interaction module + dialogue manager** 三部分；独立轮次控制器可把 half-duplex 流水线升级为 full duplex。
*Modular, plug-and-play **full-duplex system** (not end-to-end): **turn-taking controller + interaction module + dialogue manager**; a pluggable controller upgrades half-duplex pipelines to full duplex.*

**② 架构图 / Diagram**：

```
用户语音 ─RTC→ [streaming pVAD]  ← 抑制噪声/非目标说话人
   │ 产出时间戳 → 检测主说话人 → 暂停 TTS 实现 barge-in
   │ ★ pVAD 降噪后的音频不喂下游；只用时间戳从原始音频截取
   ▼
原始片段 → [ASR] → transcript → [EoT: BERT分类器170M]
   │                             语义停止判断（未完则缓冲 partial）
   ▼ 结束
[dialogue manager: Dify, 可 tool invocation] → text LLM / AudioLLM
   ▼
[TTS: FireRedTTS-1s / -2] → 播放
```

**③ 核心模块与数据流 / Modules & flow**：**pVAD**（mel → causal conv → ECAPA-TDNN 目标说话人 embedding → GRU → 每步 10ms 说话概率）做 barge-in；关键设计是**降噪音频不喂下游**，只用时间戳从原始音频截取以保留副语言线索；**EoT**（BERT 分类器，170M）对累积 transcript 做语义停止判断。
*pVAD (mel→causal conv→ECAPA-TDNN→GRU, 10ms/step) for barge-in; crucially, denoised audio is NOT fed downstream — timestamps crop the RAW audio to preserve paralinguistic cues; **EoT** (BERT classifier, 170M) judges semantic end-of-turn.*

**④ 关键创新 / Innovations**：pVAD + EoT 双模块互补（"是否真的开始说/目标说话人" + "语义是否说完"）；cascade（FireRedASR+Qwen2.5+FireRedTTS-1s）与 semi-cascade（AudioLLM+FireRedTTS2）两种实现；Dify 对话管理。
*pVAD + EoT complementarity; cascade and semi-cascade implementations; Dify dialogue manager.*

**⑤ 参数与指标 / Scale & metrics**：EoT **170M**；barge-in T90 170ms / false barge-in 10.2%；EoT 准确率中文 96.0% / 英文 94.9%；端到端延迟 P50 2.341s。
*EoT 170M; barge-in T90 170ms; EoT accuracy 96.0% zh; E2E latency P50 2.341s.*

**⑥ 架构约束 / Constraints**：音频采样率、pVAD 超参、context window 论文未披露；speech-to-speech 全双工称可集成但无自研细节。
*Sample rate, pVAD hyperparams, context window undisclosed; S2S full duplex claimed integrable but no details.*

---

### 6.2 SenseVoice（阿里 / Alibaba）

**① 架构范式 / Paradigm**：需分两版——**Small：非自回归 encoder-only（SAN-M）**（"非自回归多任务"仅指它）；**Large：自回归 encoder-decoder**（类 Whisper）。
*Two variants — **Small: non-autoregressive encoder-only (SAN-M)**; **Large: autoregressive encoder-decoder** (Whisper-like).*

**② 架构图 / Diagram**：

```
raw waveform → 80-dim log-mel → 帧堆叠 + down-sample ×6 → X_speech
        │
  [special tokens] <LID> <SER> <AEC> <ITN>/<NoITN>  ← 前置，统一词表
        │
  ┌─────▼──────────────────────────────┐
  │ SAN-M encoder (memory-equipped     │  ← Small: 非自回归，234M
  │ self-attention network)            │
  └─────┬──────────────────────────────┘
        ├─ CTC loss → ASR 文本
        └─ CE loss  → LID / 情感(SER) / 声音事件(AEC)
  输出 rich transcription（文字内嵌 <music>/<happy> 等标签）
```

**③ 核心模块与数据流 / Modules & flow**：raw waveform → 80-dim log-mel → down-sample ×6 → **SAN-M（带记忆的自注意力网络）**；在特征前 concat 4 个 task embedding（`<LID>`/`<SER>`/`<AEC>`/`<ITN>`）统一建模；ASR 用 CTC loss，其余用 CE loss，一次推理同时出 ASR/语种/情感/声音事件。
*Raw waveform → log-mel → ×6 downsample → **SAN-M**; prepend 4 task embeddings; CTC for ASR, CE for others — one pass outputs ASR/LID/SER/AED.*

**④ 关键创新 / Innovations**：单次推理多任务输出；10s 音频延迟 70ms / RTF 0.007（比 Whisper-Small 快 >5×、Large 快 >15×）；训练时 0.8 概率替换 `<LID>` 为真值以支持条件控制。
*Single-pass multi-task; 70ms/10s, RTF 0.007 (>5× vs Whisper-Small); 0.8-prob LID replacement for conditional control.*

**⑤ 参数与指标 / Scale & metrics**：Small **234M**（5 语言，10s 70ms）；Large **1587M≈1.6B**（50+ 语言，10s 1623ms）；AIShell-1 CER Small 2.96 / Large 2.09。
*Small 234M (5 langs, 70ms); Large ~1.6B (50+ langs); AIShell-1 CER 2.96/2.09.*

**⑥ 架构约束 / Constraints**：**非流式设计**（论文明确"not designed for streaming"）；Small 每次 utterance 最多一个 event、无 timestamp；采样率/层数/词表未披露。
*Not designed for streaming; Small outputs at most one event with no timestamp; sample rate/layers undisclosed.*

---

## 七、全模态大模型 / 7. Omni-Modal Large Models

### 7.1 Qwen2.5-Omni-7B（阿里通义 / Alibaba Qwen）

**① 架构范式 / Paradigm**：**Thinker-Talker 架构**（官方未用"双核"一词）——Thinker（Transformer decoder）做全模态理解并生成文本；Talker（**dual-track AR Transformer Decoder**）流式接收 Thinker 高维表示与文本、输出语音 token；端到端联合训练；**TMRoPE（Time-aligned Multimodal RoPE）** 按时间交错组织音视频。
***Thinker-Talker architecture** (no official "dual-core"): Thinker (Transformer decoder) for omni understanding/text; Talker (dual-track AR decoder) for streaming speech tokens; **TMRoPE** interleaves audio/video by time.*

**② 架构图 / Diagram**：

```
输入: text / image / audio / video
  ├─ audio encoder (Whisper-large-v3 初始化, 16kHz, 2s 分块流式)
  └─ vision encoder (Qwen2.5-VL ViT ~675M, patch14, 2×2 MLP合并)
        │ + TMRoPE（音视频时间戳交错对齐）
        ▼
  ┌──────────────────────────────┐
  │ Thinker (Transformer decoder)│ → 文本 (+ 高维表示传给 Talker)
  └───────────────┬──────────────┘
                  ▼ 复用 Thinker 全部历史上下文
  ┌──────────────────────────────┐
  │ Talker (dual-track AR)       │ → speech token
  └───────────────┬──────────────┘
                  ▼
  [sliding-window DiT (Flow Matching, 感受野4块)] → mel
        → [modified BigVGAN] → waveform（流式）
```

**③ 核心模块与数据流 / Modules & flow**：audio encoder 初始化自 Whisper-large-v3（16kHz、128 通道 mel、2s 分块）；vision encoder 来自 Qwen2.5-VL（ViT ~675M、patch14、2×2 token 合并）；Thinker 生成文本，Talker 复用其全部历史上下文流式生成 speech token；speech → **sliding-window DiT**（感受野限 4 block 降首包延迟）→ mel → modified BigVGAN → waveform。
*Audio encoder from Whisper-large-v3; vision encoder from Qwen2.5-VL; Talker reuses Thinker's full context; sliding-window DiT (4-block receptive field) → BigVGAN for low-latency streaming speech.*

**④ 关键创新 / Innovations**：Thinker-Talker 端到端（避免模态干扰）；TMRoPE 显式时间对齐；全模态流式理解+语音生成（block-wise 编码 + sliding-window DiT）；端到端语音指令跟随接近纯文本。
*Thinker-Talker end-to-end; TMRoPE; omni streaming understanding+generation; speech instruction-following near text-level.*

**⑤ 参数与指标 / Scale & metrics**：7B / 3B 两尺寸（+4bit GPTQ/AWQ 量化版）；OmniBench 平均 **56.13%**（SOTA）；MMAU 65.60；seed-tts-eval WER 1.42%(zh)/2.33%(en)。Thinker/Talker 分项参数未披露。
*7B/3B + 4bit versions; OmniBench 56.13% (SOTA); MMAU 65.60; Thinker/Talker params undisclosed.*

**⑥ 架构约束 / Constraints**：编码器 block 化 + sliding-window DiT 是为降首包延迟的取舍；语音生成不必与文本逐词对齐；7B BF16 处理 60s 视频需 60GB（4bit 可降至 ~11.6GB）。
*Block-wise encoding + sliding-window DiT trade-offs for latency; 7B BF16 needs 60GB for 60s video (4bit ~11.6GB).*

---

### 7.2 InternVL3.5（上海 AI Lab · OpenGVLab，书生·万象3.5）

**① 架构范式 / Paradigm**：**ViT-MLP-LLM** 范式（延续 InternVL 系列）——LLM 从 Qwen3/GPT-OSS 初始化，vision encoder 用 InternViT-300M/6B，保留 **Dynamic High Resolution**；训练用 **Cascade RL**。
*ViT-MLP-LLM paradigm: LLM from Qwen3/GPT-OSS, InternViT vision encoder, Dynamic High Resolution, trained with **Cascade RL**.*

**② 架构图 / Diagram**：

```
image/video → Dynamic High Resolution 切块
        │
  [InternViT-300M / 6B] → visual tokens
        │
  [MLP] → 对齐到 LLM 空间
        │
  ┌─────▼──────────────────────────────┐
  │ ViR 视觉分辨率路由器（仅 Flash）     │ ← patch-aware 路由，token 最多 -50%
  │ + ViCO 视觉一致性学习（一致性正则）  │
  └─────┬──────────────────────────────┘
        ▼
  [LLM: Qwen3 / GPT-OSS 初始化]
        │ Cascade RL: CPT→SFT→MPO(离线)→GSPO(在线)
        ▼
  文本输出（Thinking / Best-of-N 测试时扩展）
  部署: DvD 解耦（ViT/MLP/ViR 在 vision server, LLM 在 language server）
```

**③ 核心模块与数据流 / Modules & flow**：ViT-MLP-LLM 主干；**ViR（Visual Resolution Router，仅 Flash 版）** 按 patch 语义丰富度路由到不同压缩率（token 最多 -50%）；**ViCO（Visual Consistency Learning）** 用 KL 一致性 + router 训练保证不同分辨率鲁棒；**DvD（Decoupled Vision-Language Deployment）** 把视觉模块与 LLM 分服务器异步流水线。
*ViT-MLP-LLM backbone; **ViR** routes patches to compression rates (-50% tokens, Flash only); **ViCO** consistency learning; **DvD** decouples vision and LLM onto separate servers.*

**④ 关键创新 / Innovations**：Cascade RL（粗到细：MPO 预热→GSPO 在线）；ViR 自适应压缩 + ViCO；DvD 解耦部署；新增 GUI interaction / embodied agency；较 InternVL3 推理 +16.0%、速度 4.05×。
*Cascade RL (MPO→GSPO); ViR+ViCO; DvD; GUI/embodied capabilities; +16.0% reasoning, 4.05× speed vs InternVL3.*

**⑤ 参数与指标 / Scale & metrics**：9 种尺寸（1B–241B），旗舰 **241B-A28B**（ViT 5.5B + LLM 235.1B，激活 28B）；MMMU 8B 得 73.4、241B-A28B 得 **77.7**（开源最高）。
*9 sizes (1B–241B), flagship 241B-A28B (5.5B ViT + 235.1B LLM, 28B active); MMMU 77.7 (open-source best).*

**⑥ 架构约束 / Constraints**：ViR/ViCO 仅 Flash 变体；DvD 需多 GPU 分离部署；无音频输入输出（视觉为主）。
*ViR/ViCO only in Flash; DvD needs multi-GPU split; no audio I/O (vision-centric).*

---

## 八、横向洞察 / 8. Cross-Cutting Insights

### 8.1 架构范式收敛与分化 / Convergence & divergence

| 方向 / Domain | 收敛的范式 / Converged paradigm | 分化点 / Divergence |
|---|---|---|
| 视频生成 / Video gen | 全面转向 **DiT** | 训练目标：Flow Matching（Hunyuan/Wan）vs v-pred（CogVideoX）；条件注入：dual-stream / cross-attn / concat+expert |
| ASR | 转向 **LLM-based**（encoder+adaptor+LLM） | 连续特征（Fun-ASR/FireRedASR-LLM）vs 传统 AED；是否加 CTC 分支 |
| TTS | 全面 **离散 token + AR LLM** | 声学解码：Flow Matching（CosyVoice）vs RVQ+Vocos（FireRedTTS2） |
| 全模态 / Omni | 端到端统一 | Thinker-Talker（含语音生成）vs ViT-LLM（纯视觉） |
| 多目标跟踪 / MOT | tracking-by-detection | 关联策略（BYTE 二次关联） |
| 触觉 / Tactile | 自监督表征（JEPA） | 真机表征学习（LingBot）vs 仿真平台（TipSim） |

### 8.2 三条值得注意的技术主线 / Three notable threads

1. **"延迟"成为架构一级约束**：Qwen2.5-Omni 的 sliding-window DiT、FireRedTTS2 的 dual-transformer、CosyVoice 的流式、SenseVoice 的非自回归——都是为压低首包/端到端延迟而在架构层做的取舍。
*Latency is a first-class architectural constraint: sliding-window DiT, dual-transformer, streaming, non-autoregressive are all latency-driven design choices.*

2. **连续特征 vs 离散 token 的路线之争**：ASR 侧 FireRedASR-LLM 用连续 embedding + 轻量 adapter，TTS 侧两家都用离散 token + RVQ/FSQ——反映"理解任务保连续、生成任务需离散建模"的分工。
*Continuous vs discrete: ASR leans continuous embeddings, TTS leans discrete tokens — reflecting "understanding keeps continuous, generation needs discrete modeling".*

3. **强化学习进入感知模型训练**：Fun-ASR 的 FunRL（GRPO）、InternVL3.5 的 Cascade RL（MPO→GSPO）、CosyVoice 的 DiffRO——RL 正从 LLM 后训练扩散到感知模型。
*RL is spreading from LLM post-training into perception models: FunRL, Cascade RL, DiffRO.*

### 8.3 数据可信度提示 / Data reliability notes

- 本报告中 **"官方未披露"** 的项（如 FireRedTTS2 参数量、Qwen 各子模块参数、Fun-ASR 采样率）是有意保留的空缺，**不要用二手数字填补**。
*"Official-not-disclosed" items are deliberate gaps — do not fill them with secondary-source numbers.*
- 同一指标出现两个官方口径时（如 FireRedTTS2 延迟 140ms vs <100ms），报告并列标注，**引用时须注明来源**。
*When two official figures conflict, both are listed — cite with source.*

---

## 附：一手来源 / Appendix: First-Hand Sources

核实日期 2026-11-21；全部为官方 GitHub / arXiv / 官方博客。
All verified 2026-11-21 from official GitHub / arXiv / official blogs.

- HunyuanVideo：https://github.com/Tencent-Hunyuan/HunyuanVideo · arXiv:2412.03603
- Wan2.1：https://github.com/Wan-Video/Wan2.1 · arXiv:2503.20314
- CogVideoX：https://github.com/zai-org/CogVideo · arXiv:2408.06072
- Fun-ASR：https://github.com/QwenAudio/Fun-ASR · arXiv:2509.12508
- FireRedASR：https://github.com/FireRedTeam/FireRedASR · arXiv:2501.14350
- CosyVoice：https://github.com/FunAudioLLM/CosyVoice · arXiv:2505.17589
- FireRedTTS2：https://github.com/FireRedTeam/FireRedTTS2 · arXiv:2509.02020
- SenseVoice：https://github.com/QwenAudio/SenseVoice · arXiv:2407.04051
- FireRedChat：https://github.com/FireRedTeam/FireRedChat · arXiv:2509.06502
- Qwen2.5-Omni：https://github.com/QwenLM/Qwen2.5-Omni · arXiv:2503.20215 · https://qwenlm.github.io/blog/qwen2.5-omni/
- InternVL3.5：https://github.com/OpenGVLab/InternVL · arXiv:2508.18265 · https://internvl.github.io/blog/2025-08-26-InternVL-3.5/
- LingBot-Touch：https://github.com/Robbyant/lingbot-touch · arXiv:2611.12806
- TipSim：https://github.com/tiplab/TipSim
- PaddleVideo：https://github.com/PaddlePaddle/PaddleVideo · TSM arXiv:1811.08383
- ByteTrack：https://github.com/FoundationVision/ByteTrack · arXiv:2110.06864

> 免责声明 / Disclaimer：开源项目迭代快，架构与型号可能变化；落地前请再次核对官方仓库与论文最新版本。
*Projects iterate fast; re-check official repos and latest paper versions before deployment.*
