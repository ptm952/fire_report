# Perception AI 精选开源项目调研报告（中英逐行对照）
# Curated Open-Source Perception AI Projects — A Line-by-Line Bilingual Research Report

> 调研日期 / Research date：2026-11-20
> 调研范围 / Scope：仅覆盖用户指定的 7 个方向、共 15 个项目；每个项目均给出详细介绍、官方地址与部署方案。
> *Scope: only the 7 directions and 15 projects specified by the user; each with a detailed introduction, official links, and deployment guide.*
> 可靠性说明 / Reliability note：所有信息均以官方 GitHub / Hugging Face / ModelScope 仓库与 LICENSE 为一手来源核实（2026-11-20），未采信未经核实的技术博客。
> *All information was verified on 2026-11-20 against first-hand sources — official GitHub / Hugging Face / ModelScope repositories and LICENSE files; unverified technical blogs were excluded.*

---

## 开源状态总览（务必先读）
## Open-Source Status Overview (Read This First)

> 你要求"项目一定得是开源的"。核实后发现：多数项目完全开源，但有几个是**部分开源或非标准许可**。本报告如实标注，绝不把受限项目包装成"完全开源"。
> *You required that every project must be open source. Verification shows most are fully open, but a few are **partially open or use non-standard licenses**. This report states the facts and never presents restricted projects as fully open.*

| 项目 / Project | 开源状态（中文 / English） |
|---|---|
| LingBot-Touch | 代码 Apache-2.0 开源；权重需账号授权 / Code Apache-2.0; weights gated |
| TipSim | ⚠️ 部分开源：核心插件闭源 / Partially open: core plugin closed-source |
| PaddleVideo | ✅ Apache-2.0 完全开源 / Fully open |
| ByteTrack | ✅ MIT 完全开源（含权重）/ Fully open incl. weights |
| HunyuanVideo | ⚠️ 权重公开但非标准许可 / Weights public, non-standard license |
| Wan2.1 | ✅ Apache-2.0 完全开源 / Fully open |
| CogVideoX1.5 | ⚠️ 代码 Apache；权重商用需登记 / Code Apache; weights need registration for commercial use |
| Fun-ASR-Nano | ✅ Apache-2.0 完全开源 / Fully open |
| FireRedASR | ✅ Apache-2.0 完全开源 / Fully open |
| CosyVoice 3.0 | ✅ Apache-2.0 完全开源 / Fully open |
| FireRedTTS | ⚠️ 一代 MPL-2.0，克隆限学术 / v1 MPL-2.0, cloning academic-only |
| FireRedChat | ⚠️ 部分开源，含非商用组件 / Partially open, includes non-commercial parts |
| SenseVoice | ⚠️ 源码 MIT；权重非标准许可 / Code MIT; weights non-standard license |
| Qwen2.5-Omni-7B | ⚠️ 代码 Apache；权重 Qwen 许可 / Code Apache; weights Qwen license |
| InternVL3.5 | ✅ 代码 MIT + 权重 Apache-2.0 / Code MIT + weights Apache-2.0 |

---

## 一、传感器 AI：触觉感知
## 1. Sensor AI: Tactile Perception

### 1.1 LingBot-Touch

**开发机构 / Developer**：蚂蚁灵波（Robbyant / Ant LingBot）。
*Developer: Ant LingBot (Robbyant).*

**开源状态 / Open-source status**：代码 Apache-2.0 完全开源；权重为"开放权重"，但在 Hugging Face / ModelScope 上需具备模型仓库权限的账号才能下载（gated access）。
*Code fully open under Apache-2.0; weights are "open weights" but gated — an authorized HF/ModelScope account is required to download.*

**官方地址 / Official links**：
*Official links:*
- GitHub：https://github.com/Robbyant/lingbot-touch
- 项目主页 / Homepage：https://technology.robbyant.com/lingbot-touch
- Hugging Face：https://huggingface.co/collections/robbyant/lingbot-touch
- ModelScope：https://www.modelscope.cn/collections/Robbyant/LingBot-Touch

**核心能力 / Core capabilities**：面向灵巧操作的通用触觉基础模型，统一处理磁触觉、压阻式、视触觉三类异构传感信号，输出可跨传感器、跨灵巧手复用的表征；基于约 5000 小时真实触觉数据预训练，可即插即用地接入下游动作模型（如 LingBot-VLA）。
*A general tactile foundation model for dexterous manipulation. It unifies three heterogeneous signal types — magnetic, piezoresistive, and visuotactile — into reusable cross-sensor representations, pretrained on ~5,000 hours of real tactile data and pluggable into downstream action models such as LingBot-VLA.*

**部署方案 / Deployment**：官方环境为 Linux / Python 3.12 / PyTorch 2.11 + CUDA 12.10。
*Official environment: Linux / Python 3.12 / PyTorch 2.11 + CUDA 12.10.*

```bash
conda create -n lingbot-touch python=3.12.13 pip=26.0.1 -y && conda activate lingbot-touch
python -m pip install "torch==2.11.0" --index-url https://download.pytorch.org/whl/cu130
python -m pip install . && python -m pip check
```
> 到 HF / ModelScope 下载 `config.json` 与 `encoder.pt` 放入 `checkpoints/lingbot-touch-tiny/`（或 `-small/`），再用 `infer.py` 推理，CPU 用 `--device cpu`，GPU 用 `--device cuda:0`。
> *Download `config.json` and `encoder.pt` into `checkpoints/lingbot-touch-tiny/` (or `-small/`), then run `infer.py`; use `--device cpu` or `--device cuda:0`. 提供 Tiny / Small 两个权重版本 / Two weight sizes: Tiny and Small.*

### 1.2 TipSim

**开发机构 / Developer**：华威科（tiplab，TIP 触觉智能平台）。
*Developer: Huaweike (tiplab, TIP tactile intelligence platform).*

**开源状态 / Open-source status**：⚠️ **部分开源**——场景 XML、Python/C++ 示例、文档为 MIT；但核心触觉传感器实现是 `lib/*.so` 预编译二进制插件，明确不在 MIT 授权范围内，禁止反编译与单独再分发。
*⚠️ **Partially open** — scenes, examples, and docs are MIT; but the core sensor implementation is a precompiled `lib/*.so` binary plugin explicitly outside the MIT grant, with no reverse engineering or redistribution allowed.*

**官方地址 / Official links**：
*Official links:*
- GitHub：https://github.com/tiplab/TipSim
- 官网 / Website：https://www.hbwhhit.com

**核心能力 / Core capabilities**：基于 MuJoCo 的机器人触觉仿真平台，提供 Loong / Universa 系列传感器插件，按 taxel 阵列连续输出接触位置与受力分布，反馈可直接接入抓取控制回路，支持 XML 配置量程与触发力、固定随机种子以保证可复现，面向 Sim2Real 研究。
*A MuJoCo-based robotic tactile simulation platform with Loong/Universa sensor plugins that output contact position and force-distribution taxel arrays, feed directly into grasping control loops, and support reproducible, configurable Sim2Real research.*

**部署方案 / Deployment**：仅支持 Linux x86_64，需 MuJoCo 3.9.0–3.16.0；运行示例另需 ROS2 / GLFW。
*Linux x86_64 only; requires MuJoCo 3.9.0–3.16.0; ROS2/GLFW for examples.*

```bash
git clone https://github.com/tiplab/TipSim.git
```
> 三步集成 / Three integration steps：① 加载插件 `mujoco.mj_loadPluginLibrary("lib/libtip_tactile_loong.so")`；② 在场景 XML 的 `<plugin>` 中声明传感器阵列；③ 在 `mj_step()` 后从 `mjData.sensordata` 读取阵列数据。
> *① Load plugin; ② declare the sensor array in the scene XML `<plugin>`; ③ read the array from `mjData.sensordata` after `mj_step()`. 仓库附带 RH56E2 灵巧手抓取、UR5e 取放等场景 / Includes RH56E2 grasping and UR5e pick-and-place scenarios.*

---

## 二、视频感知：视频理解与分析
## 2. Video Perception: Video Understanding & Analysis

### 2.1 PaddleVideo

**开发机构 / Developer**：百度飞桨（PaddlePaddle）。
*Developer: Baidu PaddlePaddle.*

**开源状态 / Open-source status**：✅ Apache-2.0，源码、模型、文档完全开源。
*✅ Apache-2.0; source, models, and docs fully open.*

**官方地址 / Official links**：https://github.com/PaddlePaddle/PaddleVideo

**核心能力 / Core capabilities**：飞桨官方视频理解工具库，覆盖视频标注（BILS）、行为识别（自研 PP-TSM / PP-TSMv2，及 TSN、SlowFast、TimeSformer、VideoSwin、TokenShift 等）、骨骼点行为识别（ST-GCN、CTR-GCN）、时序动作检测（BMN、YOWO）等，打通"数据生产—训练—压缩—部署"全流程。PP-TSMv2 在 Kinetics-400 达 75.16%，10s 视频 CPU 推理仅 456ms。
*Paddle's official video-understanding toolkit covering annotation (BILS), action recognition (PP-TSM/PP-TSMv2, TSN, SlowFast, TimeSformer, etc.), skeleton-based recognition, and temporal action detection, with a full data-to-deployment pipeline. PP-TSMv2 reaches 75.16% on Kinetics-400 at 456ms per 10s clip on CPU.*

**部署方案 / Deployment**：Python ≥3.7，仅 Linux，PaddlePaddle ≥2.0；GPU 需 CUDA ≥10.1 / cuDNN ≥7.6.4。
*Python ≥3.7, Linux only, PaddlePaddle ≥2.0; GPU needs CUDA ≥10.1 / cuDNN ≥7.6.4.*

```bash
pip3 install paddlepaddle-gpu --upgrade
git clone https://github.com/PaddlePaddle/PaddleVideo.git && cd PaddleVideo
pip3 install --upgrade -r requirements.txt
pip3 install ppvideo==2.3.0
ppvideo --model_name='ppTSM_v2' --use_gpu=False --video_file='data/example.avi'
```
> 亦支持 Python/C++ 预测引擎、服务端部署、Paddle2ONNX 转换。
*Also supports Python/C++ inference engines, server deployment, and Paddle2ONNX conversion.*

### 2.2 ByteTrack

**开发机构 / Developer**：华中科技大学 + 香港大学 + 字节跳动（ECCV 2022）。
*Developer: HUST + HKU + ByteDance (ECCV 2022).*

**开源状态 / Open-source status**：✅ MIT，代码、预训练权重、训练与部署脚本全部开源。
*✅ MIT; code, pretrained weights, training and deployment scripts fully open.*

**官方地址 / Official links**：
*Official links:*
- 当前主仓库 / Main repo：https://github.com/FoundationVision/ByteTrack
- 旧地址（自动跳转）/ Old URL (redirects)：https://github.com/ifzhang/ByteTrack

**核心能力 / Core capabilities**：经典多目标跟踪（MOT）算法，核心 BYTE 策略关联几乎全部检测框（含低分框），通过二次关联恢复遮挡目标、过滤背景，无需 ReID 模块。MOT17 上达 80.3 MOTA / 77.3 IDF1 / 63.1 HOTA，单卡 V100 约 30 FPS。
*A classic MOT algorithm whose BYTE strategy associates nearly all detections (including low-score boxes) via a second association pass to recover occluded objects without ReID. Achieves 80.3 MOTA / 77.3 IDF1 / 63.1 HOTA on MOT17 at ~30 FPS on one V100.*

**部署方案 / Deployment**：依赖 PyTorch / torchvision，推荐 NVIDIA GPU（支持 `--fp16`）。
*Requires PyTorch/torchvision; NVIDIA GPU recommended (supports `--fp16`).*

```bash
git clone https://github.com/ifzhang/ByteTrack.git && cd ByteTrack
pip3 install -r requirements.txt && python3 setup.py develop
pip3 install cython_bbox
python3 tools/demo_track.py video -f exps/example/mot/yolox_x_mix_det.py \
  -c pretrained/bytetrack_x_mot17.pth.tar --fp16 --fuse --save_result
```
> 另提供 Docker、ONNXRuntime、TensorRT、ncnn、DeepStream 多种部署版本。
*Also ships Docker, ONNXRuntime, TensorRT, ncnn, and DeepStream deployment variants.*

---

## 三、视频感知：视频生成
## 3. Video Perception: Video Generation

### 3.1 HunyuanVideo

**开发机构 / Developer**：腾讯混元（Tencent Hunyuan，13B）。
*Developer: Tencent Hunyuan (13B).*

**开源状态 / Open-source status**：⚠️ 代码与权重公开，但采用非 OSI 标准的 Tencent Hunyuan Community License：授权地域排除欧盟、英国、韩国；月活超 1 亿需另行授权；禁止用于改进其他 AI 模型。
*⚠️ Code and weights public, but under the non-OSI Tencent Hunyuan Community License: excludes EU/UK/Korea; >100M MAU requires a separate license; forbidden for improving other AI models.*

**官方地址 / Official links**：
*Official links:*
- GitHub：https://github.com/Tencent-Hunyuan/HunyuanVideo
- Hugging Face：https://huggingface.co/tencent/HunyuanVideo

**核心能力 / Core capabilities**：13B 参数视频生成模型，采用 MLLM 文本编码器与 3D VAE，支持 540p / 720p、固定 129 帧（50 步推理），官方称效果优于 Runway Gen-3、Luma 1.6 等。轻量版 HunyuanVideo 1.5（8.3B，SSTA 稀疏注意力，最低 14GB 显存）另在 https://github.com/Tencent-Hunyuan/HunyuanVideo-1.5 。
*13B video generation model with MLLM text encoder and 3D VAE, supporting 540p/720p at 129 frames (50 steps). A lighter 8.3B HunyuanVideo 1.5 (14GB minimum VRAM) is at the linked repo.*

**部署方案 / Deployment**：推荐单卡 80GB；540p 峰值约 45GB，720p 约 60GB；可用 FP8 权重与 CPU offload 省显存。
*Single 80GB GPU recommended; ~45GB peak at 540p, ~60GB at 720p; FP8 weights and CPU offload reduce VRAM.*

```bash
git clone https://github.com/Tencent-Hunyuan/HunyuanVideo && cd HunyuanVideo
pip install -r requirements.txt
python3 sample_video.py --video-size 720 1280 --video-length 129 --infer-steps 50 \
  --prompt "A cat walks on the grass, realistic style." \
  --flow-reverse --use-cpu-offload --save-path ./results
```

### 3.2 Wan2.1（通义万相 / Tongyi Wanxiang）

**开发机构 / Developer**：阿里巴巴 Wan 团队（通义万相）。
*Developer: Alibaba Wan Team (Tongyi Wanxiang).*

**开源状态 / Open-source status**：✅ Apache-2.0，代码与权重全部公开、无门禁，是三家视频生成里授权最宽松者。
*✅ Apache-2.0; code and weights fully public with no gating — the most permissive of the three video generators.*

**官方地址 / Official links**：
*Official links:*
- GitHub：https://github.com/Wan-Video/Wan2.1
- Hugging Face（14B）：https://huggingface.co/Wan-AI/Wan2.1-T2V-14B

**核心能力 / Core capabilities**：含 T2V-14B（480P/720P）、T2V-1.3B、I2V-14B、VACE 等；DiT 采用 Flow Matching + T5 多语言编码，支持中英文文字生成，Wan-VAE 可编解码任意长度 1080P 视频；典型输出 81 帧、16fps（约 5 秒）。
*Includes T2V-14B (480P/720P), T2V-1.3B, I2V-14B, and VACE; DiT with Flow Matching and T5 multilingual encoding, Chinese/English text rendering, and a Wan-VAE for arbitrary-length 1080P; typical output 81 frames at 16fps (~5s).*

**部署方案 / Deployment**：需 torch ≥2.4.0；1.3B 仅需 8.19GB 显存（RTX 4090 约 4 分钟生成 5 秒 480P），14B 建议 offload。
*Requires torch ≥2.4.0; 1.3B needs only 8.19GB VRAM (~4 min for 5s 480P on RTX 4090); 14B recommends offload.*

```bash
git clone https://github.com/Wan-Video/Wan2.1.git && cd Wan2.1
pip install -r requirements.txt
huggingface-cli download Wan-AI/Wan2.1-T2V-1.3B --local-dir ./Wan2.1-T2V-1.3B
python generate.py --task t2v-1.3B --size 832*480 --ckpt_dir ./Wan2.1-T2V-1.3B \
  --offload_model True --t5_cpu --sample_shift 8 --sample_guide_scale 6 --prompt "..."
```

### 3.3 CogVideoX 1.5

**开发机构 / Developer**：智谱 AI（Zhipu AI，原清华 THUDM 团队，国际品牌 zai-org）。
*Developer: Zhipu AI (formerly THUDM, Tsinghua; international brand zai-org).*

**开源状态 / Open-source status**：⚠️ 代码 Apache-2.0；但 CogVideoX1.5-5B 权重为非标准 CogVideoX License：学术免费，商用须先登记取得授权、且月访问量不超过 100 万次。
*⚠️ Code Apache-2.0; but CogVideoX1.5-5B weights use the non-standard CogVideoX License: free for academia, commercial use requires prior registration and is capped at 1M monthly visits.*

**官方地址 / Official links**：
*Official links:*
- GitHub：https://github.com/zai-org/CogVideo （旧 / old: https://github.com/THUDM/CogVideo）
- Hugging Face：https://huggingface.co/THUDM/CogVideoX1.5-5B

**核心能力 / Core capabilities**：5B 参数，分辨率 1360×768，帧数须为 16N+1（默认 81 帧），5 秒或 10 秒、16fps，推荐 BF16；I2V 版支持任意分辨率。
*5B parameters, 1360×768 resolution, frame count 16N+1 (default 81), 5s or 10s at 16fps, BF16 recommended; I2V supports arbitrary resolutions.*

**部署方案 / Deployment**：diffusers BF16 约需 9–10GB 显存（多卡 24GB），INT8 约 7GB。
*~9–10GB VRAM with diffusers BF16 (24GB multi-GPU), ~7GB with INT8.*

```bash
pip install --upgrade transformers accelerate diffusers imageio-ffmpeg
```
```python
import torch
from diffusers import CogVideoXPipeline
from diffusers.utils import export_to_video
pipe = CogVideoXPipeline.from_pretrained("THUDM/CogVideoX1.5-5B", torch_dtype=torch.bfloat16)
pipe.enable_sequential_cpu_offload()
pipe.vae.enable_tiling(); pipe.vae.enable_slicing()
video = pipe(prompt="...", num_inference_steps=50, num_frames=81, guidance_scale=6).frames[0]
export_to_video(video, "output.mp4", fps=8)
```

---

## 四、声音 / 语音感知：ASR（语音识别）
## 4. Sound / Speech Perception: ASR (Automatic Speech Recognition)

### 4.1 Fun-ASR-Nano

**开发机构 / Developer**：阿里巴巴通义实验室（FunAudioLLM / QwenAudio 团队）。
*Developer: Alibaba Tongyi Lab (FunAudioLLM / QwenAudio).*

**开源状态 / Open-source status**：✅ Apache-2.0，源码与权重完全开源。注意与 Qwen3-ASR、SenseVoice、Paraformer 是不同项目，勿混淆。
*✅ Apache-2.0; source and weights fully open. Note it is a distinct project from Qwen3-ASR, SenseVoice, and Paraformer.*

**官方地址 / Official links**：
*Official links:*
- GitHub：https://github.com/FunAudioLLM/Fun-ASR （现跳转 / now redirects to：https://github.com/QwenAudio/Fun-ASR）
- Hugging Face：https://huggingface.co/FunAudioLLM/Fun-ASR-Nano-2512
- ModelScope：https://modelscope.cn/models/FunAudioLLM/Fun-ASR-Nano-2512

**核心能力 / Core capabilities**：0.8B 端到端 LLM 语音识别，支持中/英/日，覆盖 7 大方言与 26 种地域口音，并具歌词/说唱识别能力；远场高噪声场景准确率宣称 93%，vLLM 批处理 RTFx 达 340x（较原生 PyTorch 约 16x 加速，CER 差异 <0.2%）；AIShell1 WER 1.80。
*A 0.8B end-to-end LLM ASR supporting Chinese/English/Japanese, 7 major dialects and 26 accents, plus lyrics/rap recognition; ~93% accuracy claimed in noisy far-field settings, RTFx 340x with vLLM, and 1.80 WER on AIShell1.*

**部署方案 / Deployment**：Python ≥3.8，需 PyTorch + torchaudio；无 GPU 可用 GGUF / llama.cpp。
*Python ≥3.8 with PyTorch+torchaudio; GGUF/llama.cpp for CPU-only.*

```bash
pip install -U "funasr==1.4.15"
pip install "funasr>=1.3.26" "vllm>=0.12.0"   # 批量 / batch inference
```
```python
from funasr import AutoModel
model = AutoModel(model="FunAudioLLM/Fun-ASR-Nano-2512", device="cuda:0", hub="hf")
print(model.generate(input=["audio.wav"], hotwords=["开放时间"], language="中文")[0]["text"])
```

### 4.2 FireRedASR

**开发机构 / Developer**：小红书 FireRed 团队（FireRedTeam）。
*Developer: Xiaohongshu FireRed Team.*

**开源状态 / Open-source status**：✅ Apache-2.0，代码与权重完全开源；LLM 版需另配 Qwen2-7B-Instruct 基座。
*✅ Apache-2.0; code and weights fully open; the LLM variant additionally requires the Qwen2-7B-Instruct base.*

**官方地址 / Official links**：
*Official links:*
- GitHub：https://github.com/FireRedTeam/FireRedASR
- Hugging Face：https://huggingface.co/fireredteam
- Demo：https://fireredteam.github.io/demos/firered_asr/

**核心能力 / Core capabilities**：工业级普通话/方言/英语 ASR，含突出的歌词识别。两变体：AED（1.1B，效率优先）与 LLM（8.3B，SOTA 优先）。普通话平均 CER：LLM 3.05 / AED 3.18，优于 Seed-ASR(3.33) 与 Whisper(9.86)；方言 KeSpeech 3.56，LibriSpeech test-clean WER 1.73。
*Industrial-grade Mandarin/dialect/English ASR with strong lyric recognition. Two variants: AED (1.1B, efficiency) and LLM (8.3B, SOTA). Average Mandarin CER 3.05/3.18, beating Seed-ASR (3.33) and Whisper (9.86).*

**部署方案 / Deployment**：Python 3.10，输入须转为 16kHz/16bit 单声道 wav；AED ≤60s，LLM ≤30s。
*Python 3.10; input must be 16kHz/16bit mono wav; AED ≤60s, LLM ≤30s.*

```bash
git clone https://github.com/FireRedTeam/FireRedASR.git
conda create --name fireredasr python=3.10 && conda activate fireredasr
pip install -r requirements.txt
export PATH=$PWD/fireredasr/:$PWD/fireredasr/utils/:$PATH
export PYTHONPATH=$PWD/:$PYTHONPATH
ffmpeg -i input -ar 16000 -ac 1 -acodec pcm_s16le -f wav output.wav
bash examples/inference_fireredasr_aed.sh
```
> 权重放入 `pretrained_models/`；LLM 版需软链 `ln -s ../Qwen2-7B-Instruct`。
*Place weights in `pretrained_models/`; LLM variant needs `ln -s ../Qwen2-7B-Instruct`.*

---

## 五、声音 / 语音感知：TTS（语音合成）
## 5. Sound / Speech Perception: TTS (Text-to-Speech)

### 5.1 CosyVoice 3.0

**开发机构 / Developer**：阿里巴巴通义实验室（FunAudioLLM / QwenAudio 团队）。
*Developer: Alibaba Tongyi Lab (FunAudioLLM / QwenAudio).*

**开源状态 / Open-source status**：✅ Apache-2.0，完全开源。官方命名实为 **Fun-CosyVoice 3.0**，权重 `Fun-CosyVoice3-0.5B-2512`；注意 CosyVoice 3.5 仅有 API、无开源权重。
*✅ Apache-2.0, fully open. The official name is **Fun-CosyVoice 3.0** with weights `Fun-CosyVoice3-0.5B-2512`; note CosyVoice 3.5 is API-only with no open weights.*

**官方地址 / Official links**：
*Official links:*
- GitHub：https://github.com/FunAudioLLM/CosyVoice
- Hugging Face：https://huggingface.co/FunAudioLLM/Fun-CosyVoice3-0.5B-2512
- Demo：https://funaudiollm.github.io/cosyvoice3/

**核心能力 / Core capabilities**：基于 LLM 的零样本多语种 TTS，3 秒参考音频即可克隆音色，覆盖 9 语言 + 18 种中文方言/口音，双向流式延迟低至 150ms，支持语言/方言/情感/语速的 Instruct 控制；test-zh CER 1.21（RL 版 0.81）。
*LLM-based zero-shot multilingual TTS cloning from 3s reference audio, covering 9 languages and 18 Chinese dialects, with bidirectional streaming down to 150ms and instruct control; test-zh CER 1.21 (0.81 with RL).*

**部署方案 / Deployment**：Python 3.10，建议 `--recursive` 克隆。
*Python 3.10; clone with `--recursive`.*

```bash
git clone --recursive https://github.com/FunAudioLLM/CosyVoice.git && cd CosyVoice
conda create -n cosyvoice -y python=3.10 && conda activate cosyvoice
pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host=mirrors.aliyun.com
```
```python
from modelscope import snapshot_download
snapshot_download('FunAudioLLM/Fun-CosyVoice3-0.5B-2512', local_dir='pretrained_models/Fun-CosyVoice3-0.5B')
```
> 推理 `python example.py`；WebUI `python3 webui.py --port 50000 --model_dir pretrained_models/Fun-CosyVoice3-0.5B`。
*Inference via `python example.py`; WebUI via `python3 webui.py --port 50000 --model_dir ...`.*

### 5.2 FireRedTTS

**开发机构 / Developer**：小红书 FireRed 团队（FireRedTeam）。
*Developer: Xiaohongshu FireRed Team.*

**开源状态 / Open-source status**：⚠️ 两代许可不同——一代（FireRedTTS-1S）为 MPL-2.0，且官方声明零样本克隆"仅限学术研究"；二代（FireRedTTS2）为 Apache-2.0，更开放，推荐优先采用。
*⚠️ Licenses differ by generation — v1 (FireRedTTS-1S) is MPL-2.0 with zero-shot cloning "for academic research only"; v2 (FireRedTTS2) is Apache-2.0 and more permissive — recommended.*

**官方地址 / Official links**：
*Official links:*
- 一代 v1：https://github.com/FireRedTeam/FireRedTTS
- 二代 v2：https://github.com/FireRedTeam/FireRedTTS2
- 权重 v2：https://huggingface.co/FireRedTeam/FireRedTTS2

**核心能力 / Core capabilities**：面向播客/聊天机器人的长文本多说话人流式 TTS。二代 1.5B，支持最长 3 分钟、4 说话人对话、7 语言；12.5Hz 流式 tokenizer + 双 Transformer 实现句级生成，L20 上首包延迟低至 140ms，bf16 显存由 14GB 降至 9GB（消费级可部署）。
*Long-form multi-speaker streaming TTS for podcasts/chatbots. v2 is 1.5B, supporting 3-minute 4-speaker dialogue in 7 languages, with a 12.5Hz streaming tokenizer and dual Transformer; first-packet latency 140ms on L20 and bf16 VRAM cut from 14GB to 9GB.*

**部署方案 / Deployment（二代 v2，Python 3.11 + CUDA 12.1）**：
*Deployment (v2, Python 3.11 + CUDA 12.1):*

```bash
git clone https://github.com/FireRedTeam/FireRedTTS2.git && cd FireRedTTS2
conda create --name fireredtts2 python==3.11 && conda activate fireredtts2
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu126
pip install -e . && pip install -r requirements.txt
git lfs install && git clone https://huggingface.co/FireRedTeam/FireRedTTS2 pretrained_models/FireRedTTS2
python gradio_demo.py --pretrained-dir "./pretrained_models/FireRedTTS2"
```
> 商用注意：二代 demo 中特定主播音色未经授权不得使用，须自行确保声音来源授权。
*Commercial note: the specific demo voices are unauthorized for use; ensure you have the rights to any reference voice.*

---

## 六、全双工语音交互与音频理解
## 6. Full-Duplex Speech Interaction & Audio Understanding

### 6.1 FireRedChat

**开发机构 / Developer**：小红书 FireRed 团队（FireRedTeam）。
*Developer: Xiaohongshu FireRed Team.*

**开源状态 / Open-source status**：⚠️ 部分开源——主框架与 ASR 代码 Apache-2.0，但 FireRedTTS 代码 MPL-2.0、其权重与内置音色明确非商用，且 Context-Aware TTS / Audio LLM 仍标 "Coming soon"，并非完整全开源。
*⚠️ Partially open — main framework and ASR code are Apache-2.0, but FireRedTTS code is MPL-2.0 with non-commercial weights/voices, and Context-Aware TTS / Audio LLM remain "Coming soon" — not fully open.*

**官方地址 / Official links**：https://github.com/FireRedTeam/FireRedChat

**核心能力 / Core capabilities**：自托管全双工语音交互系统，集成 FireRedTTS（含文本正则化/G2P）、FireRedASR（含标点模型）、个性化 VAD（pVAD）与端到端轮次检测（EoT，中英双语），基于 LiveKit RTC + Redis + WebUI，LLM 由用户自接 Ollama / vLLM / Dify。
*A self-hosted full-duplex speech system integrating FireRedTTS, FireRedASR, personalized VAD, and end-of-turn detection, built on LiveKit RTC + Redis + WebUI, with a user-supplied LLM via Ollama/vLLM/Dify.*

**部署方案 / Deployment**：依赖 Docker / Docker Compose 与 GPU（承载 ASR/TTS）。
*Requires Docker/Docker Compose and a GPU for ASR/TTS.*

```bash
git clone --recurse-submodules https://github.com/FireRedTeam/FireRedChat.git
cd docker && docker-compose up -d      # RTC:7880, WebUI:3000
curl -fsSL https://ollama.com/install.sh | sh && ollama run qwen2.5
```

### 6.2 SenseVoice

**开发机构 / Developer**：阿里 FunAudioLLM / QwenAudio 团队。
*Developer: Alibaba FunAudioLLM / QwenAudio.*

**开源状态 / Open-source status**：⚠️ 源码 MIT；但 SenseVoiceSmall 权重采用 FunASR Model Open Source License（非标准，允许商用但须保留署名与模型名）。官方仅开源 SenseVoiceSmall，未开放 SenseVoice-L 权重。
*⚠️ Source code MIT; but SenseVoiceSmall weights use the FunASR Model Open Source License (non-standard, commercial use allowed with attribution). Only SenseVoiceSmall is released; SenseVoice-L weights are not public.*

**官方地址 / Official links**：
*Official links:*
- GitHub：https://github.com/QwenAudio/SenseVoice （旧 / old：https://github.com/FunAudioLLM/SenseVoice）
- Hugging Face：https://huggingface.co/FunAudioLLM/SenseVoiceSmall
- ModelScope：https://www.modelscope.cn/models/iic/SenseVoiceSmall

**核心能力 / Core capabilities**：非自回归端到端语音理解模型，单次推理同时输出 ASR 文本、语种 ID、情感标签（SER）与声音事件标签（AED，如掌声/笑声/哭声/咳嗽）。比 Whisper-Small 快 5 倍以上、比 Whisper-Large 快 15 倍，中文/粤语识别优于 Whisper。
*A non-autoregressive speech understanding model outputting ASR text, language ID, emotion (SER), and audio events (applause/laughter/crying/coughing) in one pass; 5× faster than Whisper-Small and 15× faster than Whisper-Large, with better Chinese/Cantonese recognition.*

**部署方案 / Deployment**：建议 funasr 1.4.14；支持 ONNX / Libtorch 导出、FastAPI 服务、GGUF 无 GPU 运行。
*FunASR 1.4.14 recommended; supports ONNX/Libtorch export, FastAPI serving, and GGUF CPU runtime.*

```bash
pip install -r requirements.txt
```
```python
from funasr import AutoModel
model = AutoModel(model="iic/SenseVoiceSmall", trust_remote_code=True,
                  remote_code="./model.py", vad_model="fsmn-vad", device="cuda:0")
res = model.generate(input="example/en.mp3", language="auto", use_itn=True)
```

---

## 七、全模态大模型
## 7. Omni-Modal Large Models

### 7.1 Qwen2.5-Omni-7B

**开发机构 / Developer**：阿里云通义千问团队（Qwen Team, Alibaba Cloud）。
*Developer: Qwen Team, Alibaba Cloud.*

**开源状态 / Open-source status**：⚠️ 代码 Apache-2.0；但模型权重为 Qwen Research License（非无条件的 Apache-2.0），商用前须逐条核对模型卡。权重非 gated，可直接下载。
*⚠️ Code Apache-2.0; but weights are under the Qwen Research License (not unconditional Apache-2.0) — review the model card before commercial use. Weights are non-gated and directly downloadable.*

**官方地址 / Official links**：
*Official links:*
- GitHub：https://github.com/QwenLM/Qwen2.5-Omni
- Hugging Face：https://huggingface.co/Qwen/Qwen2.5-Omni-7B
- ModelScope：https://modelscope.cn/collections/Qwen25-Omni-a2505ce0d5514e

**核心能力 / Core capabilities**：端到端真·全模态模型，采用 Thinker-Talker 双核架构与 TMRoPE，输入文本/图像/音频/视频，流式输出文本与自然语音。OmniBench（Speech/Sound/Music/Avg）7B 达 55.25/60.00/52.83/56.13，为当时开源 SOTA；MMMU val 59.2，语音指令跟随接近文本输入。
*An end-to-end true omni-modal model with Thinker-Talker dual-core architecture and TMRoPE, taking text/image/audio/video input and streaming text plus natural speech. OmniBench 7B: 55.25/60.00/52.83/56.13, an open-source SOTA at release; MMMU val 59.2.*

**部署方案 / Deployment**：需 transformers==4.52.3；7B BF16 处理 15s/30s/60s 视频分别需 31/42/60GB；4-bit（GPTQ-Int4/AWQ）可降至约 11.6GB，可在 RTX 3080/4080/5070 运行。
*Requires transformers==4.52.3; 7B BF16 needs 31/42/60GB for 15s/30s/60s video; 4-bit (GPTQ-Int4/AWQ) drops to ~11.6GB, runnable on RTX 3080/4080/5070.*

```bash
pip install "transformers==4.52.3" accelerate
pip install "vllm>=0.8.5.post1"     # vLLM serve 当前仅文本输出 / currently text output only
```
```python
from transformers import Qwen2_5OmniForConditionalGeneration
model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-Omni-7B", torch_dtype="auto", device_map="auto")
```

### 7.2 InternVL3.5（书生·万象3.5）

**开发机构 / Developer**：上海人工智能实验室 OpenGVLab 团队。
*Developer: OpenGVLab, Shanghai AI Laboratory.*

**开源状态 / Open-source status**：✅ 全系列 9 种尺寸开放、非 gated；代码 MIT、权重 Apache-2.0。
*✅ All 9 sizes released, non-gated; code MIT, weights Apache-2.0.*

**官方地址 / Official links**：
*Official links:*
- GitHub：https://github.com/OpenGVLab/InternVL
- 博客 / Blog：https://internvl.github.io/blog/2025-08-26-InternVL-3.5/
- Hugging Face（旗舰 241B-A28B）：https://huggingface.co/OpenGVLab/InternVL3_5-241B-A28B

**核心能力 / Core capabilities**：以视觉为主的多模态大模型（文本/图像/视频，无音频输入输出），9 种尺寸（1B–241B，旗舰激活 28B）。采用 Cascade RL、视觉分辨率路由 ViR 与视觉-语言解耦部署 DvD；MMMU 上 8B 得 73.4、241B-A28B 得 77.7（开源模型最高），推理较 InternVL3 最高提升 16.0%、加速 4.05×。
*A vision-centric multimodal model (text/image/video, no audio I/O) in 9 sizes (1B–241B). Uses Cascade RL, Visual Resolution Routing (ViR), and Disentangled visual-language Deployment (DvD); MMMU 73.4 (8B) and 77.7 (241B-A28B, open-source best), with up to +16.0% reasoning gain and 4.05× speedup.*

**部署方案 / Deployment**：推荐 LMDeploy；8B 单卡 80G A100 可全量加载，旗舰 241B-A28B 需 `tp=8`。
*LMDeploy recommended; 8B fits one 80GB A100, flagship 241B-A28B needs `tp=8`.*

```bash
pip install lmdeploy
lmdeploy serve api_server OpenGVLab/InternVL3_5-8B \
  --server-port 23333 --tp 1 --backend pytorch
```

---

## 八、选型与合规要点
## 8. Selection & Compliance Notes

> 完全开源（可较自由商用）：PaddleVideo、ByteTrack、Wan2.1、Fun-ASR-Nano、FireRedASR、CosyVoice 3.0、FireRedTTS2、InternVL3.5。
*Fully open (relatively free for commercial use): PaddleVideo, ByteTrack, Wan2.1, Fun-ASR-Nano, FireRedASR, CosyVoice 3.0, FireRedTTS2, InternVL3.5.*

> 非标准许可或需授权：HunyuanVideo（地域/MAU 限制）、CogVideoX1.5（商用登记）、SenseVoice（署名要求）、Qwen2.5-Omni-7B（Qwen 权重许可）。
*Non-standard licenses or authorization needed: HunyuanVideo (region/MAU limits), CogVideoX1.5 (commercial registration), SenseVoice (attribution), Qwen2.5-Omni-7B (Qwen weight license).*

> 部分开源：TipSim（核心插件闭源）、FireRedChat（含非商用组件）、FireRedTTS 一代（克隆限学术）、LingBot-Touch（权重需授权）。
*Partially open: TipSim (core plugin closed), FireRedChat (non-commercial components), FireRedTTS v1 (cloning academic-only), LingBot-Touch (gated weights).*

> 选型建议：极致中文 ASR 选 FireRedASR；中文 TTS 选 CosyVoice 3.0；播客/多说话人选 FireRedTTS2；视频生成选 Wan2.1（许可最宽松）；真全模态选 Qwen2.5-Omni；视觉多模态选 InternVL3.5。
*Selection guide: for top Chinese ASR choose FireRedASR; Chinese TTS CosyVoice 3.0; podcasts/multi-speaker FireRedTTS2; video generation Wan2.1 (most permissive); true omni-modal Qwen2.5-Omni; vision-centric multimodal InternVL3.5.*

---

## 附：信息来源与免责声明
## Appendix: Sources & Disclaimer

> 来源 / Sources：各项目官方 GitHub README 与 LICENSE 文件、官方 Hugging Face / ModelScope 模型卡（核实日期 2026-11-20）。文中具体 URL 均已逐一列出。
*Sources: each project's official GitHub README and LICENSE, official Hugging Face / ModelScope model cards (verified 2026-11-20). All URLs are listed inline above.*

> 免责声明 / Disclaimer：开源项目迭代快，许可证与型号命名可能变化；商用落地前请再次打开对应官方仓库与模型卡核对最新条款，尤其注意音色授权、地域限制与署名要求。
*Disclaimer: open-source projects iterate quickly and licenses/names may change; before commercial deployment, re-check the latest terms in the official repositories and model cards, especially voice rights, regional restrictions, and attribution requirements.*
