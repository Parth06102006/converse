# Machine Learning Model Architecture Reconnaissance

## 1. Overview and Latency Budget

Converse requires bidirectional real-time communication between signers and non-signers. The end-to-end latency budget across the system is constrained by human conversational turn-taking dynamics:

$$\text{Glass-to-Ear Latency: } T_{\text{capture}} + T_{\text{landmark}} + T_{\text{vision\_model}} + T_{\text{translation}} + T_{\text{tts}} + T_{\text{network}} \le 500\text{ms}$$

$$\text{Voice-to-Sign Latency: } T_{\text{audio\_capture}} + T_{\text{vad}} + T_{\text{asr}} + T_{\text{translation}} + T_{\text{sign\_render}} + T_{\text{network}} \le 400\text{ms}$$

This document analyzes model candidates across perception, sequence understanding, translation, ASR, and TTS to satisfy these real-time constraints.

---

## 2. Pose and Landmark Estimation

The perception frontend extracts physical joint positions from camera frames.

```
Raw RGB Frame (640x480x3) ──► Pose & Hand Estimator ──► Normalized 3D Coordinates
```

### 2.1 MediaPipe Holistic / Tasks Vision
- **Architecture**: Cascaded two-stage detection and keypoint regression networks:
  - BlazePose detector + 3D keypoint regressor (33 body keypoints).
  - BlazeHand detector + 21 keypoint 3D hand mesh regressor per hand.
  - Face Mesh regressor (468/478 landmarks).
- **Execution**: Optimized WebAssembly and CPU SIMD kernels; GPU acceleration via WebGL / WebGPU / OpenGL.
- **Latency**: 8ms to 15ms per frame on modern multi-core x86/ARM CPUs; < 6ms on WebGL.
- **Evaluation**:
  - *Pros*: Runs directly in the browser client or on lightweight backend CPU workers. Eliminates server video ingress. High developer ecosystem maturity and zero licensing cost (Apache 2.0).
  - *Cons*: Hand tracking jitter during fast ballistic signing movements; depth ($z$) coordinate is relative and subject to estimation noise; occlusions (hands crossing or touching face) cause momentary tracking loss.

### 2.2 OpenPose
- **Architecture**: Multi-stage bottom-up CNN predicting Part Affinity Fields (PAFs) and Confidence Maps for body, hands, and facial landmarks.
- **Execution**: C++ runtime with CUDA / cuDNN acceleration.
- **Latency**: 40ms to 90ms per frame on desktop GPUs (NVIDIA RTX 3080); unviable for real-time CPU execution (> 500ms per frame).
- **Evaluation**:
  - *Pros*: High accuracy for multi-person capture and complex anatomical poses; extensive research lineage.
  - *Cons*: High GPU memory footprint; severe compute overhead; restrictive academic license (GPLv3 with commercial fee exemptions required). Incompatible with client-side edge deployment.

### 2.3 MMPose / RTMPose
- **Architecture**: SimCC-based (Simple Coordinate Classification) lightweight real-time pose estimators developed by OpenMMLab.
- **Execution**: ONNX Runtime, TensorRT, CoreML.
- **Latency**: 5ms to 12ms per frame on CPU (RTMPose-m); 2ms on TensorRT.
- **Evaluation**:
  - *Pros*: Superior keypoint accuracy compared to MediaPipe on challenging occluded poses; highly optimized ONNX exports.
  - *Cons*: Requires separate hand crop pipeline; higher integration complexity than MediaPipe.

---

## 3. Sign Sequence and Recognition Models

Once landmarks or visual features are extracted, sequence models classify isolated signs or decode continuous sign sequences.

```
T Frames of Features (T x F) ──► Sequence Model ──► Sign Gloss / Linguistic Tokens
```

### 3.1 Inflated 3D Convolution (I3D)
- **Architecture**: 2D ImageNet convolutional networks (e.g. Inception-v1, ResNet) inflated into 3D spatio-temporal convolutions $(k_t \times k_h \times k_w)$ operating on RGB video volumes.
- **Input**: Tensor of shape $(B, C, T, H, W)$, typically $16\text{ or }32$ frames of $224 \times 224$ RGB crops.
- **Compute / Latency**: ~100 GFLOPs per clip; 35ms to 75ms inference latency per window on an NVIDIA T4 / RTX 3060.
- **Evaluation**:
  - *Pros*: High classification baseline on WLASL (Top-1 ~65% on WLASL-2000); captures raw visual nuances, skin deformation, and fine mouth cues.
  - *Cons*: High GPU memory requirements; requires raw video streaming from clients; difficult to deploy in serverless or low-cost infrastructure.

### 3.2 Spatial-Temporal Graph Convolutional Networks (ST-GCN / 2s-AGCN)
- **Architecture**: Constructs a spatiotemporal graph where skeleton joints form nodes, physical bones form spatial edges, and identical joints across adjacent frames form temporal edges. Adaptive Graph Convolutions learn dynamic adjacency matrices.
- **Input**: Tensor of shape $(B, C, T, V, M)$, where $C$ is coordinate dimensions (3), $T$ is temporal frames (typically 30-60), $V$ is number of joints (e.g. 54 or 75), and $M$ is number of bodies (1).
- **Compute / Latency**: ~1.5 GFLOPs; 4ms to 8ms inference latency on CPU; < 2ms on GPU.
- **Evaluation**:
  - *Pros*: Explicit anatomical inductive bias; invariant to background clutter, lighting shifts, and clothing; lightweight footprint suitable for real-time inference.
  - *Cons*: Relies entirely on upstream keypoint quality; completely blind to visual texture outside the skeletal graph (e.g. eye gaze, tongue, mouthing).

### 3.3 Temporal Convolutional Networks (TCN) and Conformer
- **Architecture**: 1D dilated casual convolutions stacked with multi-head self-attention and feed-forward modules.
- **Input**: Sequential feature vectors $(B, T, D_{\text{features}})$ representing flattened and normalized landmark coordinates.
- **Latency**: 6ms to 12ms on CPU.
- **Evaluation**:
  - *Pros*: Receptive field scales exponentially with dilation; parallel training without recurrence bottlenecks; low latency streaming capability with causal masking.
  - *Cons*: Requires extensive hyperparameter tuning for temporal window lengths.

### 3.4 Pose Transformers (Pose2Text / SPOT)
- **Architecture**: Encoder-decoder Transformer mapping continuous sequences of skeletal pose tokens directly to natural language embeddings or gloss tokens.
- **Compute / Latency**: 15ms to 35ms on CPU / GPU.
- **Evaluation**:
  - *Pros*: Learns long-range temporal dependencies; handles variable-length input sequences naturally; state-of-the-art on How2Sign translation benchmarks.
  - *Cons*: Requires substantial training data to avoid overfitting; sensitive to positional encoding strategies in streaming mode.

---

## 4. Automatic Speech Recognition (ASR)

For the speech-to-sign pipeline, the system requires fast, accurate transcription of spoken English into text.

```
Microphone Audio Stream (16kHz PCM) ──► Silero VAD ──► Streaming ASR ──► English Text
```

### 4.1 OpenAI Whisper (Faster-Whisper / CTranslate2)
- **Architecture**: Encoder-decoder Transformer trained on 680,000 hours of multilingual audio.
- **Variants**:
  - `whisper-tiny`: 39M parameters, ~60ms latency, higher WER.
  - `whisper-base`: 74M parameters, ~90ms latency, balanced accuracy.
  - `whisper-small`: 244M parameters, ~180ms latency, high accuracy.
- **Optimized Runtimes**: Faster-Whisper implemented via CTranslate2 with INT8 quantization.
- **Evaluation**:
  - *Pros*: Exceptional robustness to background noise, accents, and acoustic variation. Open source (MIT license).
  - *Cons*: Standard Whisper is designed for batch processing rather than native streaming; requires chunking strategy and Voice Activity Detection (VAD).

### 4.2 Streaming Conformer-CTC / Zipformer (Sherpa-ONNX)
- **Architecture**: Streaming Conformer with Connectionist Temporal Classification (CTC) loss, deployed via Sherpa-ONNX.
- **Latency**: Sub-100ms real-time factor ($RTF < 0.1$). Emits partial tokens every 160ms chunk.
- **Evaluation**:
  - *Pros*: True low-latency streaming without waiting for utterance end; minimal CPU footprint.
  - *Cons*: Slightly higher Word Error Rate on conversational speech with domain-specific terminology compared to Whisper.

### 4.3 Voice Activity Detection: Silero VAD
- **Architecture**: Deep learning-based enterprise-grade VAD module operating on 30ms audio chunks.
- **Latency**: < 1ms on a single CPU thread.
- **Role**: Emits speech start and speech end events to segment streaming audio before feeding into Whisper or Conformer, minimizing idle inference.

---

## 5. Text-to-Speech (TTS)

For the sign-to-speech pipeline, translated English text must be converted into clear, natural voice audio with minimal Time to First Byte (TTFB).

```
English Text Tokens ──► Neural TTS Synthesizer ──► Streaming PCM Audio (24kHz)
```

### 5.1 FastSpeech 2
- **Architecture**: Non-autoregressive Transformer with duration, pitch, and energy predictors feeding into a neural vocoder (HiFi-GAN).
- **Latency**: 25ms to 50ms total generation time for typical sentences; non-autoregressive parallel generation.
- **Evaluation**:
  - *Pros*: Extremely fast; deterministic inference time; avoids exposure bias of autoregressive models.
  - *Cons*: Requires offline phonemization and multi-stage checkpoint loading.

### 5.2 Piper / Coqui TTS (VITS / ONNX)
- **Architecture**: Conditional variational autoencoder with adversarial learning (VITS) running on ONNX Runtime.
- **Latency**: TTFB < 50ms on CPU; real-time factor ~0.08.
- **Evaluation**:
  - *Pros*: Ultra-lightweight C++/Python footprint; runs entirely locally on CPU without external API costs or cloud dependencies. High naturalness.
  - *Cons*: Voice customization requires targeted fine-tuning.

### 5.3 Microsoft Edge-TTS / Cloud TTS
- **Architecture**: Cloud-hosted neural speech synthesis accessed via streaming WebSocket protocol.
- **Latency**: TTFB 120ms to 200ms depending on network proximity.
- **Evaluation**:
  - *Pros*: Near-human prosody; zero server compute overhead; broad selection of natural voices.
  - *Cons*: Dependent on external internet connectivity and cloud API stability; introduces variable network latency.

---

## 6. Architecture Selection Matrix

| Subsystem | Selected Baseline | Alternative Under Evaluation | Selection Rationale |
|---|---|---|---|
| **Pose Estimation** | MediaPipe Holistic (Client Edge / CPU) | RTMPose (ONNX Server-Side) | Eliminates video streaming bandwidth; provides real-time 3D hand/body coordinates with low resource utilization. |
| **Isolated Sign Model** | ST-GCN / 2s-AGCN (Landmark Graph) | I3D (3D-CNN on RGB Crops) | Low computational complexity (4ms latency on CPU); background and lighting invariance. |
| **Continuous Translation** | Pose Transformer (Pose2Text) | TCN + MarianMT Translation | Natural modeling of long-range dependencies and direct continuous sequence mapping on How2Sign data. |
| **Speech Recognition** | Faster-Whisper (base/small) + Silero VAD | Sherpa-ONNX (Zipformer CTC) | Superior acoustic robustness and punctuation accuracy; CTranslate2 INT8 quantization meets latency budget. |
| **Speech Synthesis** | Piper / VITS (Local ONNX) | Cloud Edge-TTS (Streaming) | Zero cloud egress cost; predictable sub-60ms TTFB; completely offline capability. |
