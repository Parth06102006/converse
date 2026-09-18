# ASL Dataset Reconnaissance and Analysis

## 1. Executive Summary

Building a bidirectional American Sign Language (ASL) and English communication system requires addressing distinct machine learning tasks:
1. **Isolated Sign Recognition**: Mapping short video segments to discrete vocabulary classes.
2. **Continuous Sign Recognition**: Detecting and classifying sign sequences in continuous motion.
3. **Sign-to-Text Translation**: Mapping sign sequences to grammatically complete English sentences.
4. **Text-to-Sign Generation / Glossing**: Reordering English syntax into sign gloss representations.

No single public dataset fulfills all requirements. A multi-stage dataset strategy is necessary, utilizing specific datasets for isolated pre-training, signer-independent generalization, continuous sequence translation, and technical vocabulary coverage.

---

## 2. Dataset In-Depth Profiles

### 2.1 WLASL (Word-Level American Sign Language)

- **Source / Authors**: Dongxu Li, Cristian Rodriguez Opazo, Xin Yu, Hongdong Li (WACV 2020).
- **Scale**: 21,083 video clips covering 2,000 common ASL words/classes, performed by 119 signers.
- **Subsets**:
  - WLASL-100: 100 classes, 2,038 videos.
  - WLASL-300: 300 classes, 5,597 videos.
  - WLASL-1000: 1,000 classes, 14,400 videos.
  - WLASL-2000: Complete set of 2,000 classes.
- **Modalities**: RGB video clips (scraped from YouTube, ASL dictionary websites like ASL-LEX, Handspeak, SigningSavvy), pre-extracted 2D OpenPose keypoints.
- **Task Alignment**: Isolated sign recognition baseline.
- **Strengths**:
  - Standard benchmark with wide adoption in literature.
  - Clean division into tiered subsets for rapid prototyping.
  - Large lexical coverage for everyday words.
- **Weaknesses**:
  - Highly variable video resolutions, lighting, and frame rates.
  - Web scraping link decay: Approximately 15% to 20% of original YouTube source videos are no longer accessible without mirrored archives.
  - Does not model continuous ASL grammar, co-articulation, or sentence-level syntax.
- **License**: Academic Software Licence (ASL). Non-commercial research use only.

### 2.2 MS-ASL (Microsoft American Sign Language)

- **Source / Authors**: Hamid Reza Vaezi Joze, Colin Kohli (CVPR 2019).
- **Scale**: 25,513 annotated video segments across 1,000 sign classes, performed by 222 unique signers.
- **Modalities**: RGB video clips, bounding box annotations for signers, start/end timestamps.
- **Task Alignment**: Isolated sign recognition under unconstrained, real-world conditions.
- **Strengths**:
  - Explicit evaluation protocol for signer-independent generalization ($P(\text{correct} \mid \text{unseen signer})$).
  - Greater signer diversity and variable camera setups compared to laboratory datasets.
  - High degree of visual realism (varying camera angles, distances, and clothing).
- **Weaknesses**:
  - Clips are clipped from longer YouTube videos, resulting in varying boundary precision.
  - Isolated classification task only; lacks continuous discourse annotations.
- **License**: Microsoft Research License. Permitted strictly for research and academic evaluation.

### 2.3 ASL Citizen

- **Source / Authors**: Aashaka Desai et al. (Microsoft, Northeastern University, Boston University, 2023).
- **Scale**: 83,399 video recordings covering 2,731 distinct sign classes, performed by 52 Deaf signers.
- **Modalities**: Front-facing webcam RGB video recordings.
- **Task Alignment**: Real-world isolated sign recognition and dictionary retrieval.
- **Strengths**:
  - Recorded entirely via standard computer webcams in natural home/office environments.
  - Specifically designed for assistive technology and dictionary retrieval tasks.
  - Features signer-independent splits with high ecological validity for desktop and web applications.
  - Permissive licensing structure.
- **Weaknesses**:
  - Primarily isolated signs; does not cover multi-sign sentences or dialogue.
- **License**: Creative Commons Attribution 4.0 International (CC BY 4.0). Can be utilized for open research and commercial product prototyping with attribution.

### 2.4 How2Sign

- **Source / Authors**: Amanda Duarte, Shruti Palaskar, Lucas Ventura, Deepti Ghadiyaram, Kenneth DeHaan, Florian Metze, Jordi Torres, Xavier Giro-i-Nieto (CVPR 2021).
- **Scale**: 80+ hours of continuous ASL video comprising more than 35,000 sentences.
- **Modalities**:
  - Multiview RGB video (front, side, angle).
  - Depth information.
  - 2D and 3D skeletal pose annotations (Panoptic studio subset).
  - Aligned English subtitles and transcriptions.
  - Sign gloss annotations.
  - Audio speech tracks.
- **Task Alignment**: Continuous ASL recognition, sign language translation (ASL -> English text), and sign synthesis (English text -> ASL pose).
- **Strengths**:
  - Gold standard for continuous ASL translation research.
  - Parallel multimodal data enables joint modeling of speech, text, gloss, and 3D pose.
  - Highly accurate sentence-level boundary timestamps.
- **Weaknesses**:
  - Recorded in a controlled studio against green screens with consistent high-end cameras.
  - Significant domain shift between green-screen studio video and consumer webcam calls ($P_{\text{deployment}}(X) \ne P_{\text{dataset}}(X)$).
  - Limited number of unique signers (approximately 10 signers in studio setup).
- **License**: Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0). Restricts commercial deployment.

### 2.5 YouTube-ASL

- **Source / Authors**: Burak Uzkent, Evan Murphy, Cenk Baykal, et al. (Google Research, 2024).
- **Scale**: 984 hours across 11,000+ open-domain videos, containing 610,193 aligned English caption segments from over 2,500 distinct signers.
- **Modalities**: In-the-wild YouTube video clips, aligned English text captions.
- **Task Alignment**: Large-scale open-domain ASL-to-English translation.
- **Strengths**:
  - Unmatched diversity in signers, camera settings, lighting, video compression, and topics.
  - True open-domain vocabulary beyond restricted dictionary sets.
  - Pre-training resource for robust visual feature encoders.
- **Weaknesses**:
  - Automated or loosely aligned captions require aggressive filtering to remove misalignments.
  - Includes videos with cutaways, camera panning, overlays, and non-signing introductory segments.
  - High compute requirements for ingestion, downloading, and landmark extraction.
- **License**: Research dataset released under YouTube Terms of Service and Google Research guidelines. Code and indices provided; video streams fetched from YouTube.

### 2.6 ASL-STEM

- **Source / Authors**: Microsoft Research / University of Washington.
- **Scale**: Targeted lexicon covering science, technology, engineering, and mathematics vocabulary.
- **Modalities**: Clean video recordings of technical signs with associated English terms and definitions.
- **Task Alignment**: Technical domain adaptation and specialized terminology support.
- **Strengths**:
  - Essential for professional, academic, and technical meeting contexts (e.g. software development, engineering, biology).
- **Weaknesses**:
  - Small number of instances per term; intended as a reference lexicon rather than a large-scale training set.
- **License**: Open source research release (MIT / Microsoft Academic).

---

## 3. Dataset Comparative Matrix

| Dataset | Primary Task | Video Samples / Duration | Classes / Vocabulary | Signers | Capture Environment | Modalities | License | Project Utility |
|---|---|---|---|---|---|---|---|---|
| **WLASL** | Isolated Recognition | 21,083 clips | 2,000 signs | 119 | Web-scraped, varied | RGB, 2D Pose | Academic Software License | Milestone 1 baseline |
| **MS-ASL** | Isolated Recognition | 25,513 clips | 1,000 signs | 222 | Web-scraped, varied | RGB, BBoxes | MSR Research | Generalization testing |
| **ASL Citizen** | Isolated Retrieval | 83,399 clips | 2,731 signs | 52 | Consumer webcams | RGB | CC BY 4.0 | Webcam robustness |
| **How2Sign** | Continuous Translation | 80+ hours (~35k sentences) | Continuous | 10 | Studio (Green screen) | RGB, Depth, 3D Pose, Gloss, English | CC BY-NC 4.0 | Milestone 2 translation |
| **YouTube-ASL** | Open-Domain Translation | 984 hours (~610k sentences) | Open-domain | 2,500+ | In-the-wild YouTube | RGB, Captions | YouTube ToS / Research | Large-scale pre-training |
| **ASL-STEM** | Domain Lexicon | ~1,000 terms | Technical STEM | Multiple | Studio / Controlled | RGB, Text definitions | MIT / Open | Technical terminology |

---

## 4. Ingestion, Preprocessing, and Pipeline Strategy

### 4.1 Ingestion and Link Rot Mitigation
1. Public datasets that rely on scraping YouTube links suffer from link rot (videos made private or deleted). Ingestion scripts must parse official index files (`WLASL_v0.3.json`, `MSASL_train.json`), log missing URLs, and download videos in parallel with backoff retries using `yt-dlp`.
2. Raw video files are stored on secure object storage (e.g. AWS S3 or MinIO) and never checked into Git.
3. MD5 checksums are computed for every downloaded video to ensure deterministic splits.

### 4.2 Offline Landmark Feature Extraction
To eliminate repeated video decoding during model training, raw videos are preprocessed through an offline landmark extraction pipeline:
1. Decode video frames at a uniform sample rate (typically 30 FPS, resized to $640 \times 480$).
2. Pass frames through MediaPipe Holistic to extract:
   - Left hand: $21 \times (x, y, z)$ coordinates.
   - Right hand: $21 \times (x, y, z)$ coordinates.
   - Body pose: $33 \times (x, y, z, \text{visibility})$ coordinates.
   - Face: Selected salient facial contour points (eyebrows, eyelids, lips, nose tip).
3. Serialize extracted landmark sequences into compressed format (`.npy` or `.safetensors`) indexed by video ID.
4. This reduces training storage requirements from hundreds of gigabytes of raw MP4 video to a few gigabytes of numerical arrays, accelerating dataloader throughput by more than $10\times$.

---

## 5. Licensing and Deployment Implications

### 5.1 Commercial vs Non-Commercial Restrictions
- **CC BY-NC 4.0 (How2Sign)**: Explicitly prohibits commercial use. Checkpoints trained directly on How2Sign cannot be deployed in commercial paid products without specific licensing agreements with the dataset copyright holders.
- **CC BY 4.0 (ASL Citizen)**: Fully allows commercial use, adaptation, and integration provided proper attribution is given.
- **Academic Licenses (WLASL, MS-ASL)**: Restrict weights and derivative models to research and non-profit educational contexts.

### 5.2 Progressive Research-to-Production Strategy
1. **Milestone 1 and 2 (Proof of Concept & Academic Benchmark)**:
   - Use WLASL and How2Sign to establish state-of-the-art baselines and validate pipeline feasibility.
   - Benchmark model performance against published literature.
2. **Milestone 3 and 4 (Production and Commercial Path)**:
   - Train production-facing recognition models primarily on CC BY 4.0 data (ASL Citizen) combined with proprietary consented signer recordings.
   - Fine-tune translation language models on synthetic ASL gloss-to-English pairs and public domain corpora.
