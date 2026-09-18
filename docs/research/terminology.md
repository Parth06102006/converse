# ASL and Multimodal AI Terminology

This document establishes the standardized linguistic, technical, and architectural terminology used across Converse.

---

## 1. Sign Language Linguistics

### 1.1 Gloss and Glossing
- **Definition**: A written representation of sign language where signs are transcribed using approximate root words in spoken language (typically written in uppercase English lemmas).
- **Example**:
  - ASL Sign Sequence: `TEACHER BOOK GIVE-ME`
  - English Translation: "The teacher gave me the book."
- **Critical Distinction**: Gloss is a transcription convenience used by linguists and computational models. Gloss is not sign language itself, nor does it possess a 1:1 semantic or grammatical equivalence with English words. A sign may convey aspect, intensity, spatial direction, or pluralization that English gloss labels fail to fully capture.

### 1.2 Spatial Grammar and Indexing (`IX`)
- **Definition**: ASL utilizes three-dimensional signing space around the signer (the "signing space") to establish grammatical relationships.
- **Indexing (`IX`)**: The signer points to a specific point in space to establish a referent (a person, place, object, or concept). Subsequent points to that location refer back to the assigned entity, functioning equivalently to pronouns or locatives in spoken language:
  - `IX-1p`: Pointing to oneself ("I", "me", "my").
  - `IX-2p`: Pointing to the addressee ("you").
  - `IX-3p:loc_a`: Pointing to a spatial locus on the left or right to refer to an absent third party ("he", "she", "it", "over there").
- **Directional / Agreement Verbs**: Verbs that alter their motion path and palm orientation between spatial loci to indicate the subject and object (for example, `GIVE` moving from locus A to locus B indicates "A gave to B").

### 1.3 Non-Manual Markers (NMM)
- **Definition**: Grammatical signals produced using facial expressions, head movements, eye gaze, mouth shapes, and torso posture rather than the hands alone.
- **Linguistic Function**:
  - *Yes/No Questions (`q`)*: Eyebrows raised, eyes widened, head tilted forward.
  - *Wh- Questions (`wh-q`)*: Eyebrows furrowed, squinted eyes, head tilted.
  - *Negation (`neg`)*: Head shake, furrowed brow, mouth corner pulled down (can negate a manual sign without producing a separate `NOT` sign).
  - *Adverbial Morphology*: Mouth morphemes conveying size, effort, or texture (e.g. `mm` for regular/pleasant action, `th` for careless action, `cha` for large volume).
- **Engineering Implication**: A vision system that only extracts hand landmarks will fail to distinguish questions from statements or affirmative from negated statements.

### 1.4 Fingerspelling (Manual Alphabet)
- **Definition**: The process of spelling out individual letters of a word using sequential manual hand configurations corresponding to the alphabet.
- **Usage**: Used for proper nouns (names of people, cities, brands), technical acronyms, and words that lack established lexical signs.
- **Engineering Challenge**: Fingerspelling involves rapid transitions (often 4 to 8 letters per second). Capturing and decoding fingerspelling requires high frame rates ($\ge 30\text{ to }60\text{ FPS}$) and fine-grained finger landmark resolution without temporal motion blur.

### 1.5 Co-articulation
- **Definition**: The phonetic and physical modification of a sign's onset handshape, location, or trajectory influenced by the preceding sign, and its offset influenced by the subsequent sign.
- **Impact on Machine Learning**: Isolated sign classifiers trained on dictionary citation forms often fail on continuous signing because the exact starting and ending hand configurations are blurred by transitions into neighboring signs. Continuous sequence models must learn transition dynamics rather than rigid static templates.

---

## 2. Converse Pipeline Architecture Abstractions

To avoid conflating visual perception with linguistic understanding, Converse defines four distinct conceptual levels:

```
Camera Frame ──► SignObservation ──► SignUnderstanding ──► SignRepresentation ──► Translation ──► English Text
```

### 2.1 VideoFrame
- The raw RGB image captured from the optical sensor (e.g. $640 \times 480 \times 3$ uint8 tensor at timestamp $t$).

### 2.2 SignObservation
- **Definition**: Structured physical and kinematic data extracted from one or more video frames.
- **Content**: 3D coordinates of hands, wrists, shoulders, elbows, and facial contour landmarks; bounding boxes; optical flow vectors; tracking confidence scores.
- **Semantics**: An observation answers: *"What physical points and movements are present in the video?"* It contains no semantic or linguistic assertions.

### 2.3 SignUnderstanding
- **Definition**: Intermediate model inference that groups temporal observations into recognized kinetic patterns, candidates, and temporal segments.
- **Content**: Segment boundaries (start timestamp, end timestamp), candidate sign classifications with posterior probability distributions, handshape classifications, and detected movement vectors.
- **Semantics**: Understanding answers: *"What specific signing actions or candidate signs were performed over this time window?"*

### 2.4 SignRepresentation
- **Definition**: Standardized, machine-readable data structure encoding the linguistic content of the recognized signing.
- **Content**: Sign tokens, assigned spatial loci (`IX`), non-manual grammatical flags (e.g. `is_question: true`, `is_negated: true`), confidence metrics, and temporal durations.
- **Semantics**: Representation answers: *"What linguistic message did the signer convey?"* This structure serves as the input to downstream natural language translation.

### 2.5 Translation
- **Definition**: The mapping between `SignRepresentation` and spoken natural language (English text) or vice-versa.
- **Semantics**: Translation answers: *"How is this structured linguistic message expressed fluently in English syntax?"*

---

## 3. Evaluation Metrics

### 3.1 Top-1 / Top-k Accuracy
- **Usage**: Evaluates isolated sign recognition.
- **Definition**: The percentage of test samples where the ground-truth sign class matches the model's highest-probability prediction (Top-1) or is present within the model's top $k$ highest-probability predictions (Top-k).
- **Standard**: Signer-independent evaluation where signers in the test split never appear in the training split.

### 3.2 BLEU (Bilingual Evaluation Understudy)
- **Usage**: Evaluates sign-to-text machine translation quality.
- **Definition**: Computes n-gram precision ($n=1\dots 4$) between model-generated English sentences and human reference translations, adjusted with a brevity penalty for short generations.
- **Target**: How2Sign continuous translation baselines target BLEU-4 scores between $20.0$ and $28.0$.

### 3.3 chrF / chrF++
- **Usage**: Character n-gram F-score used alongside BLEU in sign translation benchmarks to account for morphological variations and minor tokenization discrepancies.

### 3.4 Word Error Rate (WER)
- **Usage**: Evaluates Automatic Speech Recognition (ASR) performance.
- **Definition**: Ratio of substitutions ($S$), deletions ($D$), and insertions ($I$) to the total number of words in the reference transcript ($N$):
  $$\text{WER} = \frac{S + D + I}{N}$$
- **Target**: Streaming Whisper ASR in Converse targets $\text{WER} < 12\%$ on standard speech.

### 3.5 Real-Time Factor (RTF)
- **Usage**: Evaluates streaming latency for ASR and TTS.
- **Definition**: Ratio of processing time to the duration of the audio input/output:
  $$\text{RTF} = \frac{\text{Processing Time (seconds)}}{\text{Audio Duration (seconds)}}$$
- **Requirement**: An $\text{RTF} < 1.0$ is required for real-time processing; Converse targets $\text{RTF} \le 0.15$ for streaming audio components.
