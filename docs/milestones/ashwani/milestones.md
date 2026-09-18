# Engineering Milestone Plan: Speech-to-Sign Model Pipeline

- **Owner**: Ashwani
- **Domain**: Machine Learning, Audio Processing & Natural Language Translation (Model Layer)
- **Primary Workspace Paths**: `ml/asr/`, `ml/translation/`, `packages/contracts/src/speech.ts`, `packages/contracts/src/translation.ts`
- **Scope Boundary**: Strictly backend and ML model execution. Avatar rendering, UI presentation, and client-side audio capture are owned by downstream tracks.

---

## Executive Summary

The Speech-to-Sign model pipeline transforms continuous spoken English audio into structured American Sign Language (ASL) animation representations. This pipeline bridges acoustic human speech and visual spatial grammar through three model-driven phases:
1. Low-latency Voice Activity Detection (VAD) and Streaming Automatic Speech Recognition (ASR).
2. Grammatical transformation from English syntax (Subject-Verb-Object) to ASL linguistic structure (Topic-Comment / Time-Subject-Object-Verb).
3. Emission of structured sign representation tokens with spatial loci, non-manual marker (NMM) flags, and keyframe hold/transition durations.

```mermaid
flowchart LR
    Audio["Streaming Audio (16kHz PCM)"] --> VAD["Silero VAD (Speech Detection)"]
    VAD --> ASR["Streaming ASR (Faster-Whisper / Zipformer)"]
    ASR --> Transcripts["Partial / Final English Transcripts"]
    Transcripts --> Grammar["English-to-ASL Grammar Compiler"]
    Grammar --> Tokens["Normalized Gloss Tokens & NMM Cues"]
    Tokens --> Timing["Sign Representation & Timing Generator"]
    Timing --> Output["SignRepresentation (Loci, Durations, NMMs)"]
```

---

## Phase 1: Audio Ingestion, VAD, and Streaming ASR Engine

### 1.1 Objective
Establish an ultra-low-latency streaming transcription engine capable of ingesting raw 16kHz audio chunks, segmenting conversational utterances, and producing accurate partial and final English transcripts.

### 1.2 Input and Output Contracts
- **Input Contract**: Streaming linear 16-bit PCM or WebM Opus audio chunks at 16kHz sample rate (chunk durations between 20ms and 100ms) conforming to `AudioChunkPayload` in `@converse/contracts`.
- **Output Contract**: Emitted transcription events conforming to `AsrTranscriptEvent`:
  ```typescript
  interface AsrTranscriptEvent {
    sessionId: string;
    sequenceId: number;
    text: string;
    isFinal: boolean;
    confidence: number;
    wordTimestamps?: Array<{ word: string; startMs: number; endMs: number }>;
    latencyMetrics: {
      audioDurationMs: number;
      processingTimeMs: number;
    };
  }
  ```

```mermaid
sequenceDiagram
    autonumber
    participant Client as Client Audio Stream
    participant VAD as Silero VAD (ONNX)
    participant Buffer as Sliding Audio Buffer
    participant ASR as Streaming ASR Engine
    participant Gateway as Real-time Gateway

    Client->>VAD: Stream 16kHz PCM chunk (30ms)
    VAD->>Buffer: Speech probability > 0.5 (Active Speech)
    Buffer->>ASR: Accumulated audio window (500ms-1500ms)
    ASR->>Gateway: Emit partial transcript (isFinal=false)
    Note over Client,Gateway: Loop continues every 150ms-250ms

    Client->>VAD: Silence chunk (Speech prob < 0.2)
    Note over VAD: Silence counter reaches 300ms threshold
    VAD->>Buffer: Trigger utterance boundary flush
    Buffer->>ASR: Final utterance segment
    ASR->>Gateway: Emit final transcript (isFinal=true)
```

### 1.3 Architectural Seams and Technical Decisions

#### Decision Seam A: Voice Activity Detection (VAD) Strategy
Conversational speech contains hesitations, breathing pauses, and background noise. ASR models degrade or hallucinate when processing long silent segments.
- **Option 1: Silero VAD (PyTorch / ONNX)**
  - *Pros*: State-of-the-art accuracy, resilient against ambient room noise, sub-1ms inference time per 30ms chunk, pre-packaged ONNX runtime.
  - *Cons*: Slightly higher memory footprint than basic energy thresholding (~5MB memory).
- **Option 2: WebRTC VAD (GMM-based)**
  - *Pros*: Extremely lightweight, zero deep-learning overhead, near-zero CPU consumption.
  - *Cons*: Highly sensitive to microphone gain and background noise; produces false negatives on soft or whispery speech.
- **Option 3: Adaptive Energy-Based Thresholding**
  - *Pros*: Zero external dependencies, pure NumPy implementation.
  - *Cons*: Fails in dynamic acoustic environments with ambient background speech.
- *Recommendation for Evaluation*: Benchmark Silero VAD ONNX against WebRTC VAD on noisy conversational clips. Test trigger threshold tuning (`threshold=0.5`, `min_speech_duration_ms=250`, `min_silence_duration_ms=300`).

#### Decision Seam B: Streaming ASR Architecture
The engine must achieve low latency without catastrophic degradation of Word Error Rate (WER).
- **Option 1: Faster-Whisper (CTranslate2-backed Whisper)**
  - *Pros*: 4x faster than standard PyTorch Whisper, native INT8 / FP16 quantization, high transcription accuracy, word-level timestamps.
  - *Cons*: Whisper was architected as an offline sequence-to-sequence model; streaming requires sliding buffer heuristics and hallucination suppression.
- **Option 2: Sherpa-ONNX (Next-Gen Kaldi / Streaming Zipformer)**
  - *Pros*: Designed ground-up for causal streaming inference; zero sliding buffer hacks; true chunk-by-chunk streaming with deterministic 160ms latency; native ONNX deployment without GPU requirement.
  - *Cons*: Slightly lower vocabulary breadth than OpenAI Whisper on niche domain technical terms.
- **Option 3: Moonshine ASR (Useful Sensors)**
  - *Pros*: Specifically optimized for edge inference and streaming; dynamic compute based on input audio length; runs sub-realtime on resource-constrained CPUs.
  - *Cons*: Relatively novel architecture with smaller community fine-tuning ecosystem.
- *Recommendation for Evaluation*: Conduct an empirical shootout between Faster-Whisper `base.en`/`small.en` (quantized INT8) and Sherpa-ONNX streaming Zipformer on the LibriSpeech test-clean and conversational datasets.

### 1.4 Technical Invariants and Edge Cases
- **Acoustic Invariant**: Audio input must be strictly mono, 16,000 Hz, 16-bit little-endian PCM. If client sends non-conforming rates (e.g., 44.1kHz or 48kHz), implement resampler with high-pass filtering.
- **Hallucination Suppression**: Whisper models running on silence loops tend to emit repetitive strings (e.g., "You", "Thank you for watching"). Ashwani must implement temperature fallback and repetition penalties.
- **Partial vs. Final Hypothesis Boundary**: Emit `isFinal=false` partial transcripts every 150ms-250ms for live visual feedback, but lock `isFinal=true` only upon VAD silence boundary detection (>300ms silence) to prevent premature grammar compilation.

### 1.5 Deliverables and Milestones
1. `ml/asr/src/vad.py`: Streaming VAD module wrapping Silero VAD ONNX with configurable attack/decay thresholds.
2. `ml/asr/src/engine.py`: ASR inference engine supporting both streaming chunk ingestion and buffered evaluation.
3. `ml/asr/scripts/benchmark_asr.py`: Benchmark suite measuring Real-Time Factor (RTF), Word Error Rate (WER), and Time-to-First-Token (TTFT) across LibriSpeech and Common Voice.
4. `ml/asr/tests/test_streaming_asr.py`: Unit tests validating audio buffer rollbacks, silence handling, and schema compliance.

### 1.6 Verification Criteria
- Time-to-partial-transcript $\le 250\text{ms}$ from chunk arrival.
- Word Error Rate (WER) $\le 12.0\%$ on clean English speech samples.
- Peak CPU utilization $\le 40\%$ of a single modern CPU core for continuous streaming.

---

## Phase 2: English-to-ASL Grammatical Transformation and Gloss Compiler

### 2.1 Objective
Translate grammatically standard English text into syntactically valid ASL gloss sequences, extracting non-manual markers (facial cues) and spatial indexing references.

### 2.2 Linguistic Context and Core Challenges
ASL is not signed English. It is an independent natural language with distinct syntax and morphology:
- **Word Order**: English uses Subject-Verb-Object (SVO). ASL predominantly uses Topic-Comment structure and Time-Subject-Object-Verb (TSOV).
  - *Example*: "I will go to the market tomorrow" -> `TOMORROW MARKET I GO`.
- **Copula Deletion**: Forms of the verb "to be" (is, are, was, were) do not exist in ASL.
  - *Example*: "The car is blue" -> `CAR BLUE`.
- **Question Morphology and Non-Manual Markers (NMM)**:
  - Wh-questions (Who, What, Where, When, Why, How): Wh-word moves to the end of the sentence accompanied by furrowed brows (`wh_question` flag).
    - *Example*: "Where do you live?" -> `YOU LIVE WHERE [NMM: EYEBROWS_FURROWED]`.
  - Yes/No questions: Eyebrows raised, head tilted forward (`yn_question` flag).
    - *Example*: "Are you tired?" -> `YOU TIRED [NMM: EYEBROWS_RAISED]`.
- **Negation**: Negative particle placed after verb or topic, accompanied by head shake (`negation` flag).
  - *Example*: "I do not want tea" -> `TEA I WANT NOT [NMM: HEAD_SHAKE]`.
- **Spatial Indexing and Pronominalization**: Referents are established at physical spatial coordinates (e.g., Locus A on left, Locus B on right). Pronouns point to these loci.

### 2.3 Input and Output Contracts
- **Input Contract**: Finalized English string with sentence punctuation (from Phase 1).
- **Output Contract**: Structured ASL translation payload conforming to `TextToSignResponse`:
  ```typescript
  interface AslGlossToken {
    gloss: string;                    // Normalized uppercase ASL gloss identifier (e.g., "TOMORROW")
    lemma: string;                    // Base English lemma
    partOfSpeech: string;             // Universal POS tag
    nonManualMarkers: {
      eyebrows: 'neutral' | 'raised' | 'furrowed';
      headMotion: 'neutral' | 'nod' | 'shake' | 'tilt_forward';
      mouthMorpheme?: string;         // e.g., "cha" (large), "mm" (normal/relaxed), "oo" (tiny)
    };
    spatialLocus?: 'center' | 'left' | 'right' | 'contralateral';
    isFingerspelled: boolean;         // True if Out-of-Vocabulary (OOV) named entity
    fingerspellSequence?: string[];   // Individual character tokens if fingerspelled
  }
  ```

### 2.4 Architectural Seams and Technical Decisions

#### Decision Seam: Transformation Architecture
- **Option 1: Deterministic Dependency Parsing and Rule-Based Reordering (spaCy / Stanza)**
  - *Mechanism*: Extract dependency parse trees, identify core nominal subjects, direct objects, temporal adverbials, and wh-determiners. Apply deterministic tree transformation rules to convert SVO to Topic-Comment.
  - *Pros*: Zero hallucination, deterministic output, sub-5ms execution time, zero GPU requirement, 100% auditable rules.
  - *Cons*: Difficult to scale to every English idiom and complex compound sentences; rules can become brittle.
- **Option 2: Compact Fine-Tuned Sequence-to-Sequence Model (MarianMT / T5-small / ByT5)**
  - *Mechanism*: Fine-tune Helsinki-NLP MarianMT or T5-small on parallel English-ASL datasets (ASLG-PC12, How2Sign gloss alignments).
  - *Pros*: Generalizes to colloquial phrases and varied sentence structures; handles lexical substitution naturally.
  - *Cons*: Susceptible to hallucinations or dropping key negation words; higher inference latency (50ms-120ms); requires curated training data.
- **Option 3: Quantized Edge LLM with Structured JSON Decoding (Qwen2.5-0.5B / Llama-3.2-1B)**
  - *Mechanism*: Prompt a small instruction-tuned model with in-context few-shot linguistic rules, enforcing JSON schema outputs via llama.cpp or Outlines.
  - *Pros*: Exceptional linguistic nuance, robust handling of idioms and complex syntax, native extraction of non-manual markers.
  - *Cons*: Highest resource footprint (~1GB-2GB RAM), inference latency between 100ms and 300ms on CPU.
- **Option 4: Hybrid Pipeline (Recommended Architecture)**
```mermaid
flowchart TD
    EnglishIn["Input English Sentence (Finalized ASR)"] --> Matcher{"Fast-Path Rule Matcher (spaCy Tree)"}
    
    Matcher -->|"High Confidence Match"| RuleCompiler["Rule-Based Reordering: SVO to Topic-Comment / TSOV"]
    Matcher -->|"Low Confidence / Complex Idiom"| NeuralFallback["Neural Fallback (MarianMT / Fine-tuned T5 / Qwen2.5)"]
    
    RuleCompiler --> Copula["Copula & Article Deletion ('is', 'the')"]
    NeuralFallback --> Copula
    
    Copula --> NMMClassifier["Non-Manual Marker (NMM) Classifier"]
    NMMClassifier --> WhCheck{"Is Wh-Question?"}
    WhCheck -->|"Yes"| Furrow["Flag Eyebrows Furrowed & Move Wh-Word to End"]
    WhCheck -->|"No"| YnCheck{"Is Yes/No Question?"}
    YnCheck -->|"Yes"| Raise["Flag Eyebrows Raised & Head Tilt"]
    YnCheck -->|"No"| Neutral["Neutral / Sentiment Head Motion"]

    Furrow --> LexiconCheck{"Check Sign Lexicon"}
    Raise --> LexiconCheck
    Neutral --> LexiconCheck

    LexiconCheck -->|"In Vocabulary"| GlossToken["Emit AslGlossToken"]
    LexiconCheck -->|"Out of Vocabulary (OOV)"| Fingerspell["Decompose to Fingerspelled Characters"]
```

### 2.5 Out-of-Vocabulary (OOV) and Fingerspelling Strategy
- When encountering named entities (person names, city names, acronyms, technical terms) not present in the ASL avatar animation lexicon:
  - Set `isFingerspelled: true`.
  - Decompose string into character tokens: `"ALICE"` -> `["A", "L", "I", "C", "E"]`.
  - Assign standardized transition intervals for fingerspelling rhythm (typically 100ms-150ms per letter).

### 2.6 Deliverables and Milestones
1. `ml/translation/src/grammar_rules.py`: Rule-based dependency tree transformer using spaCy for SVO-to-TSOV and question topicalization.
2. `ml/translation/src/nmm_detector.py`: Non-manual marker classifier predicting eyebrow, head, and mouth morphemes based on sentence mood and sentiment.
3. `ml/translation/src/gloss_compiler.py`: Main translation orchestrator implementing hybrid rule/model routing.
4. `ml/translation/scripts/evaluate_grammar.py`: Test harness evaluating BLEU, ROUGE, and syntax alignment on benchmark sentences.

### 2.7 Verification Criteria
- Accuracy $\ge 90\%$ on canonical question conversions (Wh-questions and Yes/No questions correctly flagged with respective NMMs).
- Execution latency $\le 15\text{ms}$ for rule-based compilation; $\le 80\text{ms}$ if utilizing neural fallback.
- Correct detection and decomposition of OOV words into fingerspelled sequences.

---

## Phase 3: Sign Representation, Animation Timing Tokens, and Interface Synchronization

### 3.1 Objective
Translate the high-level ASL gloss sequence into precise, time-synchronized `SignRepresentation` tokens containing animation clip identifiers, spatial coordinates, interpolation curves, and transition hold times for the downstream 3D avatar engine.

### 3.2 Input and Output Contracts
- **Input Contract**: Sequence of `AslGlossToken` objects from Phase 2.
- **Output Contract**: Fully resolved `SignRepresentation` emitted to the real-time gateway:
  ```typescript
  interface SignRepresentation {
    version: '1.0.0';
    sessionId: string;
    utteranceId: string;
    totalDurationMs: number;
    tokens: Array<{
      tokenId: string;
      clipId: string;                 // Matches avatar animation asset ID (e.g., "asl_car_01")
      gloss: string;
      timing: {
        startTimeMs: number;
        leadInDurationMs: number;     // Transition from rest / previous sign
        holdDurationMs: number;       // Peak hold of sign stroke
        leadOutDurationMs: number;    // Blend into next sign
      };
      spatialLoci: {
        anchor: 'neutral_space' | 'chest' | 'forehead' | 'left_shoulder' | 'right_shoulder';
        targetOffset: { x: number; y: number; z: number };
      };
      nonManualMarkers: {
        eyebrowIntensity: number;     // 0.0 (neutral) to 1.0 (max furrow/raise)
        eyebrowShape: 'furrow' | 'raise' | 'neutral';
        headRotation: { pitch: number; yaw: number; roll: number };
        mouthShape: string;
      };
      interpolationCurve: 'linear' | 'ease_in_out' | 'bezier_slerp';
    }>;
  }
  ```

### 3.3 Architectural Seams and Technical Decisions

#### Decision Seam A: Timing and Rhythm Heuristics
Natural signing exhibits fluid transitions (co-articulation) where the hands do not return to a resting pose between signs:
- **Stroke Phase**: The primary semantic gesture. Typical hold duration: 200ms - 400ms.
- **Transition Phase (Co-articulation)**: Moving from preceding sign stroke to next sign stroke. Typical transition duration: 100ms - 200ms.
- **Syntactic Pause**: Boundary between phrases or sentences. Pause duration: 250ms - 500ms with resting hands or slight chest drop.
- *Task for Ashwani*: Formulate an adaptive timing model that adjusts sign duration based on speaker cadence (words-per-minute of incoming speech).

#### Decision Seam B: Spatial Loci Assignment
Pronouns in ASL (HE, SHE, THEY, THAT) require consistent spatial indexing:
- Maintain an active referent cache for each conversation session:
  - Referent 1 established at `left` locus ($(-0.3, 0.0, 0.4)$).
  - Referent 2 established at `right` locus ($(0.3, 0.0, 0.4)$).
  - Self references index toward `chest` ($0.0, 0.0, 0.1$).
- Clear referents on discourse topic change.

### 3.4 Integration and Service Exposure
- Expose a high-performance Python service inside `ml/asr` and `ml/translation`.
- Provide both:
  1. Synchronous REST endpoint (`POST /internal/speech-to-sign/compile`).
  2. Persistent bi-directional WebSocket handler for streaming chunk-in / token-out integration with `services/api`.

### 3.5 Deliverables and Milestones
1. `ml/translation/src/timing_model.py`: Dynamic sign duration and co-articulation transition calculator.
2. `ml/translation/src/spatial_loci.py`: Session-aware spatial referent tracker.
3. `ml/translation/src/representation_emitter.py`: Final packaging module producing valid `SignRepresentation` JSON schemas.
4. `ml/translation/tests/test_representation.py`: Schema validation tests asserting non-overlapping timestamps and valid coordinate bounds.

### 3.6 Verification Criteria
- Emitted `SignRepresentation` passes 100% of JSON schema validations against `@converse/contracts`.
- Monotonically increasing timestamps ($T_{start, i+1} \ge T_{start, i} + \text{leadIn}_i$).
- End-to-end model pipeline latency (Audio Chunk In -> `SignRepresentation` Out) $\le 350\text{ms}$.

---

## Research References and Benchmark Datasets

1. **Sherpa-ONNX & Zipformer**:
   - Paper: *Recent Advances in Kaldi Using Emformer and Zipformer for Streaming Speech Recognition* (IEEE TASLP 2023).
   - Code: https://github.com/k2-fsa/sherpa-onnx
2. **ASL Linguistic Structure and Grammar**:
   - Paper: *Linguistics of American Sign Language: An Introduction* (Valli, Lucas, et al.).
   - Dataset: ASLG-PC12 (English to ASL Gloss Parallel Corpus, 100K sentence pairs).
   - Dataset: How2Sign (Multimodal ASL dataset with aligned gloss annotations).
3. **Silero VAD**:
   - Repository: https://github.com/snakers4/silero-vad
