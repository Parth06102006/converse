# Architecture Decision Records

This document records the architectural and design decisions for Converse.

---

## ADR-001: Monorepo Architecture (pnpm + Turborepo + Python/uv)

### Status
Accepted

### Context
Converse combines TypeScript client applications (Next.js web dashboard, Chrome Manifest V3 extension, React Native mobile application), an Express.js API gateway, shared TypeScript libraries, and multiple Python machine learning services (vision perception, translation sequence models, ASR, TTS).

Managing these across disparate repositories introduces synchronization friction, fragmented contract definitions, and complex release cycles. Conversely, forcing Python ML dependencies into Node-centric package management creates brittle wrappers and breaks native Python packaging standards.

### Decision
Adopt a hybrid monorepo architecture:
1. Use `pnpm` workspaces combined with Turborepo for all JavaScript and TypeScript projects (`apps/*`, `services/api`, `packages/*`).
2. Manage Python ML services (`ml/*`) as independent Python projects using `uv` (`uv venv`, `uv pip`, `uv run`) with dedicated `pyproject.toml` configurations.
3. Share data contracts and protocols across TypeScript services via internal workspace packages (`@converse/contracts`, `@converse/protocol`).
4. Keep root task pipelines (`build`, `lint`, `test`, `typecheck`) coordinated via Turborepo, while exposing Python validation via discrete scripts.

### Consequences

#### Positive
- Single source of truth for shared domain interfaces and protocol schemas.
- High-performance incremental build caching and pipeline execution via Turborepo.
- Clean separation between Node.js runtime dependencies and Python machine learning ecosystems.
- Hermetic Python environments without global state pollution.

#### Negative
- CI/CD workflows must configure both Node.js and Python runtime environments.
- Developers working across full-stack boundaries must maintain tooling for both ecosystems.

---

## ADR-002: ASL over ISL (Linguistic Scope and Dataset Availability)

### Status
Accepted

### Context
The initial conceptual inquiry evaluated Indian Sign Language (ISL). However, reconnaissance of available machine learning assets revealed severe structural limitations for ISL:
1. Publicly accessible ISL datasets are predominantly small-scale, vocabulary-limited, and recorded in highly constrained laboratory settings with few signers.
2. High-quality continuous sign-to-text datasets with verified sentence-level alignments are scarce for ISL.
3. Pretrained feature extractors, spatial-temporal graph baselines, and established translation benchmarks are largely unavailable for ISL.

In contrast, American Sign Language (ASL) possesses extensive open research datasets (WLASL, MS-ASL, ASL Citizen, How2Sign, YouTube-ASL), validated baseline architectures, standardized gloss lexicons, and established evaluation benchmarks. Building a production-grade bidirectional communication system within project timelines requires a rich data foundation.

### Decision
Focus the initial product and machine learning scope strictly on American Sign Language (ASL) bidirectional translation with spoken English (ASL <-> English).

### Consequences

#### Positive
- Availability of large-scale isolated datasets (WLASL with 2,000 signs, MS-ASL with 25,000+ clips, ASL Citizen with 83,000+ clips).
- Availability of continuous translation benchmarks (How2Sign with 80+ hours of multimodal data, YouTube-ASL with 980+ hours).
- Access to established baseline models (I3D, ST-GCN, Pose Transformers) for reproducible benchmarking.
- Faster iteration on core algorithmic challenges: signer independence, temporal segmentation, and co-articulation.

#### Negative
- Limits initial regional deployment and addressable user base to ASL signers.
- Expanding to ISL, British Sign Language (BSL), or other sign languages will require dedicated future data acquisition and transfer learning initiatives.

---

## ADR-003: Backend Framework (Express.js over Hono)

### Status
Accepted

### Context
Lightweight edge-first frameworks such as Hono were evaluated for the backend API gateway. While Hono provides small footprint execution on edge runtimes (Cloudflare Workers, Deno), Converse imposes specific runtime requirements:
1. Long-running stateful sessions managing bidirectional WebSocket connections and real-time audio/video streaming.
2. Direct integration with Node.js streaming APIs, native binary buffers, and WebRTC termination tooling (Mediasoup, node-webrtc).
3. Orchestration of asynchronous worker communication with Python ML services.
4. The engineering team possesses deep operational familiarity with Express.js and the standard Node.js ecosystem.

### Decision
Standardize the API backend on Node.js with TypeScript and Express.js, using the `ws` package for WebSocket management.

### Consequences

#### Positive
- Deep ecosystem maturity for WebSockets, WebRTC, and media streaming.
- Predictable execution behavior in containerized environments (Docker, AWS ECS/Fargate).
- Rapid team velocity and minimal ramp-up overhead.
- Broad compatibility with observability, telemetry, and debugging libraries.

#### Negative
- Higher memory footprint compared to minimal edge runtimes.
- Does not deploy natively to serverless edge platforms without compatibility shims.

---

## ADR-004: Rust-Inspired Result and Option Semantics for Cross-Boundary Errors

### Status
Accepted

### Context
Real-time media pipelines involve continuous streaming of landmark frames, audio chunks, classification probabilities, and translation tokens. In streaming loops, throwing uncaught exceptions causes unhandled promise rejections, breaks WebSocket socket pipelines, and terminates client sessions.

Furthermore, many failures in perception and translation are recoverable or degraded states rather than fatal crashes (for example, temporary loss of hand tracking, low-confidence sign classification, or silence during voice activity detection).

### Decision
Define lightweight, zero-dependency `Result<T, E>` and `Option<T>` types in `@converse/contracts` and enforce their use across service, contract, and adapter boundaries.

```ts
export type Result<T, E> =
  | { readonly ok: true; readonly value: T }
  | { readonly ok: false; readonly error: E };

export type Option<T> =
  | { readonly some: true; readonly value: T }
  | { readonly some: false };
```

Exceptions (`throw`) are restricted strictly to unrecoverable system invariant violations. All expected domain errors (validation failures, classification under threshold, connection timeouts) must return `Result.err`.

### Consequences

#### Positive
- Forces compile-time handling of missing data and error conditions via TypeScript disciminated unions.
- Prevents unhandled exceptions from terminating long-running streaming connections.
- Cleanly separates partial degradation (e.g. low tracking confidence) from fatal system failure.
- Predictable serialization format for error messages over WebSocket channels.

#### Negative
- Requires explicit wrapping and unwrapping of values across codebase layers.
- Developers must avoid throwing exceptions in asynchronous utility functions.

---

## ADR-005: Model Pipeline Boundary (Pose Landmarking vs Direct Video)

### Status
Accepted (Landmark-First with Hybrid Fallback)

### Context
Sign language vision can follow two primary architectures:
1. **Direct Video (Pixel-Based)**: Feeding raw RGB video frames into 3D Convolutional Neural Networks (e.g. I3D) or Video Transformers.
2. **Pose Landmarking (Geometry-Based)**: Extracting 2D/3D coordinate keypoints for hands, body pose, and facial expressions, then feeding coordinate sequences into temporal models (e.g. ST-GCN, TCN, Transformers).

Ingesting uncompressed 640x480 RGB video at 30 FPS consumes ~27.6 MB/s per client stream. Streaming raw video to cloud GPUs creates prohibitive network ingress costs, server GPU bottlenecks, and severe privacy concerns for deaf signers.

Extracting landmarks compresses the input from millions of bytes to hundreds of coordinate values per frame, strips private visual features (background, skin tone, room environment), and allows perception to run near the edge (client browser or local device).

However, landmarks discard subtle visual signals such as fine finger contacts, skin deformation, and nuanced mouthing patterns.

### Decision
Adopt a **Landmark-First perception architecture** as the foundational baseline:
1. Extract 3D skeletal landmarks (21 per hand, 33 body pose, key facial points) using MediaPipe Holistic at or near the client capture edge.
2. Normalize coordinates relative to skeletal anchor points (wrist position and shoulder width) to achieve scale and position invariance.
3. Feed normalized coordinate sequences into downstream temporal recognition models.
4. Architect model contracts with an optional visual crop fallback payload, allowing future hybrid processing for ambiguous hand configurations without changing pipeline boundaries.

### Consequences

#### Positive
- Dramatic reduction in network bandwidth (kilobytes per second instead of megabytes per second).
- Significantly reduced cloud compute costs; temporal models run efficiently on CPU or modest GPUs.
- Enhanced user privacy; raw camera video frames never need to leave the client device.
- Stronger inductive bias; the model focuses on motion and geometry rather than background lighting and clothing.

#### Negative
- Model accuracy is bounded by upstream landmark extractor fidelity; tracking failures propagate downstream.
- Subtle non-manual markers and mouth shapes may suffer from coordinate coarseness.

---

## ADR-006: Independent ML Environments via `uv`

### Status
Accepted

### Context
Converse incorporates multiple Python machine learning components:
- `ml/asl-vision`: Computer vision, pose estimation, and temporal landmark modeling (PyTorch, MediaPipe, OpenCV).
- `ml/translation`: Sequence-to-sequence language modeling (Hugging Face Transformers, PyTorch).
- `ml/asr`: Automatic speech recognition (Whisper, ONNX Runtime, Silero VAD).
- `ml/tts`: Text-to-speech synthesis (FastSpeech2, Coqui, ONNX).

Combining all ML components into a single Python virtual environment leads to severe dependency pinning conflicts (conflicting CUDA wheels, divergent PyTorch versions, incompatible tokenizers, or platform-specific binaries).

### Decision
Isolate each ML service in `ml/*` with its own independent `pyproject.toml` and manage virtual environments exclusively using `uv`. Global Python packages and standard `pip` invocations are strictly disallowed.

### Consequences

#### Positive
- Hermetic, conflict-free dependency graphs for each ML component.
- Sub-second environment resolution and installation via `uv` cache.
- Independent containerization for deployment (e.g. lightweight CPU containers for ASR/TTS and GPU-accelerated containers for vision).
- Clear ownership boundaries across team members.

#### Negative
- Multiple virtual environments consume additional local disk space.
- Requires independent test and lint commands per ML service directory.
