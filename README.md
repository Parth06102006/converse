# Converse

Real-time bidirectional American Sign Language (ASL) <-> English communication engine.

Converse bridges the communication gap between Deaf/Hard-of-Hearing signers and spoken English communicators across in-person interactions, web browsers, mobile devices, and live video/VoIP calls.

---

## Architecture Overview

Converse operates two concurrent, low-latency streaming pipelines:

### 1. Sign -> Speech Pipeline

```text
Camera Video
    │
    ▼
ASL Vision (MediaPipe 3D Landmark Tracking)
    │
    ▼
Sign Representation (Linguistic Tokens & Non-Manual Markers)
    │
    ▼
Translation Engine (ASL Grammar -> English Syntax)
    │
    ▼
English Text
    │
    ▼
Neural Text-to-Speech (Streaming Audio)
    │
    ▼
Device Speaker / Virtual Audio Output
```

### 2. Speech -> Sign Pipeline

```text
Microphone Audio
    │
    ▼
Streaming ASR (Speech -> English Text)
    │
    ▼
Translation Engine (English Syntax -> ASL Gloss Sequence)
    │
    ▼
ASL Representation (Spatial Loci & Timing Tokens)
    │
    ▼
3D Avatar / Sign Renderer (Visual Motion Display)
```

---

## Repository Structure

The repository is organized as a hybrid monorepo separating TypeScript applications and services from Python machine learning components:

```text
converse/
├── apps/
│   ├── web/                    # Next.js 16 web application and developer console
│   ├── extension/              # Chrome Manifest V3 extension for VoIP / Google Meet / Zoom
│   └── mobile/                 # Mobile client application
├── services/
│   └── api/                    # Express.js session coordinator and streaming gateway (Port 4000)
├── packages/
│   ├── contracts/              # Single source of truth for types, API schemas, and Result/Option failure contracts
│   ├── ui/                     # Shared React component library and visual design system
│   ├── typescript-config/      # Shared tsconfig configurations
│   ├── eslint-config/          # Shared ESLint flat configurations
│   ├── protocol/               # Wire-level streaming protocol definitions
│   ├── config/                 # Shared environment and network configurations
│   └── client/                 # Reusable client SDK
├── ml/
│   ├── asl-vision/             # 3D landmark extraction and sign detection (Python / uv)
│   ├── translation/            # Sequence-to-sequence ASL <-> English translation models
│   ├── asr/                    # Speech recognition models and audio preprocessing
│   └── tts/                    # Neural speech synthesis engines
├── models/
│   ├── checkpoints/            # Model weights and serialized checkpoints (gitignored)
│   ├── configs/                # Model hyperparameter and architecture configurations
│   └── evaluation/             # Evaluation metrics and benchmark test suites
└── docs/
    ├── prd.md                  # Product requirements, user personas, latency budgets
    ├── architecture.md         # Master architecture specification
    ├── contracts.md            # REST API signatures, schemas, and Result/Option semantics
    ├── decisions.md            # Architecture Decision Records (ADRs 001 to 006)
    ├── milestones.md           # Phased roadmap and implementation checklists
    ├── architecture/           # Subsystem architecture dossiers (data flow, realtime, runtime)
    ├── api/                    # Streaming protocol and wire-level framing
    └── research/               # ML research dossiers (datasets, models, linguistics)
```

---

## Getting Started

### Prerequisites

- **Node.js**: `>= 24.0.0`
- **pnpm**: `>= 11.0.0`
- **Python**: `>= 3.12`
- **uv**: `>= 0.12.0` (for Python ML package isolation)

### Installation

Install all JavaScript/TypeScript monorepo dependencies:

```bash
pnpm install
```

Set up Python ML environments (managed independently via `uv`):

```bash
cd ml/asl-vision
uv venv
uv sync
cd ../..
```

---

## Development

### Run Development Services

Start all workspace applications and services concurrently:

```bash
pnpm dev
```

Target specific services:

```bash
# Start the web frontend (http://localhost:3000)
pnpm --filter web dev

# Start the API gateway (http://localhost:4000)
pnpm --filter @converse/api dev
```

---

## Verification and Quality Routine

Before committing changes, ensure that all static checks pass:

```bash
# Typecheck all packages
pnpm check-types

# Lint all packages
pnpm lint

# Build all packages
pnpm build

# Run Python ML tests
cd ml/asl-vision && uv run pytest && cd ../..
```

---

## Technical Documentation

Detailed specifications and architecture records are maintained in `docs/`:

- [Product Requirements Document (`docs/prd.md`)](docs/prd.md): Vision, user personas, latency budgets, and functional requirements.
- [System Architecture (`docs/architecture.md`)](docs/architecture.md): Component topology, data flow, and subsystem boundaries.
- [Data Contracts (`docs/contracts.md`)](docs/contracts.md): REST signatures, schemas, and `Result<T, E>` / `Option<T>` error patterns.
- [Streaming Protocol (`docs/api/protocol.md`)](docs/api/protocol.md): Real-time WebRTC media channels and WebSocket message framing.
- [Architecture Decisions (`docs/decisions.md`)](docs/decisions.md): Formally recorded ADRs (ADR-001 through ADR-006).
- [Roadmap & Milestones (`docs/milestones.md`)](docs/milestones.md): Detailed phase completion criteria from Milestone 0 to 4.
- [Research Dossiers (`docs/research/`)](docs/research/): Analysis on datasets (WLASL, YouTube-ASL, How2Sign), ML models, and sign linguistics.

---

## Engineering Standards

- **Strict Typing**: Zero unvetted `any` types; all interfaces must have explicit boundaries with strict null and undefined safety.
- **Contract Centric**: Shared interfaces live exclusively in `@converse/contracts`.
- **Zero Emoji Standard**: Strictly zero emojis across all code, comments, documentation, UI strings, and commit messages.
- **Tooling Isolation**: Use `pnpm` exclusively for Node workspaces and `uv` exclusively for Python ML environments. Never invoke global `pip` or alternative Node package managers.
