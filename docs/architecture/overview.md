# System Architecture

## Scope

ASL <-> English communication system.

## Directions

### Sign -> Speech

Camera -> ASL Vision -> ASL Representation -> Translation -> English Text -> TTS -> Audio

### Speech -> Sign

Audio -> ASR -> English Text -> Translation -> ASL Representation -> Sign Renderer

## Applications

- Web (`apps/web`)
- Chrome Extension (`apps/extension`)
- Mobile (`apps/mobile`)

## Core Principle

Logical component boundaries do not necessarily imply deployment boundaries.

## Architecture Documentation Suite

- Master Architecture Specification: [docs/architecture.md](../architecture.md)
- Data Flow Architecture: [docs/architecture/data-flow.md](data-flow.md)
- Real-Time Communication & Streaming Architecture: [docs/architecture/realtime.md](realtime.md)
- Runtime Topology & State Management: [docs/architecture/runtime.md](runtime.md)
