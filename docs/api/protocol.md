# Converse Real-time Streaming Protocol Specification

## 1. Protocol Architecture and Transport Overview

The Converse real-time streaming protocol coordinates bidirectional, sub-second communication between client applications (Web, Chrome Extension, Mobile) and backend processing pipelines.

### 1.1 Dual-Track Transport Architecture

Converse supports a dual-track transport model combining WebSockets and WebRTC:

```text
┌───────────────────────────────────────────────────────────┐
│                      Client Application                   │
└─────────────┬───────────────────────────────┬─────────────┘
              │                               │
       WebSocket Track                 WebRTC Track
   (Control, Events, Meta)         (Media Tracks & Data)
              │                               │
              ▼                               ▼
┌──────────────────────────┐    ┌───────────────────────────┐
│     Node.js Gateway      │    │     Media SFU / Worker    │
│  (Session Orchestration) │    │  (Direct Media Ingestion) │
└─────────────┬────────────┘    └─────────────┬─────────────┘
              │                               │
              └───────────────┬───────────────┘
                              ▼
               Internal Redis / IPC Streams
                              ▼
                 Python ML Inference Workers
                 (Vision, ASR, Translation, TTS)
```

1. **WebSocket Track (`/ws/realtime`)**:
   - Primary control plane and event stream.
   - Handles connection handshake, session initialization, capability negotiation, ping/pong keepalives, and session termination.
   - Transmits structured JSON events: sign detection tokens, intermediate transcript updates, translation results, and synthesized audio chunks.
   - Serves as the universal baseline transport when WebRTC peer connections cannot be established due to restrictive enterprise firewalls or NAT configurations.

2. **WebRTC Track (`/webrtc/offer`, `/webrtc/answer`, `/webrtc/ice`)**:
   - Low-latency media transport track.
   - Transports camera video tracks directly to GPU inference workers for real-time landmark extraction.
   - Transports microphone Opus audio tracks directly to streaming ASR engines.
   - Provides `RTCDataChannel` for ultra-low-overhead binary transport of 3D landmark arrays and timestamp sync messages.

---

## 2. Session Lifecycle and State Machine

A Converse real-time session transitions through four distinct phases:

```text
[DISCONNECTED]
      │
      ▼  (WebSocket Connect)
[CONNECTING]
      │
      ▼  (Client sends 'session_init')
[INITIALIZING]
      │
      ▼  (Server sends 'session_ready')
[ACTIVE] ◄────────────────────────┐
      │                           │  (ping/pong keepalive)
      ├──► [STREAMING] ───────────┘
      │
      ▼  (Client/Server 'session_close' or error)
[TERMINATED]
```

### 2.1 Phase 1: Connection and Session Initialization

1. Client opens a WebSocket connection to `wss://<host>:<port>/ws/realtime`.
2. Within 3000 milliseconds of socket connection, the client must transmit a `session_init` message defining the communication direction, client identifier, and media configuration.
3. The server allocates a session state record, reserves downstream ML worker pipelines, and replies with `session_ready`.
4. If `session_init` is not received within the 3000 millisecond timeout window, the server closes the connection with WebSocket close code `4408` (Initialization Timeout).

### 2.2 Phase 2: Active Bidirectional Streaming

Once in the `ACTIVE` state:
- **Sign-to-Speech**:
  - Client streams `frame_landmarks` at 20 to 30 Hz.
  - Server aggregates landmark sequences, evaluates temporal gesture boundaries, and emits `sign_detected` when a sign gloss is recognized.
  - Server Translation Engine translates accumulated sign glosses into English text, emitting `translation_result`.
  - Server TTS synthesizes speech and streams `tts_audio` chunks to the client.
- **Speech-to-Sign**:
  - Client streams `audio_chunk` packets at 50 to 100 millisecond intervals.
  - Server ASR emits partial and final `transcript_update` messages.
  - Server Translation Engine translates English sentences into ASL tokens, emitting `translation_result`.
  - Client Sign Renderer animates the avatar using the received tokens.

### 2.3 Phase 3: Keepalive and Health Monitoring

- Client sends a `ping` message every 15 seconds containing `clientTimestampMs`.
- Server responds immediately with a `pong` message containing `clientTimestampMs` and `serverTimestampMs`.
- If the client does not receive a `pong` within 5000 milliseconds of sending a `ping`, it marks the connection degraded.
- If the server receives no frames or pings from a client for 30 consecutive seconds, the server terminates the session.

### 2.4 Phase 4: Session Teardown

- Clean termination: Either party transmits a WebSocket close frame with code `1000` (Normal Closure) or sends an explicit `session_close` event.
- The server releases worker GPU contexts, flushes remaining telemetry to disk, and purges transient audio/video buffers.

---

## 3. Protocol Framing and Envelopes

### 3.1 JSON Message Envelope

All text frames sent over the WebSocket follow the canonical envelope structure:

```ts
export type RealtimeMessageType =
  | "session_init"
  | "session_ready"
  | "frame_landmarks"
  | "audio_chunk"
  | "sign_detected"
  | "transcript_update"
  | "translation_result"
  | "tts_audio"
  | "error"
  | "ping"
  | "pong";

export interface RealtimeMessage<T = unknown> {
  type: RealtimeMessageType;
  sessionId: string;
  timestampMs: number;
  payload: T;
  sequence?: number;
  traceId?: string;
}
```

| Field | Type | Required | Description |
|---|---|---|---|
| `type` | `RealtimeMessageType` | Yes | Discriminator string identifying the message payload schema |
| `sessionId` | `string` | Yes | Active session UUID |
| `timestampMs` | `number` | Yes | Monotonic millisecond timestamp at packet creation |
| `payload` | `T` | Yes | Type-specific data object |
| `sequence` | `number` | No | Monotonically increasing 32-bit unsigned integer per session for ordering |
| `traceId` | `string` | No | Distributed trace identifier for cross-service observability |

### 3.2 High-Throughput Binary Framing

For scenarios where base64-encoded JSON adds unacceptable network or serialization overhead (such as high-density 3D landmark arrays or high-frequency raw PCM audio), Converse supports a packed binary frame format:

#### Binary Frame Layout
```text
 0                   1                   2                   3
 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|          Magic (0x4356)       |  Version (1)  | Msg Type (ID) |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|                      Sequence Number (Uint32)                 |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|                    Timestamp Milliseconds (Uint64)            |
|                                                               |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|                     Session UUID (16 Bytes)                   |
|                                                               |
|                                                               |
|                                                               |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|                     Payload Length in Bytes (Uint32)          |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|                     Payload Bytes (Variable)                  |
|                             ...                               |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
```

- **Magic**: Two bytes `0x43 0x56` (ASCII `"CV"`).
- **Version**: One byte unsigned integer (currently `1`).
- **Msg Type ID**: One byte enum (`1` = `frame_landmarks`, `2` = `audio_chunk`, `3` = `tts_audio`).
- **Sequence Number**: 4 bytes unsigned big-endian integer.
- **Timestamp Milliseconds**: 8 bytes unsigned big-endian integer.
- **Session UUID**: 16 bytes raw binary UUID.
- **Payload Length**: 4 bytes unsigned big-endian integer.
- **Payload Bytes**: Raw uncompressed bytes (for example IEEE-754 32-bit floats for landmarks or signed 16-bit PCM for audio).

---

## 4. Comprehensive Payload Schemas and Wire Examples

### 4.1 `session_init`
- **Direction**: Client to Server
- **Purpose**: Establishes session intent, pipeline direction, client metadata, and media parameters.

#### Schema
```ts
export interface SessionInitPayload {
  direction: "sign_to_speech" | "speech_to_sign";
  clientId: string;
  sampleRate?: number;
  videoFps?: number;
}
```

#### Wire Example
```json
{
  "type": "session_init",
  "sessionId": "sess_d3a8e941-487b-4029-bb11-1c5c99e984f1",
  "timestampMs": 1726580000000,
  "sequence": 1,
  "payload": {
    "direction": "sign_to_speech",
    "clientId": "web-client-v0.1.0",
    "sampleRate": 16000,
    "videoFps": 30
  }
}
```

---

### 4.2 `session_ready`
- **Direction**: Server to Client
- **Purpose**: Confirms session establishment, returns allocated parameters and negotiated capabilities.

#### Schema
```ts
export interface SessionReadyPayload {
  sessionId: string;
  assignedDirection: "sign_to_speech" | "speech_to_sign";
  heartbeatIntervalMs: number;
  protocolVersion: string;
  capabilities: string[];
}
```

#### Wire Example
```json
{
  "type": "session_ready",
  "sessionId": "sess_d3a8e941-487b-4029-bb11-1c5c99e984f1",
  "timestampMs": 1726580000050,
  "sequence": 1,
  "payload": {
    "sessionId": "sess_d3a8e941-487b-4029-bb11-1c5c99e984f1",
    "assignedDirection": "sign_to_speech",
    "heartbeatIntervalMs": 15000,
    "protocolVersion": "1.0.0",
    "capabilities": [
      "landmarks_v1",
      "audio_pcm16",
      "translation_continuous",
      "streaming_tts"
    ]
  }
}
```

---

### 4.3 `frame_landmarks`
- **Direction**: Client to Server (or Vision worker to Gateway)
- **Purpose**: Ingests normalized 3D hand and body landmarks extracted from a single video frame.

#### Schema
```ts
export interface Point3D {
  x: number;
  y: number;
  z: number;
  visibility?: number;
}

export interface HandLandmarks {
  landmarks: Point3D[];
  handedness: "left" | "right";
  confidence: number;
}

export interface PoseLandmarks {
  landmarks: Point3D[];
}

export interface FrameLandmarksPayload {
  frameId: number;
  timestampMs: number;
  leftHand?: HandLandmarks;
  rightHand?: HandLandmarks;
  pose?: PoseLandmarks;
}
```

#### Wire Example
```json
{
  "type": "frame_landmarks",
  "sessionId": "sess_d3a8e941-487b-4029-bb11-1c5c99e984f1",
  "timestampMs": 1726580000100,
  "sequence": 24,
  "payload": {
    "frameId": 182,
    "timestampMs": 1726580000100,
    "rightHand": {
      "handedness": "right",
      "confidence": 0.96,
      "landmarks": [
        { "x": 0.482, "y": 0.612, "z": -0.012, "visibility": 0.99 },
        { "x": 0.495, "y": 0.589, "z": -0.015, "visibility": 0.98 }
      ]
    },
    "pose": {
      "landmarks": [
        { "x": 0.501, "y": 0.220, "z": 0.000, "visibility": 0.99 }
      ]
    }
  }
}
```

---

### 4.4 `audio_chunk`
- **Direction**: Client to Server (Speech-to-Sign)
- **Purpose**: Streams raw or compressed audio segments for real-time speech recognition.

#### Schema
```ts
export interface AudioChunkPayload {
  sequence: number;
  timestampMs: number;
  format: "pcm_s16le" | "opus" | "wav";
  sampleRate: number;
  channels: number;
  audioBase64: string;
  isFinal: boolean;
}
```

#### Wire Example
```json
{
  "type": "audio_chunk",
  "sessionId": "sess_d3a8e941-487b-4029-bb11-1c5c99e984f1",
  "timestampMs": 1726580000300,
  "sequence": 6,
  "payload": {
    "sequence": 6,
    "timestampMs": 1726580000300,
    "format": "pcm_s16le",
    "sampleRate": 16000,
    "channels": 1,
    "audioBase64": "A8j///v8AgD6/P8A/v///wIA//8BAAAAAAD//w==",
    "isFinal": false
  }
}
```

---

### 4.5 `sign_detected`
- **Direction**: Server to Client
- **Purpose**: Emitted when the temporal vision model recognizes a distinct sign gesture.

#### Schema
```ts
export interface SignDetectedPayload {
  gloss: string;
  confidence: number;
  startTimeMs: number;
  endTimeMs: number;
}
```

#### Wire Example
```json
{
  "type": "sign_detected",
  "sessionId": "sess_d3a8e941-487b-4029-bb11-1c5c99e984f1",
  "timestampMs": 1726580001200,
  "sequence": 89,
  "payload": {
    "gloss": "HELLO",
    "confidence": 0.942,
    "startTimeMs": 350,
    "endTimeMs": 1150
  }
}
```

---

### 4.6 `transcript_update`
- **Direction**: Server to Client
- **Purpose**: Emitted by the ASR service as speech hypotheses are formed and finalized.

#### Schema
```ts
export interface TranscriptUpdatePayload {
  transcript: string;
  isFinal: boolean;
  confidence: number;
  durationMs?: number;
  startTimeMs?: number;
  endTimeMs?: number;
}
```

#### Wire Example (Partial)
```json
{
  "type": "transcript_update",
  "sessionId": "sess_d3a8e941-487b-4029-bb11-1c5c99e984f1",
  "timestampMs": 1726580000850,
  "sequence": 14,
  "payload": {
    "transcript": "where are",
    "isFinal": false,
    "confidence": 0.88,
    "startTimeMs": 100,
    "endTimeMs": 850
  }
}
```

#### Wire Example (Final)
```json
{
  "type": "transcript_update",
  "sessionId": "sess_d3a8e941-487b-4029-bb11-1c5c99e984f1",
  "timestampMs": 1726580001500,
  "sequence": 15,
  "payload": {
    "transcript": "Where are you going?",
    "isFinal": true,
    "confidence": 0.965,
    "durationMs": 1400,
    "startTimeMs": 100,
    "endTimeMs": 1500
  }
}
```

---

### 4.7 `translation_result`
- **Direction**: Server to Client
- **Purpose**: Emitted when the Translation Engine synthesizes natural text from signs, or sign tokens from speech.

#### Schema
```ts
export interface SignToken {
  gloss: string;
  durationMs: number;
  emphasis?: boolean;
}

export interface TranslationResultPayload {
  direction: "sign_to_speech" | "speech_to_sign";
  text?: string;
  tokens?: SignToken[];
  confidence: number;
  status: "partial" | "final";
  latencyMs: number;
}
```

#### Wire Example (Sign to Speech Translation)
```json
{
  "type": "translation_result",
  "sessionId": "sess_d3a8e941-487b-4029-bb11-1c5c99e984f1",
  "timestampMs": 1726580001650,
  "sequence": 92,
  "payload": {
    "direction": "sign_to_speech",
    "text": "Hello, nice to meet you.",
    "confidence": 0.931,
    "status": "final",
    "latencyMs": 35
  }
}
```

#### Wire Example (Speech to Sign Translation)
```json
{
  "type": "translation_result",
  "sessionId": "sess_d3a8e941-487b-4029-bb11-1c5c99e984f1",
  "timestampMs": 1726580001650,
  "sequence": 93,
  "payload": {
    "direction": "speech_to_sign",
    "tokens": [
      { "gloss": "HELLO", "durationMs": 400, "emphasis": false },
      { "gloss": "NICE", "durationMs": 350, "emphasis": false },
      { "gloss": "MEET", "durationMs": 400, "emphasis": false },
      { "gloss": "YOU", "durationMs": 300, "emphasis": false }
    ],
    "confidence": 0.945,
    "status": "final",
    "latencyMs": 28
  }
}
```

---

### 4.8 `tts_audio`
- **Direction**: Server to Client
- **Purpose**: Streams synthesized speech audio chunks generated from the translated English text.

#### Schema
```ts
export interface TtsAudioPayload {
  sequence: number;
  audioBase64: string;
  audioFormat: "wav" | "mp3" | "pcm_s16le";
  durationMs: number;
  isFinal: boolean;
}
```

#### Wire Example
```json
{
  "type": "tts_audio",
  "sessionId": "sess_d3a8e941-487b-4029-bb11-1c5c99e984f1",
  "timestampMs": 1726580001850,
  "sequence": 95,
  "payload": {
    "sequence": 1,
    "audioFormat": "pcm_s16le",
    "durationMs": 350,
    "isFinal": false,
    "audioBase64": "UklGRiQAAABXQVZFZm10IBAAAAABAAEAQB8AAEAfAAABAAgAZGF0YQAAAAA="
  }
}
```

---

### 4.9 `error`
- **Direction**: Bidirectional (Server to Client or Client to Server)
- **Purpose**: Transmits structured domain errors conforming to the `Result`/`DomainError` standard.

#### Schema
```ts
export interface RealtimeErrorPayload {
  component:
    | "camera"
    | "vision"
    | "asr"
    | "translation"
    | "tts"
    | "renderer"
    | "network"
    | "protocol";
  code: string;
  message: string;
  recoverable: boolean;
  details?: Record<string, unknown>;
}
```

#### Wire Example
```json
{
  "type": "error",
  "sessionId": "sess_d3a8e941-487b-4029-bb11-1c5c99e984f1",
  "timestampMs": 1726580002100,
  "sequence": 98,
  "payload": {
    "component": "vision",
    "code": "LOW_DETECTION_CONFIDENCE",
    "message": "Hand landmarks confidence fell below 0.30 threshold",
    "recoverable": true,
    "details": {
      "observedConfidence": 0.22,
      "recommendedAction": "ADJUST_LIGHTING"
    }
  }
}
```

---

### 4.10 `ping` and `pong`
- **Direction**: `ping` (Client to Server), `pong` (Server to Client)
- **Purpose**: Liveness verification, connection heartbeating, and round-trip time (RTT) calculation.

#### Schemas
```ts
export interface PingPayload {
  clientTimestampMs: number;
}

export interface PongPayload {
  clientTimestampMs: number;
  serverTimestampMs: number;
}
```

#### Wire Example (`ping`)
```json
{
  "type": "ping",
  "sessionId": "sess_d3a8e941-487b-4029-bb11-1c5c99e984f1",
  "timestampMs": 1726580015000,
  "payload": {
    "clientTimestampMs": 1726580015000
  }
}
```

#### Wire Example (`pong`)
```json
{
  "type": "pong",
  "sessionId": "sess_d3a8e941-487b-4029-bb11-1c5c99e984f1",
  "timestampMs": 1726580015018,
  "payload": {
    "clientTimestampMs": 1726580015000,
    "serverTimestampMs": 1726580015018
  }
}
```

---

## 5. Pipeline Concurrency and Latency Budget

### 5.1 End-to-End Latency Target

To achieve natural, conversational interaction, the end-to-end latency target from physical gesture or speech utterance to received translation is:

| Pipeline Stage | Target Latency (Sign -> Speech) | Target Latency (Speech -> Sign) |
|---|---|---|
| Ingestion & Transport | 25 ms | 25 ms |
| Feature Extraction / ASR | 80 ms (Vision temporal window) | 120 ms (Streaming ASR partial) |
| Translation Engine | 35 ms | 25 ms |
| Synthesis / Render prep | 60 ms (TTS first-chunk) | 30 ms (Avatar animation init) |
| Network Return Transport | 25 ms | 25 ms |
| **Total Response Time** | **225 ms** | **225 ms** |

### 5.2 Speculative Pipeline Overlap

Sequential batch processing results in additive latency (`T_total = T_ingest + T_recognize + T_translate + T_synthesize = 500ms+`). Converse utilizes speculative pipeline concurrency to overlap stages:

```text
Time (ms)  0     50    100   150   200   250   300   350
Ingestion: ████████████████████████████
ASR/Vision:      ██████████████████████
Translation:           ████████████████
TTS / Render:                ██████████
Audio Playback:                    ██████████████
```

1. **Intermediate Partial Translation**: When ASR outputs a high-confidence partial transcript (`isFinal: false`), the Translation Engine speculatively pre-computes the target token structure.
2. **Commit Boundary**: When the speech or sign recognizer emits `isFinal: true`, the Translation Engine immediately flushes the pre-warmed output, saving up to 150 milliseconds of sequential processing time.
3. **Chunked TTS Streaming**: TTS does not buffer the full translated sentence. The synthesis engine processes phrase by phrase, streaming 100-millisecond audio chunks to the client as they are generated.

---

## 6. Error Handling, Reconnection, and Fallback Strategy

### 6.1 Error Classification

- **Recoverable Errors (`recoverable: true`)**: Transient conditions such as frame drops, low lighting, temporary network jitter, or context buffer pruning. The session remains open; the client applies UI guidance (for example lighting prompts) without restarting the session.
- **Fatal Errors (`recoverable: false`)**: Unrecoverable failures such as invalid session tokens, incompatible codec configurations, hardware access denial, or server worker crashes. The session is closed; the client must re-authenticate or re-initialize.

### 6.2 Reconnection Protocol with Exponential Backoff

If the WebSocket abruptly disconnects:
1. The client immediately preserves the existing session identifier `sessionId` and last acknowledged `sequence` number.
2. The client attempts reconnection using exponential backoff with jitter:
   ```text
   t_retry = min(t_max, t_base * (2 ^ attempt)) ± jitter
   ```
   Where `t_base = 500ms`, `t_max = 10000ms`, and `jitter = uniform(-0.2, 0.2) * t_retry`.
3. Upon reconnecting, the client sends `session_init` including the preserved `sessionId`. If the server still maintains the session context, it replies with `session_ready` and resumes processing without losing dialogue history.

### 6.3 Graceful Degradation Hierarchy

When transport conditions or client devices are constrained:
1. **Tier 1 (Optimal)**: WebRTC Video/Audio Media Tracks + WebRTC DataChannel for landmarks.
2. **Tier 2 (Standard)**: WebSocket binary stream (client extracts landmarks locally via MediaPipe and streams binary landmark frames).
3. **Tier 3 (Constrained)**: WebSocket JSON stream with dynamic frame-rate reduction (dropping landmark stream from 30 FPS to 15 FPS).
4. **Tier 4 (Fallback)**: Chunked HTTP REST requests (`/api/sign-to-speech/translate` and `/api/speech/asr`).
