# Converse Master Contracts Specification

## 1. Document Overview and Scope

This document specifies the authoritative data contracts, interface definitions, error handling semantics, and validation invariants for the Converse communication platform. Converse provides bidirectional real-time translation between American Sign Language (ASL) and spoken English.

The contracts defined herein govern all interactions across system boundaries:

1. Client applications (Web, Chrome Extension, Mobile) interacting with backend gateway services.
2. Inter-service remote procedure calls between the Node.js API gateway and Python ML inference microservices.
3. Real-time event streams transmitted over WebSocket connections and WebRTC data channels.

Adherence to these contracts ensures that model engineers, backend service developers, and frontend application engineers can implement, test, and deploy components independently without breaking end-to-end compatibility.

---

## 2. Architectural Principles and Invariants

### 2.1 Separation of Logical Capabilities and Physical Topology

Logical component boundaries do not dictate physical deployment boundaries. The system exposes five logical capabilities:

1. **ASL Vision**: Extracts 3D spatial landmarks and classifies temporal sign sequences from video frames.
2. **ASR (Automatic Speech Recognition)**: Transcribes streaming or buffered audio into natural English text.
3. **Translation Engine**: Performs bidirectional translation between ASL representations (gloss sequences, spatial tokens) and grammatical English text.
4. **TTS (Text-to-Speech)**: Synthesizes spoken audio from English text.
5. **Sign Renderer**: Converts structured ASL tokens into avatar animations or synthesized video streams.

In monolithic local development, all capabilities may execute on a single host. In production, ASL Vision and ASR execute on GPU-accelerated worker nodes, while the API gateway handles session orchestration and client routing. The contracts defined in this document remain invariant regardless of deployment topology.

### 2.2 Core Pipeline Architectures

The Converse engine operates along two primary communication axes:

#### Sign-to-Speech Pipeline

```text
Camera
  │
  ▼
Video Frames / Landmarks
  │
  ▼
ASL Vision (Landmark Extraction + Temporal Recognition)
  │
  ▼
Sign Detection / ASL Representation
  │
  ▼
Translation Engine (ASL -> English)
  │
  ▼
English Text
  │
  ▼
TTS (Text-to-Speech Synthesis)
  │
  ▼
Audio Stream
```

#### Speech-to-Sign Pipeline

```text
Microphone
  │
  ▼
Audio Chunks
  │
  ▼
ASR (Speech Recognition)
  │
  ▼
English Text
  │
  ▼
Translation Engine (English -> ASL)
  │
  ▼
Sign Tokens / ASL Representation
  │
  ▼
Sign Renderer (Avatar / Animation Engine)
  │
  ▼
Visual Sign Output
```

### 2.3 Partial Versus Final Results

Real-time human conversation cannot tolerate end-of-utterance batch latency. The contracts therefore distinguish between:

- **Partial results (`status: "partial"`, `isFinal: false`)**: Ephemeral, speculative hypotheses emitted as audio chunks or video frames arrive. Downstream stages use partials for predictive pre-warming and speculative translation. Clients display partials with visual distinction.
- **Final results (`status: "final"`, `isFinal: true`)**: Authoritative, committed segments produced when an acoustic pause or gestural rest is detected. Downstream stages commit translations and trigger permanent audio synthesis.

---

## 3. Rust-Inspired Result and Option Semantics

To eliminate unhandled runtime exceptions, inconsistent null values, and silent error propagation across language boundaries (TypeScript, Python, C++ WebRTC workers), all boundary operations and fallible contracts enforce explicit Rust-inspired `Result` and `Option` algebraic types.

### 3.1 Type Definitions

#### Result Type

A discriminated union that forces the caller to explicitly handle success and failure paths:

```ts
export type Result<T, E = DomainError> =
  | { readonly ok: true; readonly value: T }
  | { readonly ok: false; readonly error: E };
```

#### Option Type

A discriminated union that represents the explicit presence or absence of a value without ambiguous `null` or `undefined` semantics:

```ts
export type Option<T> =
  { readonly some: true; readonly value: T } | { readonly some: false };
```

### 3.2 Constructor Helpers

```ts
export function ok<T>(value: T): Result<T, never> {
  return { ok: true, value };
}

export function err<E>(error: E): Result<never, E> {
  return { ok: false, error };
}

export function some<T>(value: T): Option<T> {
  return { some: true, value };
}

export function none(): Option<never> {
  return { some: false };
}
```

### 3.3 Domain Error Taxonomy

All operational failures crossing subsystem boundaries conform to `DomainError`:

```ts
export type ErrorComponent =
  | "camera"
  | "vision"
  | "asr"
  | "translation"
  | "tts"
  | "renderer"
  | "network"
  | "protocol";

export interface DomainError {
  component: ErrorComponent;
  code: string;
  message: string;
  recoverable: boolean;
  details?: Record<string, unknown>;
}
```

### 3.4 Standard Error Codes Catalog

| Component     | Code                       | Recoverable | Description                                     | Client Action                                   |
| ------------- | -------------------------- | ----------- | ----------------------------------------------- | ----------------------------------------------- |
| `camera`      | `DEVICE_NOT_FOUND`         | false       | No video capture device accessible              | Prompt user to connect camera                   |
| `camera`      | `PERMISSION_DENIED`        | false       | OS or browser denied camera permission          | Prompt user to grant permission                 |
| `vision`      | `NO_PERSON_DETECTED`       | true        | Pose detection found zero subjects              | Prompt user to stand in front of camera         |
| `vision`      | `HANDS_OUT_OF_BOUNDS`      | true        | Hands exit camera capture frustum               | Display guide box overlay                       |
| `vision`      | `LOW_DETECTION_CONFIDENCE` | true        | Landmark confidence below operational threshold | Suggest improving room lighting                 |
| `vision`      | `MODEL_INFERENCE_TIMEOUT`  | true        | Vision inference deadline exceeded              | Drop intermediate frame and continue            |
| `asr`         | `AUDIO_BUFFER_OVERFLOW`    | true        | Ingestion rate exceeds processing capacity      | Apply client-side throttling                    |
| `asr`         | `UNSUPPORTED_AUDIO_FORMAT` | false       | Audio codec or sample rate unsupported          | Reconfigure capture settings to 16kHz mono      |
| `asr`         | `SPEECH_ININTEL`           | true        | Speech signal indistinct or masked by noise     | Prompt speaker to speak clearly                 |
| `translation` | `OUT_OF_VOCABULARY`        | true        | Sign gloss or English word not in lexicon       | Fall back to finger-spelling or nearest synonym |
| `translation` | `CONTEXT_WINDOW_EXCEEDED`  | true        | Conversation context buffer full                | Prune oldest context turns                      |
| `translation` | `TRANSLATION_TIMEOUT`      | true        | Translation model latency threshold exceeded    | Emit fallback literal gloss match               |
| `tts`         | `VOICE_NOT_FOUND`          | true        | Requested TTS voice profile unavailable         | Fall back to default system voice               |
| `tts`         | `SYNTHESIS_FAILED`         | true        | Synthesis engine failure                        | Re-request synthesis with plain text            |
| `renderer`    | `AVATAR_LOAD_FAILED`       | false       | 3D model assets failed to initialize            | Fall back to 2D skeleton rendering              |
| `protocol`    | `SESSION_EXPIRED`          | false       | WebSocket session idle timeout reached          | Re-initialize session via `session_init`        |
| `protocol`    | `MALFORMED_FRAME`          | true        | Inbound payload failed schema validation        | Log warning and discard packet                  |
| `protocol`    | `RATE_LIMIT_EXCEEDED`      | true        | Client exceeds allowable frame/audio rate       | Back off transmission rate                      |

### 3.5 Cross-Language Result Mapping

In TypeScript, consumers unwrap results via discriminated union checks:

```ts
const translationResult = await translateSignToText(request);
if (!translationResult.ok) {
  handleTranslationError(translationResult.error);
  return;
}
renderEnglishTranscript(translationResult.value.englishText);
```

In Python ML workers, equivalent Pydantic models enforce the same serialization contract:

```python
from typing import Generic, Optional, TypeVar, Union
from pydantic import BaseModel

T = TypeVar("T")
E = TypeVar("E", bound="DomainError")

class DomainError(BaseModel):
    component: str
    code: str
    message: str
    recoverable: bool
    details: Optional[dict] = None

class Ok(BaseModel, Generic[T]):
    ok: bool = True
    value: T

class Err(BaseModel, Generic[E]):
    ok: bool = False
    error: E

Result = Union[Ok[T], Err[DomainError]]
```

---

## 4. Canonical Data Contracts

### 4.1 Media and Vision Schemas

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

export interface FrameLandmarks {
  frameId: number;
  timestampMs: number;
  leftHand?: HandLandmarks;
  rightHand?: HandLandmarks;
  pose?: PoseLandmarks;
}

export interface SignDetection {
  gloss: string;
  confidence: number;
  startTimeMs: number;
  endTimeMs: number;
  id?: string;
  durationMs?: number;
  isFingerspelled?: boolean;
  handDominance?: "left" | "right";
}
```

#### Coordinate System Invariant

All `x` and `y` coordinates in `Point3D` are normalized floats in the range `[0.0, 1.0]`, relative to image width and height. Origin `(0.0, 0.0)` is the top-left corner of the frame. Coordinate `z` represents depth relative to the wrist or mid-hip landmark, scaled identically to `x`.

### 4.2 Sign Representation and Translation Schemas

```ts
export interface SignToken {
  gloss: string;
  durationMs: number;
  emphasis?: boolean;
}

export interface NonManualMarkers {
  eyebrows: "neutral" | "raised" | "furrowed";
  headMotion: "neutral" | "nod" | "shake" | "tilt_forward";
  mouthMorpheme?: string;
}

export interface AslGlossToken {
  gloss: string;
  lemma: string;
  partOfSpeech: string;
  nonManualMarkers: NonManualMarkers;
  spatialLocus?: "neutral_space" | "chest" | "forehead" | "left" | "right";
  isFingerspelled: boolean;
  fingerspellSequence?: string[];
}

export interface SignTokenTiming {
  startTimeMs: number;
  leadInDurationMs: number;
  holdDurationMs: number;
  leadOutDurationMs: number;
}

export interface SpatialLociTarget {
  anchor: "neutral_space" | "chest" | "forehead" | "left" | "right";
  targetOffset: Point3D;
}

export interface SignRepresentationToken {
  tokenId: string;
  clipId: string;
  gloss: string;
  timing: SignTokenTiming;
  spatialLoci: SpatialLociTarget;
  nonManualMarkers: {
    eyebrowIntensity: number;
    eyebrowShape: "furrow" | "raise" | "neutral";
    headRotation: {
      pitch: number;
      yaw: number;
      roll: number;
    };
    mouthShape: string;
  };
  interpolationCurve: "linear" | "ease_in_out" | "bezier_slerp";
}

export interface SignRepresentation {
  version: string;
  sessionId: string;
  utteranceId: string;
  totalDurationMs: number;
  tokens: SignRepresentationToken[];
}

export interface SignToTextRequest {
  detections: SignDetection[];
  sessionId?: string;
  context?: string;
}

export interface SignToTextResponse {
  englishText: string;
  confidence: number;
  glosses: string[];
  latencyMs: number;
}

export interface TextToSignRequest {
  englishText: string;
  sessionId?: string;
}

export interface TextToSignResponse {
  tokens: SignToken[];
  aslTokens?: AslGlossToken[];
  representation?: SignRepresentation;
  totalDurationMs: number;
  latencyMs: number;
}
```

### 4.3 Speech and Audio Schemas

```ts
export interface WordTimestamp {
  word: string;
  startMs: number;
  endMs: number;
  confidence?: number;
}

export interface AsrLatencyMetrics {
  audioDurationMs: number;
  processingTimeMs: number;
}

export interface AsrTranscriptEvent {
  sessionId: string;
  sequenceId: number;
  text: string;
  isFinal: boolean;
  confidence: number;
  wordTimestamps?: WordTimestamp[];
  latencyMetrics: AsrLatencyMetrics;
}

export type AsrBackend = "aws" | "whisper";

export interface AsrRequest {
  audioBase64?: string;
  audioFormat?: "wav" | "webm" | "pcm" | "pcm_s16le";
  sampleRate?: number;
  sessionId?: string;
  backend?: AsrBackend;
}

export interface AsrResponse {
  transcript: string;
  isFinal: boolean;
  confidence: number;
  durationMs: number;
}

export interface TtsRequest {
  text: string;
  voice?: string;
  speed?: number;
}

export interface TtsResponse {
  audioBase64: string;
  audioFormat: "wav" | "mp3";
  durationMs: number;
}
```

#### Audio Canonical Standard

The canonical uncompressed internal audio representation across Converse is:

- **Encoding**: Signed 16-bit linear PCM (`pcm_s16le`)
- **Sample Rate**: 16,000 Hz
- **Channels**: 1 (Mono)
- **Byte Order**: Little-endian

### 4.4 Health and Service Metadata Schemas

```ts
export type ServiceStatus = "ok" | "degraded" | "error";

export interface HealthCheckResponse {
  status: ServiceStatus;
  service: string;
  version?: string;
  uptimeSeconds?: number;
  timestamp: string;
}

export interface ServiceEndpointInfo {
  path: string;
  method: "GET" | "POST" | "PUT" | "DELETE";
  description: string;
}

export interface ServiceCatalogResponse {
  name: string;
  system: string;
  version: string;
  pipelines: {
    signToSpeech: string;
    speechToSign: string;
  };
  endpoints: ServiceEndpointInfo[];
}
```

---

## 5. REST API Contract Specification

Base URL: `http://<host>:<port>`

Standard Headers:

- `Content-Type: application/json`
- `Accept: application/json`
- `X-Session-ID`: Optional string identifier associating request with ongoing conversation session.
- `X-Request-ID`: UUID for distributed request tracing.

Standard Error Response Envelope:

```json
{
  "ok": false,
  "error": {
    "component": "asr",
    "code": "UNSUPPORTED_AUDIO_FORMAT",
    "message": "Audio format must be wav, webm, or pcm at 16000Hz",
    "recoverable": false,
    "details": {
      "providedFormat": "mp3",
      "supportedFormats": ["wav", "webm", "pcm"]
    }
  }
}
```

---

### 5.1 `GET /health`

Probes runtime availability, component status, and operational uptime.

- **Method**: `GET`
- **Authentication**: None
- **Query Parameters**: None
- **Success Response Code**: `200 OK` (when status is `ok` or `degraded`), `503 Service Unavailable` (when status is `error`)
- **Response Type**: `HealthCheckResponse`

#### Response Example

```json
{
  "status": "ok",
  "service": "@converse/api",
  "version": "0.1.0",
  "uptimeSeconds": 14285,
  "timestamp": "2026-09-17T19:50:00.000Z"
}
```

---

### 5.2 `GET /api/info`

Returns the service catalog, architecture description, active pipelines, and supported endpoint specifications.

- **Method**: `GET`
- **Authentication**: None
- **Query Parameters**: None
- **Success Response Code**: `200 OK`
- **Response Type**: `ServiceCatalogResponse`

#### Response Example

```json
{
  "name": "Converse API",
  "system": "ASL <-> English Communication Engine",
  "version": "0.1.0",
  "pipelines": {
    "signToSpeech": "Camera -> ASL Vision -> ASL Representation -> Translation -> English Text -> TTS -> Audio",
    "speechToSign": "Audio -> ASR -> English Text -> Translation -> ASL Representation -> Sign Renderer"
  },
  "endpoints": [
    {
      "path": "/health",
      "method": "GET",
      "description": "Service health and readiness probe"
    },
    {
      "path": "/api/info",
      "method": "GET",
      "description": "Service catalog and capability descriptions"
    },
    {
      "path": "/api/sign-to-speech/translate",
      "method": "POST",
      "description": "Translates accumulated ASL sign detections to fluent English text"
    },
    {
      "path": "/api/speech-to-sign/translate",
      "method": "POST",
      "description": "Translates English text into sequenced ASL sign tokens for rendering"
    },
    {
      "path": "/api/speech/asr",
      "method": "POST",
      "description": "Transcribes audio speech samples to text"
    },
    {
      "path": "/api/speech/tts",
      "method": "POST",
      "description": "Synthesizes spoken audio waveform from text"
    }
  ]
}
```

---

### 5.3 `POST /api/sign-to-speech/translate`

Translates a sequence of recognized ASL sign detections into grammatical English text.

- **Method**: `POST`
- **Request Type**: `SignToTextRequest`
- **Success Response Code**: `200 OK`
- **Response Type**: `SignToTextResponse`

#### Request Schema

| Field        | Type              | Required | Description                                                     |
| ------------ | ----------------- | -------- | --------------------------------------------------------------- |
| `detections` | `SignDetection[]` | Yes      | Ordered array of recognized sign glosses with timing boundaries |
| `sessionId`  | `string`          | No       | Identifier for conversational context retention                 |
| `context`    | `string`          | No       | Optional prior turn dialogue context for disambiguation         |

#### Request Example

```json
{
  "sessionId": "sess_8f9a2b",
  "context": "Previous topic: dining",
  "detections": [
    {
      "gloss": "WHERE",
      "confidence": 0.94,
      "startTimeMs": 100,
      "endTimeMs": 550
    },
    {
      "gloss": "STORE",
      "confidence": 0.91,
      "startTimeMs": 600,
      "endTimeMs": 1100
    }
  ]
}
```

#### Response Example

```json
{
  "englishText": "Where is the store?",
  "confidence": 0.925,
  "glosses": ["WHERE", "STORE"],
  "latencyMs": 18
}
```

---

### 5.4 `POST /api/speech-to-sign/translate`

Translates English text into grammatical ASL sign tokens with duration and timing directives for avatar rendering.

- **Method**: `POST`
- **Request Type**: `TextToSignRequest`
- **Success Response Code**: `200 OK`
- **Response Type**: `TextToSignResponse`

#### Request Schema

| Field         | Type     | Required | Description                                    |
| ------------- | -------- | -------- | ---------------------------------------------- |
| `englishText` | `string` | Yes      | The natural language English text to translate |
| `sessionId`   | `string` | No       | Conversation session identifier                |

#### Request Example

```json
{
  "sessionId": "sess_8f9a2b",
  "englishText": "I am going to the store"
}
```

#### Response Example

```json
{
  "tokens": [
    {
      "gloss": "STORE",
      "durationMs": 450,
      "emphasis": false
    },
    {
      "gloss": "ME",
      "durationMs": 300,
      "emphasis": false
    },
    {
      "gloss": "GO",
      "durationMs": 500,
      "emphasis": true
    }
  ],
  "totalDurationMs": 1250,
  "latencyMs": 14
}
```

---

### 5.5 `POST /api/speech/asr`

Transcribes an audio chunk or complete utterance into text.

- **Method**: `POST`
- **Request Type**: `AsrRequest`
- **Success Response Code**: `200 OK`
- **Response Type**: `AsrResponse`

#### Request Schema

| Field         | Type     | Required | Default | Description                              |
| ------------- | -------- | -------- | ------- | ---------------------------------------- |
| `audioBase64` | `string` | Yes      | None    | Base64-encoded audio byte buffer         |
| `audioFormat` | `string` | No       | `"pcm"` | One of `"wav"`, `"webm"`, `"pcm"`        |
| `sampleRate`  | `number` | No       | `16000` | Sample frequency in Hz (typically 16000) |
| `sessionId`   | `string` | No       | None    | Tracking session identifier              |

#### Request Example

```json
{
  "sessionId": "sess_8f9a2b",
  "audioFormat": "wav",
  "sampleRate": 16000,
  "audioBase64": "UklGRiQAAABXQVZFZm10IBAAAAABAAEAQB8AAEAfAAABAAgAZGF0YQAAAAA="
}
```

#### Response Example

```json
{
  "transcript": "Where are you going?",
  "isFinal": true,
  "confidence": 0.96,
  "durationMs": 1820
}
```

---

### 5.6 `POST /api/speech/tts`

Synthesizes spoken audio from English text.

- **Method**: `POST`
- **Request Type**: `TtsRequest`
- **Success Response Code**: `200 OK`
- **Response Type**: `TtsResponse`

#### Request Schema

| Field   | Type     | Required | Default                   | Description                           |
| ------- | -------- | -------- | ------------------------- | ------------------------------------- |
| `text`  | `string` | Yes      | None                      | The English sentence to synthesize    |
| `voice` | `string` | No       | `"en-US-ChristopherNeural"` | Selected voice identifier (gateway default; spec placeholder was `"default-neutral"`) |
| `speed` | `number` | No       | `1.0`                     | Playback rate multiplier (0.5 to 2.0) |

#### Request Example

```json
{
  "text": "Hello, how can I assist you today?",
  "voice": "en-US-Standard-C",
  "speed": 1.0
}
```

#### Response Example

```json
{
  "audioBase64": "UklGRiQAAABXQVZFZm10IBAAAAABAAEAQB8AAEAfAAABAAgAZGF0YQAAAAA=",
  "audioFormat": "wav",
  "durationMs": 2150
}
```

---

### 5.7 Internal ML Service Endpoints (`ml/service.py`, port 5050)

The Python ML microservice backs the gateway proxies above. It is internal (no auth) and returns `{ "ok": true, "data": ... }` envelopes.

| Method | Path | Description |
| ------ | ---- | ----------- |
| `GET` | `/health` | Liveness probe (`converse-ml-speech-to-sign`, `0.1.0`) |
| `POST` | `/internal/speech-to-sign/compile` | Compile `englishText` to `SignRepresentation` (`{ englishText, sessionId }`) |
| `POST` | `/internal/speech/asr` | Transcribe audio (`{ audioBase64, audioFormat, sessionId, backend? }`), per-request backend override without silent fallback |
| `POST` | `/internal/tts/synthesize` | Synthesize speech (`{ text, voice, rate, format }`); returns binary WAV by default or base64 JSON when `Accept: application/json` |

### 5.8 Real-Time Meeting Gateway (`/ws/meeting`)

Implemented in `services/api/src/realtime.ts` (see ADR-011; full message catalog in `docs/api/protocol.md`). Rooms are keyed by `sessionId` with multi-client fan-out:

- `session_init` (query params `sessionId`, `clientId`, `direction` also honored) -> `session_ready` with `heartbeatIntervalMs: 30000` and capabilities (`vision_landmarks`, `asr_streaming`, `sign_translation`, `tts_synthesis`, `threejs_webgl_skeletal_playback`).
- `ping` -> `pong`; `frame_landmarks` relayed to peers.
- `sign_detected` accumulates glosses per room, runs contracts-owned `reconstructSentence`, and broadcasts `transcript_update` + `translation_result` (`sign_to_speech`).
- `audio_chunk` transcribes via the ML service, broadcasts `transcript_update`, compiles speech-to-sign, and broadcasts `translation_result` (`speech_to_sign`) with fallback gloss tokenization when the ML service is unreachable.
- `transcript_update`, `translation_result`, `tts_audio` pass-through broadcasts; malformed JSON and unknown types yield recoverable `error` envelopes.

---

## 6. Backward Compatibility, Schema Versioning, and Validation Invariants

### 6.1 Versioning Policy

All schemas in `@converse/contracts` follow Semantic Versioning (`MAJOR.MINOR.PATCH`):

- **PATCH**: Non-breaking internal adjustments, documentation enhancements, tightening of internal types without modifying wire format.
- **MINOR**: Additive changes to contracts. Adding optional fields with sensible defaults, adding new event types to union types, or exposing new optional endpoints. Backward compatibility is strictly maintained.
- **MAJOR**: Breaking changes. Renaming required fields, changing coordinate normalization formulas, removing event types, or altering serialization formats.

API endpoints are namespaced with version indicators when breaking changes are deployed (for example `/v1/api/...` to `/v2/api/...`).

### 6.2 Additive Evolution Rules

To maintain backward compatibility between differing versions of web clients, mobile apps, and backend services:

1. **Never rename or delete fields in minor versions**: If `englishText` is established, it cannot be renamed to `text` without supporting both simultaneously during a documented deprecation window.
2. **All newly introduced fields must be optional**: New fields must specify sensible default fallback values when omitted by older clients.
3. **Clients must ignore unrecognized fields**: Clients and servers must implement open record parsing (non-strict field stripping) to permit forward-compatible extensions.

### 6.3 Runtime Validation Strategy

TypeScript types exist solely at compile-time. To prevent invalid payloads from breaching system boundaries at runtime, all inputs must be validated against schema validators before processing.

#### Zod Validation (Node.js API Gateway)

```ts
import { z } from "zod";

export const Point3DSchema = z.object({
  x: z.number().min(0.0).max(1.0),
  y: z.number().min(0.0).max(1.0),
  z: z.number(),
  visibility: z.number().min(0.0).max(1.0).optional(),
});

export const HandLandmarksSchema = z.object({
  landmarks: z.array(Point3DSchema).length(21),
  handedness: z.enum(["left", "right"]),
  confidence: z.number().min(0.0).max(1.0),
});

export const SignDetectionSchema = z.object({
  gloss: z.string().min(1),
  confidence: z.number().min(0.0).max(1.0),
  startTimeMs: z.number().nonnegative(),
  endTimeMs: z.number().nonnegative(),
});

export const SignToTextRequestSchema = z.object({
  detections: z.array(SignDetectionSchema),
  sessionId: z.string().optional(),
  context: z.string().optional(),
});
```

#### Pydantic Validation (Python ML Services)

```python
from typing import List, Optional
from pydantic import BaseModel, Field

class Point3DModel(BaseModel):
    x: float = Field(..., ge=0.0, le=1.0)
    y: float = Field(..., ge=0.0, le=1.0)
    z: float
    visibility: Optional[float] = Field(None, ge=0.0, le=1.0)

class HandLandmarksModel(BaseModel):
    landmarks: List[Point3DModel] = Field(..., min_length=21, max_length=21)
    handedness: str = Field(..., regex="^(left|right)$")
    confidence: float = Field(..., ge=0.0, le=1.0)
```

### 6.4 Critical System Invariants

1. **Monotonicity**: In real-time streams, `timestampMs` and packet `sequence` numbers must be strictly monotonically increasing per session. Packet drops are detected when `sequence_n != sequence_{n-1} + 1`.
2. **Bounded Confidence Values**: All confidence metrics across ASR, Vision, and Translation must fall strictly in the closed interval `[0.0, 1.0]`. Values outside this range are rejected with `MALFORMED_FRAME`.
3. **Temporal Invariants**: For all observations, detections, and segments, `startTimeMs <= endTimeMs`. Negative durations are invalid.
4. **Hand Landmark Density**: When present, standard MediaPipe-derived hand landmarks must contain exactly 21 landmarks ordered in canonical joint hierarchy (0: wrist, 1-4: thumb, 5-8: index, 9-12: middle, 13-16: ring, 17-20: pinky).
