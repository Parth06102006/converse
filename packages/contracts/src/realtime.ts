import type { DomainError } from "./result.js";
import type { AslGlossToken, SignRepresentation, SignToken } from "./translation.js";
import type { FrameLandmarks, SignDetection } from "./vision.js";

export type RealtimeDirection = "sign_to_speech" | "speech_to_sign";

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

export interface SessionInitPayload {
  direction: RealtimeDirection;
  clientId: string;
  sampleRate?: number;
  videoFps?: number;
}

export interface SessionReadyPayload {
  sessionId: string;
  assignedDirection: RealtimeDirection;
  heartbeatIntervalMs: number;
  protocolVersion: string;
  capabilities: string[];
}

export type FrameLandmarksPayload = FrameLandmarks;

export interface AudioChunkPayload {
  sequence: number;
  timestampMs: number;
  format: "pcm_s16le" | "opus" | "wav";
  sampleRate: number;
  channels: number;
  audioBase64: string;
  isFinal: boolean;
}

export type SignDetectedPayload = SignDetection;

export interface TranscriptUpdatePayload {
  transcript: string;
  isFinal: boolean;
  confidence: number;
  durationMs?: number;
  startTimeMs?: number;
  endTimeMs?: number;
}

export interface TranslationResultPayload {
  direction: RealtimeDirection;
  text?: string;
  tokens?: SignToken[];
  aslTokens?: AslGlossToken[];
  representation?: SignRepresentation;
  confidence: number;
  status: "partial" | "final";
  latencyMs: number;
}

export interface TtsAudioPayload {
  sequence: number;
  audioBase64: string;
  audioFormat: "wav" | "mp3" | "pcm_s16le";
  durationMs: number;
  isFinal: boolean;
}

export type RealtimeErrorPayload = DomainError;

export interface PingPayload {
  clientTimestampMs: number;
}

export interface PongPayload {
  clientTimestampMs: number;
  serverTimestampMs: number;
}
