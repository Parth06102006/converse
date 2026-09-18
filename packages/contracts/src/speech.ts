export interface AsrRequest {
  audioBase64?: string;
  audioFormat?: "wav" | "webm" | "pcm";
  sampleRate?: number;
  sessionId?: string;
}

export interface AsrResponse {
  transcript: string;
  isFinal: boolean;
  confidence: number;
  durationMs: number;
}

export interface WordTimestamp {
  word: string;
  startMs: number;
  endMs: number;
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
