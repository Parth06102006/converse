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
