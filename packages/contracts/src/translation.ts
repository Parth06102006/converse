import type { SignDetection } from "./vision.js";

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

export interface SignToken {
  gloss: string;
  durationMs: number;
  emphasis?: boolean;
}

export interface TextToSignResponse {
  tokens: SignToken[];
  totalDurationMs: number;
  latencyMs: number;
}
