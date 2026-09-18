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

export type EyebrowMarker = "neutral" | "raised" | "furrowed";
export type HeadMotionMarker = "neutral" | "nod" | "shake" | "tilt_forward";
export type SpatialLocus = "center" | "left" | "right" | "contralateral";

export interface NonManualMarkers {
  eyebrows: EyebrowMarker;
  headMotion: HeadMotionMarker;
  mouthMorpheme?: string;
}

export interface AslGlossToken {
  gloss: string;
  lemma: string;
  partOfSpeech: string;
  nonManualMarkers: NonManualMarkers;
  spatialLocus?: SpatialLocus;
  isFingerspelled: boolean;
  fingerspellSequence?: string[];
}

export type SpatialAnchor =
  | "neutral_space"
  | "chest"
  | "forehead"
  | "left_shoulder"
  | "right_shoulder";

export type InterpolationCurve = "linear" | "ease_in_out" | "bezier_slerp";

export interface SignTokenTiming {
  startTimeMs: number;
  leadInDurationMs: number;
  holdDurationMs: number;
  leadOutDurationMs: number;
}

export interface SpatialLociTarget {
  anchor: SpatialAnchor;
  targetOffset: {
    x: number;
    y: number;
    z: number;
  };
}

export interface NonManualMarkerDirectives {
  eyebrowIntensity: number;
  eyebrowShape: "furrow" | "raise" | "neutral";
  headRotation: {
    pitch: number;
    yaw: number;
    roll: number;
  };
  mouthShape: string;
}

export interface SignRepresentationToken {
  tokenId: string;
  clipId: string;
  gloss: string;
  timing: SignTokenTiming;
  spatialLoci: SpatialLociTarget;
  nonManualMarkers: NonManualMarkerDirectives;
  interpolationCurve: InterpolationCurve;
}

export interface SignRepresentation {
  version: "1.0.0";
  sessionId: string;
  utteranceId: string;
  totalDurationMs: number;
  tokens: SignRepresentationToken[];
}

export interface TextToSignResponse {
  tokens: SignToken[];
  totalDurationMs: number;
  latencyMs: number;
  aslTokens?: AslGlossToken[];
  representation?: SignRepresentation;
}
