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
  handDominance?: "left" | "right" | "both";
}

