/**
 * Non-manual marker (NMM) to facial-pose mapping for the WebGL signing avatar.
 *
 * Maps the linguistic directives carried by each SignRepresentationToken
 * one-to-one onto the face rig channels consumed by webgl-avatar.tsx:
 *   - furrowed brows (wh-questions) -> eyebrowFurrow
 *   - raised brows (yes/no questions) -> eyebrowRaise
 *   - mouth morphemes (cha / oo / mm / th / open) -> mouthOpen / mouthSmile / mouthWidth
 *   - head shake / nod / tilt (already resolved to Euler angles upstream)
 *     -> headPitch / headYaw / headRoll passthrough
 *
 * Pure function with no Three.js dependency so the mapping stays unit-testable.
 */

import type { SignRepresentationToken } from "@converse/contracts";

/** Face rig channels driven per active token. Head angles are in radians. */
export interface FacePose {
  eyebrowRaise: number;
  eyebrowFurrow: number;
  mouthOpen: number;
  mouthSmile: number;
  mouthWidth: number;
  headPitch: number;
  headYaw: number;
  headRoll: number;
}

interface MouthShape {
  readonly mouthOpen: number;
  readonly mouthSmile: number;
  readonly mouthWidth: number;
}

/** Rest face: matches REST_POSE in webgl-avatar.tsx. */
const NEUTRAL_MOUTH: MouthShape = {
  mouthOpen: 0,
  mouthSmile: 0.1,
  mouthWidth: 1,
};

/**
 * Mouth morpheme table. Keys are lowercase mouthShape directives; values are
 * rig-tuned blendshape weights for the box-geometry mouth mesh in
 * webgl-avatar.tsx (scale.y = 1 + mouthOpen * 2.5, scale.x = mouthWidth).
 */
const MOUTH_SHAPES: Readonly<Record<string, MouthShape>> = {
  // Large open mouth for size adjectives (BIG, LARGE, HUGE).
  cha: { mouthOpen: 0.6, mouthSmile: 0, mouthWidth: 1.3 },
  // Rounded narrow lips for small adjectives (SMALL, TINY, THIN).
  oo: { mouthOpen: 0.35, mouthSmile: 0, mouthWidth: 0.55 },
  // Pressed closed lips for regular adjectives (MEDIUM, NORMAL, EASY).
  mm: { mouthOpen: 0, mouthSmile: 0.05, mouthWidth: 0.8 },
  // Tongue-out morpheme (careless / error marking).
  th: { mouthOpen: 0.25, mouthSmile: 0, mouthWidth: 1.1 },
  // Generic open mouth used by the realtime fallback emitter.
  open: { mouthOpen: 0.35, mouthSmile: 0.1, mouthWidth: 1 },
  neutral: { mouthOpen: 0, mouthSmile: 0.1, mouthWidth: 1 },
};

/** Maximum absolute head angle in radians accepted from token directives. */
const MAX_HEAD_ANGLE = 0.6;

function clamp01(value: number): number {
  if (!Number.isFinite(value)) {
    return 0;
  }
  return Math.min(1, Math.max(0, value));
}

function clampHeadAngle(value: number): number {
  if (!Number.isFinite(value)) {
    return 0;
  }
  return Math.min(MAX_HEAD_ANGLE, Math.max(-MAX_HEAD_ANGLE, value));
}

/**
 * Derive the avatar face pose for the given sign token from its linguistic
 * non-manual markers. Unknown or missing markers degrade to the neutral face.
 */
export function facePoseForToken(token: SignRepresentationToken): FacePose {
  const markers = token.nonManualMarkers;
  const intensity = clamp01(markers?.eyebrowIntensity ?? 0);
  const shape = markers?.eyebrowShape ?? "neutral";

  const mouthKey = (markers?.mouthShape ?? "neutral").trim().toLowerCase();
  const mouth: MouthShape = MOUTH_SHAPES[mouthKey] ?? NEUTRAL_MOUTH;

  const rotation = markers?.headRotation;

  return {
    eyebrowRaise: shape === "raise" ? intensity : 0,
    eyebrowFurrow: shape === "furrow" ? intensity : 0,
    mouthOpen: mouth.mouthOpen,
    mouthSmile: mouth.mouthSmile,
    mouthWidth: mouth.mouthWidth,
    headPitch: clampHeadAngle(rotation?.pitch ?? 0),
    headYaw: clampHeadAngle(rotation?.yaw ?? 0),
    headRoll: clampHeadAngle(rotation?.roll ?? 0),
  };
}
