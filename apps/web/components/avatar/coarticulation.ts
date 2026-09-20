/**
 * Coarticulation blending for the WebGL signing avatar.
 *
 * Interpolates between the pose at the end of sign N and the target pose of
 * sign N+1 with smoothstep easing, so the avatar flows through token
 * boundaries during the lead-in / lead-out windows instead of snapping.
 * The easing mirrors the ease_in_out curve previously applied at call sites.
 */

import type { AvatarPose } from "./webgl-avatar";

type ArmPose = AvatarPose["leftArm"];

/** Smoothstep easing over the normalized window t in [0, 1]. */
function smoothstep(t: number): number {
  const clamped = Number.isFinite(t) ? Math.min(1, Math.max(0, t)) : 1;
  return clamped * clamped * (3 - 2 * clamped);
}

function lerp(a: number, b: number, s: number): number {
  return a + (b - a) * s;
}

function blendArmPose(from: ArmPose, to: ArmPose, s: number): ArmPose {
  return {
    shoulderRotation: [
      lerp(from.shoulderRotation[0], to.shoulderRotation[0], s),
      lerp(from.shoulderRotation[1], to.shoulderRotation[1], s),
      lerp(from.shoulderRotation[2], to.shoulderRotation[2], s),
    ],
    elbowRotation: [
      lerp(from.elbowRotation[0], to.elbowRotation[0], s),
      lerp(from.elbowRotation[1], to.elbowRotation[1], s),
      lerp(from.elbowRotation[2], to.elbowRotation[2], s),
    ],
    wristRotation: [
      lerp(from.wristRotation[0], to.wristRotation[0], s),
      lerp(from.wristRotation[1], to.wristRotation[1], s),
      lerp(from.wristRotation[2], to.wristRotation[2], s),
    ],
    fingerCurl: [
      lerp(from.fingerCurl[0], to.fingerCurl[0], s),
      lerp(from.fingerCurl[1], to.fingerCurl[1], s),
      lerp(from.fingerCurl[2], to.fingerCurl[2], s),
      lerp(from.fingerCurl[3], to.fingerCurl[3], s),
      lerp(from.fingerCurl[4], to.fingerCurl[4], s),
    ],
  };
}

/**
 * Blend two full avatar poses. t is linear progress across the transition
 * window (0 = fully prev, 1 = fully next); smoothstep easing is applied
 * internally. Out-of-range or non-finite t is clamped to the window ends.
 */
export function blendPoses(
  prev: AvatarPose,
  next: AvatarPose,
  t: number,
): AvatarPose {
  const s = smoothstep(t);
  return {
    headRotation: [
      lerp(prev.headRotation[0], next.headRotation[0], s),
      lerp(prev.headRotation[1], next.headRotation[1], s),
      lerp(prev.headRotation[2], next.headRotation[2], s),
    ],
    eyebrowRaise: lerp(prev.eyebrowRaise, next.eyebrowRaise, s),
    eyebrowFurrow: lerp(prev.eyebrowFurrow, next.eyebrowFurrow, s),
    mouthOpen: lerp(prev.mouthOpen, next.mouthOpen, s),
    mouthSmile: lerp(prev.mouthSmile, next.mouthSmile, s),
    mouthWidth: lerp(prev.mouthWidth, next.mouthWidth, s),
    chestPitch: lerp(prev.chestPitch, next.chestPitch, s),
    leftArm: blendArmPose(prev.leftArm, next.leftArm, s),
    rightArm: blendArmPose(prev.rightArm, next.rightArm, s),
  };
}
