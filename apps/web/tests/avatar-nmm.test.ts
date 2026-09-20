import { describe, it } from "node:test";
import assert from "node:assert/strict";
import { facePoseForToken } from "../components/avatar/nmm.ts";
import { blendPoses } from "../components/avatar/coarticulation.ts";

function makeToken(
  overrides: Record<string, unknown> = {},
): Parameters<typeof facePoseForToken>[0] {
  return {
    tokenId: "tok_0_test",
    clipId: "asl_test_01",
    gloss: "TEST",
    timing: {
      startTimeMs: 0,
      leadInDurationMs: 100,
      holdDurationMs: 250,
      leadOutDurationMs: 100,
    },
    spatialLoci: {
      anchor: "neutral_space",
      targetOffset: { x: 0, y: 0, z: 0 },
    },
    nonManualMarkers: {
      eyebrowIntensity: 0,
      eyebrowShape: "neutral",
      headRotation: { pitch: 0, yaw: 0, roll: 0 },
      mouthShape: "neutral",
    },
    interpolationCurve: "ease_in_out",
    ...overrides,
  } as Parameters<typeof facePoseForToken>[0];
}

function restPose(): Parameters<typeof blendPoses>[0] {
  const arm = {
    shoulderRotation: [0.35, 0, 0.22],
    elbowRotation: [0.45, 0, 0],
    wristRotation: [0.1, 0, 0],
    fingerCurl: [0.25, 0.25, 0.25, 0.25, 0.25],
  };
  return {
    headRotation: [0, 0, 0],
    eyebrowRaise: 0,
    eyebrowFurrow: 0,
    mouthOpen: 0,
    mouthSmile: 0.1,
    mouthWidth: 1,
    chestPitch: 0,
    leftArm: { ...arm },
    rightArm: { ...arm },
  } as Parameters<typeof blendPoses>[0];
}

describe("facePoseForToken", () => {
  it("maps furrowed brows for wh-questions", () => {
    const face = facePoseForToken(
      makeToken({
        nonManualMarkers: {
          eyebrowIntensity: 0.85,
          eyebrowShape: "furrow",
          headRotation: { pitch: 0.15, yaw: 0, roll: 0 },
          mouthShape: "neutral",
        },
      }),
    );
    assert.equal(face.eyebrowFurrow, 0.85);
    assert.equal(face.eyebrowRaise, 0);
    assert.equal(face.headPitch, 0.15);
  });

  it("maps raised brows for yes/no questions", () => {
    const face = facePoseForToken(
      makeToken({
        nonManualMarkers: {
          eyebrowIntensity: 0.8,
          eyebrowShape: "raise",
          headRotation: { pitch: 0.15, yaw: 0, roll: 0 },
          mouthShape: "neutral",
        },
      }),
    );
    assert.equal(face.eyebrowRaise, 0.8);
    assert.equal(face.eyebrowFurrow, 0);
  });

  it("maps mouth morphemes cha, oo, and mm", () => {
    const cha = facePoseForToken(
      makeToken({
        nonManualMarkers: {
          eyebrowIntensity: 0,
          eyebrowShape: "neutral",
          headRotation: { pitch: 0, yaw: 0, roll: 0 },
          mouthShape: "cha",
        },
      }),
    );
    assert.equal(cha.mouthOpen, 0.6);
    assert.ok(cha.mouthWidth > 1);

    const oo = facePoseForToken(
      makeToken({
        nonManualMarkers: {
          eyebrowIntensity: 0,
          eyebrowShape: "neutral",
          headRotation: { pitch: 0, yaw: 0, roll: 0 },
          mouthShape: "oo",
        },
      }),
    );
    assert.ok(oo.mouthWidth < 1);
    assert.ok(oo.mouthOpen > 0);

    const mm = facePoseForToken(
      makeToken({
        nonManualMarkers: {
          eyebrowIntensity: 0,
          eyebrowShape: "neutral",
          headRotation: { pitch: 0, yaw: 0, roll: 0 },
          mouthShape: "mm",
        },
      }),
    );
    assert.equal(mm.mouthOpen, 0);
  });

  it("passes head shake yaw through and clamps out-of-range values", () => {
    const shake = facePoseForToken(
      makeToken({
        nonManualMarkers: {
          eyebrowIntensity: 0,
          eyebrowShape: "neutral",
          headRotation: { pitch: 0, yaw: 0.25, roll: 0 },
          mouthShape: "neutral",
        },
      }),
    );
    assert.equal(shake.headYaw, 0.25);

    const extreme = facePoseForToken(
      makeToken({
        nonManualMarkers: {
          eyebrowIntensity: 2.5,
          eyebrowShape: "furrow",
          headRotation: { pitch: 5, yaw: -5, roll: 0 },
          mouthShape: "unknown-shape",
        },
      }),
    );
    assert.equal(extreme.eyebrowFurrow, 1);
    assert.equal(extreme.headPitch, 0.6);
    assert.equal(extreme.headYaw, -0.6);
    assert.equal(extreme.mouthOpen, 0);
  });
});

describe("blendPoses", () => {
  it("returns endpoints at t=0 and t=1", () => {
    const prev = restPose();
    const next: typeof prev = {
      ...restPose(),
      eyebrowRaise: 0.8,
      mouthOpen: 0.6,
      headRotation: [0.15, 0.25, 0],
    };
    assert.deepEqual(blendPoses(prev, next, 0), prev);
    assert.deepEqual(blendPoses(prev, next, 1), next);
  });

  it("applies smoothstep easing at the midpoint of the lead-in window", () => {
    const prev = restPose();
    const next: typeof prev = { ...restPose(), eyebrowRaise: 1 };
    const mid = blendPoses(prev, next, 0.5);
    assert.equal(mid.eyebrowRaise, 0.5);

    const quarter = blendPoses(prev, next, 0.25);
    assert.ok(Math.abs(quarter.eyebrowRaise - 0.15625) < 1e-9);
  });

  it("clamps out-of-range progress to the window ends", () => {
    const prev = restPose();
    const next: typeof prev = { ...restPose(), mouthWidth: 0.55 };
    assert.equal(blendPoses(prev, next, -2).mouthWidth, 1);
    assert.equal(blendPoses(prev, next, 42).mouthWidth, 0.55);
  });
});
