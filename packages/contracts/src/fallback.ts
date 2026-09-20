import type { SignRepresentation, SignRepresentationToken } from "./translation.js";

const TOKEN_DURATION_MS = 600;
const LEAD_IN_MS = 150;
const HOLD_MS = 300;
const LEAD_OUT_MS = 150;

const QUESTION_GLOSSES: ReadonlySet<string> = new Set(["WHAT", "HOW", "WHY", "WHERE"]);

/**
 * Build a synthetic SignRepresentation when the ML service is unreachable.
 * Single source of truth shared by the API gateway and the web client so
 * fallback timing and markers cannot diverge between server and browser.
 */
export function buildFallbackSignRepresentation(
  glosses: string[],
  sessionId: string,
): SignRepresentation {
  const tokens: SignRepresentationToken[] = glosses.map((gloss, index) => {
    const upper = gloss.toUpperCase();
    const isQuestion = QUESTION_GLOSSES.has(upper);
    return {
      tokenId: `token_${index}_${gloss.toLowerCase()}`,
      clipId: `clip_${gloss.toLowerCase()}`,
      gloss: upper,
      timing: {
        startTimeMs: index * TOKEN_DURATION_MS,
        leadInDurationMs: LEAD_IN_MS,
        holdDurationMs: HOLD_MS,
        leadOutDurationMs: LEAD_OUT_MS,
      },
      spatialLoci: {
        anchor: isQuestion ? "neutral_space" : "chest",
        targetOffset: {
          x: index % 2 === 0 ? 0.05 : -0.05,
          y: 0.1,
          z: 0.25,
        },
      },
      nonManualMarkers: {
        eyebrowIntensity: isQuestion ? 0.85 : 0.2,
        eyebrowShape: isQuestion ? "furrow" : "neutral",
        headRotation: {
          pitch: isQuestion ? 0.08 : 0,
          yaw: 0,
          roll: 0,
        },
        // Neutral mouth by default: the avatar renders per-token NMMs, and a
        // held-open mouth on every fallback token reads as wrong.
        mouthShape: "neutral",
      },
      interpolationCurve: "ease_in_out",
    };
  });

  return {
    version: "1.0.0",
    sessionId,
    utteranceId: `utt_${Date.now()}`,
    totalDurationMs: Math.max(tokens.length * TOKEN_DURATION_MS, TOKEN_DURATION_MS),
    tokens,
  };
}
