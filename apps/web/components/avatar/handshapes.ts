/**
 * ASL fingerspelling handshape table for the WebGL signing avatar.
 *
 * Maps a fingerspelled token gloss (single character A-Z, 0-9) to the
 * five-value fingerCurl tuple consumed by webgl-avatar.tsx:
 *   [thumb, index, middle, ring, pinky], where 0 = fully open, 1 = fully curled.
 *
 * Values approximate standard ASL fingerspelling descriptions within what a
 * curl-only rig can express. Properties the rig cannot express (finger spread
 * for U/V, thumb placement for M/N/S/T, wrist motion for J/Z) are noted below.
 */

export type FingerCurl = [number, number, number, number, number];

type HandshapeEntry = {
  readonly curl: FingerCurl;
  readonly description: string;
};

const HANDSHAPE_TABLE: Readonly<Record<string, HandshapeEntry>> = {
  // Fist, thumb pressed alongside the index side.
  A: { curl: [0.25, 1, 1, 1, 1], description: "Fist, thumb alongside fingers" },
  // Flat hand, four fingers extended together, thumb folded across the palm.
  B: { curl: [0.75, 0, 0, 0, 0], description: "Flat hand, thumb across palm" },
  // Open hand curved into a C shape.
  C: { curl: [0.4, 0.45, 0.45, 0.45, 0.45], description: "Curved open hand" },
  // Index extended, remaining fingertips grouped against the thumb.
  D: { curl: [0.7, 0, 0.9, 0.9, 0.9], description: "Index up, others grouped with thumb" },
  // Loose fist, fingertips folded back toward the thumb held across.
  E: { curl: [0.8, 0.85, 0.85, 0.85, 0.85], description: "Loose fist, thumb across fingertips" },
  // Thumb and index form a ring, middle/ring/pinky extended.
  F: { curl: [0.5, 0.5, 0, 0, 0], description: "Thumb-index ring, three fingers up" },
  // Thumb and index pinched sideways, remaining fingers folded.
  G: { curl: [0.2, 0.2, 1, 1, 1], description: "Sideways pinch, palm facing sideways" },
  // Index and middle extended together sideways, thumb resting between.
  H: { curl: [0.6, 0.1, 0.1, 1, 1], description: "Two fingers extended sideways" },
  // Pinky extended, thumb held across the folded fingers.
  I: { curl: [0.9, 1, 1, 1, 0], description: "Pinky up" },
  // Same handshape as I; true J adds a wrist-drawn hook motion (not implemented).
  J: { curl: [0.9, 1, 1, 1, 0], description: "Pinky up plus wrist hook (motion not modeled)" },
  // Index and middle up in a V with the thumb between them.
  K: { curl: [0.1, 0, 0, 1, 1], description: "V with thumb between index and middle" },
  // Thumb and index form an L, remaining fingers folded.
  L: { curl: [0, 0, 1, 1, 1], description: "Thumb-index L" },
  // Thumb tucked under the folded index, middle, and ring fingers.
  M: { curl: [1, 0.95, 0.95, 0.95, 1], description: "Thumb under three folded fingers" },
  // Thumb tucked under the folded index and middle fingers.
  N: { curl: [1, 0.95, 0.95, 1, 1], description: "Thumb under two folded fingers" },
  // All digits converge into a tapered ring.
  O: { curl: [0.6, 0.6, 0.6, 0.6, 0.6], description: "Tapered ring" },
  // Same finger posture as K, pointed downward via the wrist (not modeled here).
  P: { curl: [0.1, 0, 0, 1, 1], description: "K handshape pointed down (wrist not modeled)" },
  // Same finger posture as G, pointed downward via the wrist (not modeled here).
  Q: { curl: [0.2, 0.2, 1, 1, 1], description: "Sideways pinch pointed down (wrist not modeled)" },
  // Index and middle extended and crossed; curl-only rig shows them together.
  R: { curl: [0.6, 0, 0, 1, 1], description: "Index and middle crossed (cross not modeled)" },
  // Fist with the thumb folded across the front of the fingers.
  S: { curl: [0.9, 1, 1, 1, 1], description: "Fist, thumb across front" },
  // Fist with the thumb tip held between index and middle.
  T: { curl: [0.7, 1, 1, 1, 1], description: "Fist, thumb between index and middle" },
  // Index and middle extended together and touching.
  U: { curl: [0.6, 0, 0, 1, 1], description: "Index and middle together" },
  // Index and middle extended apart in a V; spread is not modeled by the rig.
  V: { curl: [0.6, 0, 0, 1, 1], description: "Index and middle in V (spread not modeled)" },
  // Index, middle, and ring extended, thumb resting across the palm side.
  W: { curl: [0.6, 0, 0, 0, 1], description: "Three fingers up" },
  // Fist with the index hooked, thumb resting alongside.
  X: { curl: [0.5, 0.6, 1, 1, 1], description: "Hooked index" },
  // Thumb and pinky extended, middle fingers folded.
  Y: { curl: [0, 1, 1, 1, 0], description: "Thumb and pinky out" },
  // Index extended to draw a Z in the air (wrist motion not implemented).
  Z: { curl: [0.6, 0, 1, 1, 1], description: "Index extended plus wrist Z (motion not modeled)" },
  // Digits use palm-forward ASL number handshapes.
  "0": { curl: [0.6, 0.6, 0.6, 0.6, 0.6], description: "Closed ring" },
  "1": { curl: [0.9, 0, 1, 1, 1], description: "Index up" },
  "2": { curl: [0.6, 0, 0, 1, 1], description: "Two fingers up" },
  "3": { curl: [0, 0, 0, 1, 1], description: "Thumb, index, and middle extended" },
  "4": { curl: [0.1, 0, 0, 0, 0], description: "Four fingers up, thumb open" },
  "5": { curl: [0, 0, 0, 0, 0], description: "Open hand" },
  // 6-9 contact the thumb against each finger in turn; approximated with partial curls.
  "6": { curl: [0.6, 0, 0, 0, 0.5], description: "Thumb near pinky" },
  "7": { curl: [0.6, 0, 0, 0.5, 0], description: "Thumb near ring finger" },
  "8": { curl: [0.6, 0, 0.5, 0, 0], description: "Thumb near middle finger" },
  "9": { curl: [0.6, 0.5, 0, 0, 0], description: "Thumb near index finger" },
};

/**
 * Return the fingerCurl tuple for a fingerspelled token gloss, or null when
 * the gloss is not a single fingerspellable character (A-Z, 0-9).
 */
export function handshapeForToken(gloss: string): FingerCurl | null {
  const key = gloss.trim().toUpperCase();
  if (key.length !== 1) {
    return null;
  }
  const entry: HandshapeEntry | undefined = HANDSHAPE_TABLE[key];
  if (entry === undefined) {
    return null;
  }
  const curl = entry.curl;
  return [curl[0], curl[1], curl[2], curl[3], curl[4]];
}
