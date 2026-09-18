import {
  type AslGlossToken,
  type AsrRequest,
  type AsrResponse,
  type DomainError,
  type Result,
  type SignRepresentation,
  type SignToken,
  type TextToSignRequest,
  type TextToSignResponse,
  err,
  ok,
} from "@converse/contracts";

const ML_SERVICE_URL = process.env.ML_SERVICE_URL ?? "http://127.0.0.1:5050";

interface MlServiceCompileResponse {
  ok: boolean;
  data?: SignRepresentation;
  error?: string;
}

interface MlServiceAsrResponse {
  ok: boolean;
  data?: {
    transcript: string;
    isFinal: boolean;
    confidence: number;
    durationMs: number;
  };
  error?: string;
}

function buildFallbackRepresentation(
  text: string,
  sessionId = "default",
): SignRepresentation {
  const words = text.trim().split(/\s+/).filter(Boolean);
  const now = Date.now();

  const tokens = words.map((word, index) => {
    const gloss = word.toUpperCase();
    const startTime = index * 420;

    return {
      tokenId: `tok_${index}_${gloss.toLowerCase()}`,
      clipId: `asl_${gloss.toLowerCase()}_01`,
      gloss,
      timing: {
        startTimeMs: startTime,
        leadInDurationMs: 120,
        holdDurationMs: 300,
        leadOutDurationMs: 100,
      },
      spatialLoci: {
        anchor: (gloss === "ME" || gloss === "I" ? "chest" : "neutral_space") as
          | "chest"
          | "neutral_space",
        targetOffset: {
          x: 0.0,
          y: 0.0,
          z: gloss === "ME" || gloss === "I" ? 0.1 : 0.35,
        },
      },
      nonManualMarkers: {
        eyebrowIntensity: text.includes("?") ? 0.85 : 0.0,
        eyebrowShape: (text.includes("?") ? "furrow" : "neutral") as
          | "furrow"
          | "neutral",
        headRotation: { pitch: 0.0, yaw: 0.0, roll: 0.0 },
        mouthShape: "neutral",
      },
      interpolationCurve: "bezier_slerp" as const,
    };
  });

  const totalDurationMs =
    tokens.length > 0
      ? tokens[tokens.length - 1]!.timing.startTimeMs + 520
      : 0;

  return {
    version: "1.0.0",
    sessionId,
    utteranceId: `utt_${now}`,
    totalDurationMs,
    tokens,
  };
}

export async function translateSpeechToSign(
  request: TextToSignRequest,
): Promise<Result<TextToSignResponse, DomainError>> {
  const startTime = Date.now();
  const englishText = request.englishText?.trim() ?? "";
  const sessionId = request.sessionId ?? "default_session";

  if (!englishText) {
    const emptyResponse: TextToSignResponse = {
      tokens: [],
      totalDurationMs: 0,
      latencyMs: Date.now() - startTime,
    };
    return ok(emptyResponse);
  }

  try {
    const response = await fetch(`${ML_SERVICE_URL}/internal/speech-to-sign/compile`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        englishText,
        sessionId,
      }),
      signal: AbortSignal.timeout(1500),
    });

    if (response.ok) {
      const body = (await response.json()) as MlServiceCompileResponse;
      if (body.ok && body.data) {
        const representation = body.data;
        const tokens: SignToken[] = representation.tokens.map((t) => ({
          gloss: t.gloss,
          durationMs:
            t.timing.leadInDurationMs +
            t.timing.holdDurationMs +
            t.timing.leadOutDurationMs,
        }));

        const eyebrowMap: Record<"furrow" | "raise" | "neutral", "furrowed" | "raised" | "neutral"> = {
          furrow: "furrowed",
          raise: "raised",
          neutral: "neutral",
        };

        const aslTokens: AslGlossToken[] = representation.tokens.map((t) => ({
          gloss: t.gloss,
          lemma: t.gloss.toLowerCase(),
          partOfSpeech: "NOUN",
          nonManualMarkers: {
            eyebrows: eyebrowMap[t.nonManualMarkers.eyebrowShape],
            headMotion: "neutral",
            mouthMorpheme:
              t.nonManualMarkers.mouthShape === "neutral"
                ? undefined
                : t.nonManualMarkers.mouthShape,
          },
          isFingerspelled: t.clipId.startsWith("asl_fs_"),
        }));

        return ok({
          tokens,
          aslTokens,
          representation,
          totalDurationMs: representation.totalDurationMs,
          latencyMs: Date.now() - startTime,
        });
      }
    }
  } catch {
    // Network / service timeout - fall back to deterministic local compilation
  }

  // Graceful local fallback
  const fallbackRepresentation = buildFallbackRepresentation(
    englishText,
    sessionId,
  );
  const fallbackTokens: SignToken[] = fallbackRepresentation.tokens.map(
    (t) => ({
      gloss: t.gloss,
      durationMs: 400,
    }),
  );

  return ok({
    tokens: fallbackTokens,
    representation: fallbackRepresentation,
    totalDurationMs: fallbackRepresentation.totalDurationMs,
    latencyMs: Date.now() - startTime,
  });
}

export async function transcribeSpeech(
  request: AsrRequest,
): Promise<Result<AsrResponse, DomainError>> {
  const startTime = Date.now();
  const audioBase64 = request.audioBase64 ?? "";
  const audioFormat = request.audioFormat ?? "pcm_s16le";
  const sessionId = request.sessionId ?? "default_session";

  if (!audioBase64) {
    return err({
      component: "asr",
      code: "EMPTY_AUDIO_PAYLOAD",
      message: "audioBase64 payload must not be empty",
      recoverable: true,
    });
  }

  try {
    const response = await fetch(`${ML_SERVICE_URL}/internal/speech/asr`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        audioBase64,
        audioFormat,
        sessionId,
      }),
      signal: AbortSignal.timeout(2000),
    });

    if (response.ok) {
      const body = (await response.json()) as MlServiceAsrResponse;
      if (body.ok && body.data) {
        return ok({
          transcript: body.data.transcript,
          isFinal: body.data.isFinal,
          confidence: body.data.confidence,
          durationMs: body.data.durationMs,
        });
      }
    }
  } catch {
    // Network / timeout
  }

  // Fallback response
  return ok({
    transcript: "",
    isFinal: true,
    confidence: 0.0,
    durationMs: Date.now() - startTime,
  });
}
