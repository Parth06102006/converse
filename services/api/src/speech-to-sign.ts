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

export async function translateSpeechToSign(
  request: TextToSignRequest,
): Promise<Result<TextToSignResponse, DomainError>> {
  const startTime = Date.now();
  const englishText = request.englishText?.trim() ?? "";
  const sessionId = request.sessionId ?? "default_session";

  if (!englishText) {
    const emptyResponse: TextToSignResponse = {
      tokens: [],
      aslTokens: [],
      totalDurationMs: 0,
      latencyMs: Date.now() - startTime,
    };
    return ok(emptyResponse);
  }

  try {
    const response = await fetch(
      `${ML_SERVICE_URL}/internal/speech-to-sign/compile`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          englishText,
          sessionId,
        }),
        signal: AbortSignal.timeout(5000),
      },
    );

    if (!response.ok) {
      return err({
        component: "translation",
        code: "HTTP_ERROR",
        message: `Translation service HTTP error: ${response.status} ${response.statusText}`,
        recoverable: true,
      });
    }

    const body = (await response.json()) as MlServiceCompileResponse;
    if (!body.ok || !body.data) {
      return err({
        component: "translation",
        code: "COMPILATION_FAILED",
        message: body.error ?? "Failed to compile text to ASL representation",
        recoverable: false,
      });
    }

    const representation = body.data;
    const tokens: SignToken[] = representation.tokens.map((t) => ({
      gloss: t.gloss,
      durationMs:
        t.timing.leadInDurationMs +
        t.timing.holdDurationMs +
        t.timing.leadOutDurationMs,
    }));

    const eyebrowMap: Record<
      "furrow" | "raise" | "neutral",
      "furrowed" | "raised" | "neutral"
    > = {
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
  } catch (error) {
    return err({
      component: "translation",
      code: "SERVICE_UNAVAILABLE",
      message:
        error instanceof Error
          ? `Translation service unavailable: ${error.message}`
          : "Translation service unavailable",
      recoverable: true,
    });
  }
}

export async function transcribeSpeech(
  request: AsrRequest,
): Promise<Result<AsrResponse, DomainError>> {
  const audioBase64 = request.audioBase64 ?? "";
  const audioFormat = request.audioFormat ?? "pcm_s16le";
  const sessionId = request.sessionId ?? "default_session";
  const backend = request.backend;

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
        ...(backend ? { backend } : {}),
      }),
      signal: AbortSignal.timeout(10000),
    });

    if (!response.ok) {
      const errorBody = (await response.json().catch(() => ({}))) as {
        ok?: boolean;
        error?: string;
        code?: string;
      };
      return err({
        component: "asr",
        code: errorBody.code ?? "HTTP_ERROR",
        message:
          errorBody.error ??
          `ASR service HTTP error: ${response.status} ${response.statusText}`,
        recoverable: response.status >= 500,
      });
    }

    const body = (await response.json()) as MlServiceAsrResponse;
    if (!body.ok || !body.data) {
      return err({
        component: "asr",
        code: "TRANSCRIPTION_FAILED",
        message: body.error ?? "ASR transcription failed",
        recoverable: false,
      });
    }

    return ok({
      transcript: body.data.transcript,
      isFinal: body.data.isFinal,
      confidence: body.data.confidence,
      durationMs: body.data.durationMs,
    });
  } catch (error) {
    return err({
      component: "asr",
      code: "SERVICE_UNAVAILABLE",
      message:
        error instanceof Error
          ? `ASR service unavailable: ${error.message}`
          : "ASR service unavailable",
      recoverable: true,
    });
  }
}
