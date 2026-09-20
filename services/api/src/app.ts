import cors from "cors";
import express, { type Express, type Request, type Response } from "express";
import type {
  AsrRequest,
  AsrResponse,
  HealthCheckResponse,
  SignToTextRequest,
  SignToTextResponse,
  TextToSignRequest,
  TextToSignResponse,
  TtsRequest,
  TtsResponse,
} from "@converse/contracts";
import { reconstructSentence } from "@converse/contracts";
import {
  synthesizeSpeech,
  transcribeSpeech,
  translateSpeechToSign,
} from "./speech-to-sign.js";

export function createApp(): Express {
  const app: Express = express();

  app.use(cors());
  app.use(express.json());

  const startTime = Date.now();

  app.get("/health", (_req: Request, res: Response) => {
    const response: HealthCheckResponse = {
      status: "ok",
      service: "@converse/api",
      version: "0.1.0",
      uptimeSeconds: Math.floor((Date.now() - startTime) / 1000),
      timestamp: new Date().toISOString(),
    };
    res.json(response);
  });

  app.get("/api/info", (_req: Request, res: Response) => {
    res.json({
      name: "Converse API",
      system: "ASL <-> English Communication Engine",
      pipelines: {
        signToSpeech:
          "Camera -> ASL Vision -> ASL Representation -> Translation -> English Text -> TTS -> Audio",
        speechToSign:
          "Audio -> ASR -> English Text -> Translation -> ASL Representation -> Sign Renderer",
      },
      endpoints: [
        { path: "/health", method: "GET", description: "Service health probe" },
        {
          path: "/api/info",
          method: "GET",
          description: "Architecture and pipeline capabilities",
        },
        {
          path: "/api/sign-to-speech/translate",
          method: "POST",
          description: "Translate ASL sign detections to English text",
        },
        {
          path: "/api/speech-to-sign/translate",
          method: "POST",
          description: "Translate English text into ASL tokens",
        },
        {
          path: "/api/speech/asr",
          method: "POST",
          description: "Transcribe speech audio to text using streaming ASR",
        },
        {
          path: "/api/speech/tts",
          method: "POST",
          description: "Synthesize speech audio from text using neural TTS",
        },
      ],
    });
  });

  app.post(
    "/api/sign-to-speech/translate",
    (req: Request<unknown, unknown, SignToTextRequest>, res: Response) => {
      const { detections = [] } = req.body;
      const glosses = detections.map((d) => d.gloss);
      const avgConfidence =
        detections.length > 0
          ? detections.reduce((acc, curr) => acc + curr.confidence, 0) /
            detections.length
          : 0;
      const reconstructed = reconstructSentence(glosses, {
        confidence: avgConfidence,
      });
      res.json(reconstructed);
    },
  );

  app.post(
    "/api/speech-to-sign/translate",
    async (
      req: Request<unknown, unknown, TextToSignRequest>,
      res: Response,
    ) => {
      const result = await translateSpeechToSign(req.body);
      if (result.ok) {
        res.json(result.value);
      } else {
        res.status(400).json({ ok: false, error: result.error });
      }
    },
  );

  app.post(
    "/api/speech/asr",
    async (req: Request<unknown, unknown, AsrRequest>, res: Response) => {
      const result = await transcribeSpeech(req.body);
      if (result.ok) {
        res.json(result.value);
      } else {
        res.status(400).json({ ok: false, error: result.error });
      }
    },
  );

  app.post(
    "/api/speech/tts",
    async (req: Request<unknown, unknown, TtsRequest>, res: Response) => {
      const result = await synthesizeSpeech(req.body);
      if (result.ok) {
        res.json(result.value);
      } else {
        const statusCode =
          result.error.code === "EMPTY_TEXT_PAYLOAD" ? 400 : 503;
        res.status(statusCode).json({ ok: false, error: result.error });
      }
    },
  );

  return app;
}

export const app: Express = createApp();
