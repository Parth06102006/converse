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
} from "@converse/contracts";
import {
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
      ],
    });
  });

  app.post(
    "/api/sign-to-speech/translate",
    (req: Request<unknown, unknown, SignToTextRequest>, res: Response) => {
      const { detections = [] } = req.body;
      const glosses = detections.map((d) => d.gloss);
      const englishText = glosses.length > 0 ? glosses.join(" ") : "";

      const response: SignToTextResponse = {
        englishText,
        confidence:
          detections.length > 0
            ? detections.reduce((acc, curr) => acc + curr.confidence, 0) /
              detections.length
            : 0,
        glosses,
        latencyMs: 12,
      };

      res.json(response);
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

  return app;
}

export const app: Express = createApp();
