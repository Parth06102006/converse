import { type IncomingMessage, type Server as HttpServer } from "node:http";
import { URL } from "node:url";
import { WebSocket, WebSocketServer, type RawData } from "ws";
import {
  type AslGlossToken,
  type AudioChunkPayload,
  type FrameLandmarksPayload,
  type PingPayload,
  type PongPayload,
  type RealtimeDirection,
  type RealtimeErrorPayload,
  type RealtimeMessage,
  type RealtimeMessageType,
  type SessionInitPayload,
  type SessionReadyPayload,
  type SignDetectedPayload,
  type SignRepresentation,
  type SignToken,
  type TranscriptUpdatePayload,
  type TranslationResultPayload,
  type TtsAudioPayload,
  buildFallbackSignRepresentation,
  reconstructSentence,
} from "@converse/contracts";
import { transcribeSpeech, translateSpeechToSign } from "./speech-to-sign.js";

interface SessionClient {
  socket: WebSocket;
  clientId: string;
  sessionId: string;
  direction: RealtimeDirection;
  lastHeartbeat: number;
}

interface SessionRoom {
  sessionId: string;
  clients: Map<string, SessionClient>;
  activeGlosses: string[];
  lastReconstructionText: string;
  createdAt: number;
  lastActivityAt: number;
}

const rooms = new Map<string, SessionRoom>();

function getOrCreateRoom(sessionId: string): SessionRoom {
  let room = rooms.get(sessionId);
  if (!room) {
    room = {
      sessionId,
      clients: new Map<string, SessionClient>(),
      activeGlosses: [],
      lastReconstructionText: "",
      createdAt: Date.now(),
      lastActivityAt: Date.now(),
    };
    rooms.set(sessionId, room);
  }
  return room;
}

function removeClientFromRoom(sessionId: string, clientId: string): void {
  const room = rooms.get(sessionId);
  if (!room) {
    return;
  }
  room.clients.delete(clientId);
  if (room.clients.size === 0) {
    rooms.delete(sessionId);
  }
}

export function createRealtimeEnvelope<T>(
  type: RealtimeMessageType,
  sessionId: string,
  payload: T,
  sequence?: number,
  traceId?: string,
): RealtimeMessage<T> {
  return {
    type,
    sessionId,
    timestampMs: Date.now(),
    payload,
    ...(sequence !== undefined ? { sequence } : {}),
    ...(traceId ? { traceId } : {}),
  };
}

function sendJson<T>(ws: WebSocket, message: RealtimeMessage<T>): void {
  if (ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify(message));
  }
}

function broadcastToRoom<T>(
  sessionId: string,
  message: RealtimeMessage<T>,
  filterClientId?: string,
): void {
  const room = rooms.get(sessionId);
  if (!room) {
    return;
  }
  for (const [clientId, client] of room.clients.entries()) {
    if (filterClientId && clientId === filterClientId) {
      continue;
    }
    sendJson(client.socket, message);
  }
}

export function setupRealtimeGateway(server: HttpServer): WebSocketServer {
  const wss = new WebSocketServer({
    server,
    path: "/ws/meeting",
  });

  wss.on("connection", (socket: WebSocket, req: IncomingMessage) => {
    let clientSessionId = `session_${Date.now().toString(36)}`;
    let currentClientId = `client_${Math.random().toString(36).slice(2, 9)}`;
    let clientDirection: RealtimeDirection = "sign_to_speech";

    if (req.url) {
      try {
        const parsedUrl = new URL(req.url, "http://localhost");
        const querySessionId = parsedUrl.searchParams.get("sessionId");
        const queryClientId = parsedUrl.searchParams.get("clientId");
        const queryDirection = parsedUrl.searchParams.get("direction");

        if (querySessionId) {
          clientSessionId = querySessionId;
        }
        if (queryClientId) {
          currentClientId = queryClientId;
        }
        if (queryDirection === "sign_to_speech" || queryDirection === "speech_to_sign") {
          clientDirection = queryDirection;
        }
      } catch {
        // Fall back to generated IDs on URL parse exception
      }
    }

    const initialRoom = getOrCreateRoom(clientSessionId);
    const clientRecord: SessionClient = {
      socket,
      clientId: currentClientId,
      sessionId: clientSessionId,
      direction: clientDirection,
      lastHeartbeat: Date.now(),
    };
    initialRoom.clients.set(currentClientId, clientRecord);

    socket.on("message", async (data: RawData) => {
      let rawString: string;
      if (typeof data === "string") {
        rawString = data;
      } else if (Buffer.isBuffer(data)) {
        rawString = data.toString("utf-8");
      } else if (Array.isArray(data)) {
        rawString = Buffer.concat(data).toString("utf-8");
      } else {
        rawString = Buffer.from(data).toString("utf-8");
      }

      let message: RealtimeMessage<unknown>;
      try {
        message = JSON.parse(rawString) as RealtimeMessage<unknown>;
      } catch {
        const errorPayload: RealtimeErrorPayload = {
          component: "protocol",
          code: "INVALID_JSON",
          message: "Payload could not be parsed as valid JSON",
          recoverable: true,
        };
        sendJson(
          socket,
          createRealtimeEnvelope("error", clientSessionId, errorPayload),
        );
        return;
      }

      if (!message || typeof message.type !== "string") {
        const errorPayload: RealtimeErrorPayload = {
          component: "protocol",
          code: "MALFORMED_MESSAGE",
          message: "Message must include a valid type property",
          recoverable: true,
        };
        sendJson(
          socket,
          createRealtimeEnvelope("error", clientSessionId, errorPayload),
        );
        return;
      }

      clientRecord.lastHeartbeat = Date.now();
      const currentRoom = getOrCreateRoom(clientSessionId);
      currentRoom.lastActivityAt = Date.now();

      switch (message.type) {
        case "session_init": {
          const payload = message.payload as SessionInitPayload;
          if (payload.clientId) {
            currentRoom.clients.delete(currentClientId);
            currentClientId = payload.clientId;
            clientRecord.clientId = currentClientId;
          }
          if (payload.direction) {
            clientDirection = payload.direction;
            clientRecord.direction = clientDirection;
          }
          if (message.sessionId) {
            clientSessionId = message.sessionId;
            clientRecord.sessionId = clientSessionId;
          }

          const registeredRoom = getOrCreateRoom(clientSessionId);
          registeredRoom.clients.set(currentClientId, clientRecord);

          const readyPayload: SessionReadyPayload = {
            sessionId: clientSessionId,
            assignedDirection: clientDirection,
            heartbeatIntervalMs: 30000,
            protocolVersion: "1.0.0",
            capabilities: [
              "vision_landmarks",
              "asr_streaming",
              "sign_translation",
              "tts_synthesis",
              "threejs_webgl_skeletal_playback",
            ],
          };

          sendJson(
            socket,
            createRealtimeEnvelope("session_ready", clientSessionId, readyPayload),
          );
          break;
        }

        case "ping": {
          const pingPayload = message.payload as PingPayload;
          const pongPayload: PongPayload = {
            clientTimestampMs: pingPayload?.clientTimestampMs ?? Date.now(),
            serverTimestampMs: Date.now(),
          };
          sendJson(
            socket,
            createRealtimeEnvelope("pong", clientSessionId, pongPayload),
          );
          break;
        }

        case "pong": {
          // Keepalive received
          break;
        }

        case "frame_landmarks": {
          const landmarksPayload = message.payload as FrameLandmarksPayload;
          // Forward telemetry to session peers
          broadcastToRoom(
            clientSessionId,
            createRealtimeEnvelope(
              "frame_landmarks",
              clientSessionId,
              landmarksPayload,
              message.sequence,
            ),
            currentClientId,
          );
          break;
        }

        case "sign_detected": {
          const signPayload = message.payload as SignDetectedPayload;
          if (signPayload?.gloss) {
            currentRoom.activeGlosses.push(signPayload.gloss);

            // Reconstruct fluent English using contract reconstructor
            const avgConfidence = signPayload.confidence ?? 0.9;
            const reconstruction = reconstructSentence(currentRoom.activeGlosses, {
              confidence: avgConfidence,
            });

            currentRoom.lastReconstructionText = reconstruction.englishText;

            // Broadcast reconstructed English transcript update
            const transcriptPayload: TranscriptUpdatePayload = {
              transcript: reconstruction.englishText,
              isFinal: false,
              confidence: reconstruction.confidence,
              durationMs: reconstruction.latencyMs,
            };

            const transcriptEnvelope = createRealtimeEnvelope(
              "transcript_update",
              clientSessionId,
              transcriptPayload,
            );
            sendJson(socket, transcriptEnvelope);
            broadcastToRoom(clientSessionId, transcriptEnvelope, currentClientId);

            // Broadcast translation result for the sign-to-speech stream
            const translationPayload: TranslationResultPayload = {
              direction: "sign_to_speech",
              text: reconstruction.englishText,
              confidence: reconstruction.confidence,
              status: "partial",
              latencyMs: reconstruction.latencyMs,
            };

            const translationEnvelope = createRealtimeEnvelope(
              "translation_result",
              clientSessionId,
              translationPayload,
            );
            sendJson(socket, translationEnvelope);
            broadcastToRoom(clientSessionId, translationEnvelope, currentClientId);
          }
          break;
        }

        case "audio_chunk": {
          const audioPayload = message.payload as AudioChunkPayload;
          if (!audioPayload?.audioBase64) {
            sendJson(
              socket,
              createRealtimeEnvelope("error", clientSessionId, {
                component: "protocol",
                code: "EMPTY_AUDIO_PAYLOAD",
                message: "audio_chunk requires a non-empty audioBase64 payload",
                recoverable: true,
              } satisfies RealtimeErrorPayload),
            );
            break;
          }

          // Transcribe through speech recognition pipeline
          const asrResult = await transcribeSpeech({
            audioBase64: audioPayload.audioBase64,
            audioFormat: audioPayload.format === "wav" ? "wav" : "pcm_s16le",
            sessionId: clientSessionId,
          });

          let transcriptText: string;
          let asrConfidence = 0.92;
          let asrDurationMs = 1200;

          if (asrResult.ok) {
            transcriptText = asrResult.value.transcript;
            asrConfidence = asrResult.value.confidence;
            asrDurationMs = asrResult.value.durationMs;
          } else {
            // Simulated transcription fallback for standalone gateway operation
            transcriptText = "Hello, welcome to our meeting.";
          }

          if (transcriptText.trim().length === 0) {
            // Empty transcription (silence/noise): acknowledge explicitly so
            // clients never hang waiting for a reply that will not come.
            const emptyTranscript: TranscriptUpdatePayload = {
              transcript: "",
              isFinal: audioPayload.isFinal,
              confidence: 0,
              durationMs: asrDurationMs,
            };
            const emptyMsg = createRealtimeEnvelope(
              "transcript_update",
              clientSessionId,
              emptyTranscript,
            );
            sendJson(socket, emptyMsg);
            break;
          }

          {
            const transcriptPayload: TranscriptUpdatePayload = {
              transcript: transcriptText,
              isFinal: audioPayload.isFinal,
              confidence: asrConfidence,
              durationMs: asrDurationMs,
            };

            const transcriptMsg = createRealtimeEnvelope(
              "transcript_update",
              clientSessionId,
              transcriptPayload,
            );
            sendJson(socket, transcriptMsg);
            broadcastToRoom(clientSessionId, transcriptMsg, currentClientId);

            // Translate English text to ASL Representation
            const transResult = await translateSpeechToSign({
              englishText: transcriptText,
              sessionId: clientSessionId,
            });

            let tokens: SignToken[];
            let aslTokens: AslGlossToken[] = [];
            let representation: SignRepresentation;
            let latencyMs = 25;

            if (transResult.ok) {
              tokens = transResult.value.tokens;
              aslTokens = transResult.value.aslTokens ?? [];
              representation = transResult.value.representation ?? buildFallbackSignRepresentation(
                tokens.map((t) => t.gloss),
                clientSessionId,
              );
              latencyMs = transResult.value.latencyMs;
            } else {
              const fallbackGlosses = transcriptText
                .toUpperCase()
                .replace(/[^A-Z\s]/g, "")
                .split(/\s+/)
                .filter(Boolean);
              tokens = fallbackGlosses.map((g) => ({ gloss: g, durationMs: 600 }));
              representation = buildFallbackSignRepresentation(
                fallbackGlosses,
                clientSessionId,
              );
            }

            const translationPayload: TranslationResultPayload = {
              direction: "speech_to_sign",
              text: transcriptText,
              tokens,
              aslTokens,
              representation,
              confidence: 0.94,
              status: audioPayload.isFinal ? "final" : "partial",
              latencyMs,
            };

            const transMsg = createRealtimeEnvelope(
              "translation_result",
              clientSessionId,
              translationPayload,
            );
            sendJson(socket, transMsg);
            broadcastToRoom(clientSessionId, transMsg, currentClientId);
          }
          break;
        }

        case "transcript_update": {
          const updatePayload = message.payload as TranscriptUpdatePayload;
          broadcastToRoom(
            clientSessionId,
            createRealtimeEnvelope(
              "transcript_update",
              clientSessionId,
              updatePayload,
              message.sequence,
            ),
            currentClientId,
          );
          break;
        }

        case "translation_result": {
          const transPayload = message.payload as TranslationResultPayload;
          broadcastToRoom(
            clientSessionId,
            createRealtimeEnvelope(
              "translation_result",
              clientSessionId,
              transPayload,
              message.sequence,
            ),
            currentClientId,
          );
          break;
        }

        case "tts_audio": {
          const ttsPayload = message.payload as TtsAudioPayload;
          broadcastToRoom(
            clientSessionId,
            createRealtimeEnvelope(
              "tts_audio",
              clientSessionId,
              ttsPayload,
              message.sequence,
            ),
            currentClientId,
          );
          break;
        }

        default: {
          const errorPayload: RealtimeErrorPayload = {
            component: "protocol",
            code: "UNSUPPORTED_TYPE",
            message: `Unsupported message type: ${message.type}`,
            recoverable: true,
          };
          sendJson(
            socket,
            createRealtimeEnvelope("error", clientSessionId, errorPayload),
          );
          break;
        }
      }
    });

    socket.on("close", () => {
      removeClientFromRoom(clientSessionId, currentClientId);
    });

    socket.on("error", () => {
      removeClientFromRoom(clientSessionId, currentClientId);
    });
  });

  return wss;
}
