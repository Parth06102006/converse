"use client";

import React, { useState, useEffect, useRef, useCallback } from "react";
import Link from "next/link";
import type {
  RealtimeMessage,
  RealtimeMessageType,
  SessionInitPayload,
  SessionReadyPayload,
  SignDetectedPayload,
  TranscriptUpdatePayload,
  TranslationResultPayload,
  PingPayload,
  PongPayload,
  SignRepresentation,
} from "@converse/contracts";
import { reconstructSentence } from "@converse/contracts";
import { buildFallbackSignRepresentation } from "@converse/contracts";
import { WebglAvatar } from "../../components/avatar/webgl-avatar";
import {
  IconActivity,
  IconAudioWave,
  IconHandGesture,
} from "../../components/marketing/icons";
import styles from "./meeting.module.css";

interface AslTokenDisplay {
  gloss: string;
  pos: string;
  marker: string;
}

const SAMPLE_SIGNS = [
  "HELLO",
  "WELCOME",
  "MEETING",
  "NAME",
  "HOW",
  "YOU",
  "THANK_YOU",
  "HELP",
  "GOOD",
];

const SAMPLE_HEARING_PHRASES = [
  "Welcome to our meeting today.",
  "How can I help you?",
  "Thank you for joining me.",
  "My name is Alex.",
  "Good to meet you.",
];

export default function MeetingPage() {
  const [isConnected, setIsConnected] = useState<boolean>(false);
  const [latencyMs, setLatencyMs] = useState<number>(14);
  const sessionId = "room_alpha_teleconference";
  const rawId = React.useId();
  const clientId = `client_${rawId.replace(/:/g, "_")}`;
  const socketRef = useRef<WebSocket | null>(null);
  const pingIntervalRef = useRef<NodeJS.Timeout | null>(null);

  // Left Pane: Local Deaf Signer State
  const [cameraActive, setCameraActive] = useState<boolean>(false);
  const [overlayActive, setOverlayActive] = useState<boolean>(true);
  const [activeGlosses, setActiveGlosses] = useState<string[]>(["HELLO", "WELCOME"]);
  const [reconstructedText, setReconstructedText] = useState<string>(
    "Hello, welcome.",
  );
  const [reconstructionConfidence, setReconstructionConfidence] = useState<number>(0.96);
  const [autoSpeak, setAutoSpeak] = useState<boolean>(true);
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const mediaStreamRef = useRef<MediaStream | null>(null);

  // Right Pane: Hearing Participant State
  const [speechInput, setSpeechInput] = useState<string>("");
  const [speechTranscript, setSpeechTranscript] = useState<string>(
    "Welcome to our meeting today.",
  );
  const [aslTokens, setAslTokens] = useState<AslTokenDisplay[]>([
    { gloss: "WELCOME", pos: "VERB", marker: "EYEBROWS: RAISE" },
    { gloss: "MEETING", pos: "NOUN", marker: "LOCUS: CHEST" },
  ]);
  const [avatarRepresentation, setAvatarRepresentation] = useState<SignRepresentation | null>(null);
  const [playbackSpeed, setPlaybackSpeed] = useState<number>(1.0);

  // Speech Synthesis helper
  const speakText = useCallback((textToSpeak: string) => {
    if (typeof window === "undefined" || !("speechSynthesis" in window)) {
      return;
    }
    window.speechSynthesis.cancel();
    const utterance = new SpeechSynthesisUtterance(textToSpeak);
    utterance.rate = 1.0;
    utterance.pitch = 1.0;
    window.speechSynthesis.speak(utterance);
  }, []);

  // Safe WebSocket message sender
  const sendRealtimeMessage = useCallback(
    <T,>(type: RealtimeMessageType, payload: T) => {
      if (socketRef.current && socketRef.current.readyState === WebSocket.OPEN) {
        const message: RealtimeMessage<T> = {
          type,
          sessionId,
          timestampMs: Date.now(),
          payload,
        };
        socketRef.current.send(JSON.stringify(message));
      }
    },
    [sessionId],
  );

  // Connect to Real-time Gateway WebSocket
  useEffect(() => {
    const wsHost = process.env.NEXT_PUBLIC_WS_HOST ?? "localhost:4000";
    const wsUrl = `ws://${wsHost}/ws/meeting?sessionId=${sessionId}&clientId=${clientId}&direction=sign_to_speech`;

    let socket: WebSocket;
    try {
      socket = new WebSocket(wsUrl);
      socketRef.current = socket;
    } catch {
      return;
    }

    socket.onopen = () => {
      setIsConnected(true);
      const initPayload: SessionInitPayload = {
        direction: "sign_to_speech",
        clientId,
        sampleRate: 16000,
        videoFps: 30,
      };
      sendRealtimeMessage("session_init", initPayload);

      // Start ping heartbeat
      pingIntervalRef.current = setInterval(() => {
        const pingPayload: PingPayload = { clientTimestampMs: Date.now() };
        sendRealtimeMessage("ping", pingPayload);
      }, 15000);
    };

    socket.onmessage = (event: MessageEvent) => {
      try {
        const msg = JSON.parse(event.data as string) as RealtimeMessage<unknown>;
        if (!msg || !msg.type) {
          return;
        }

        switch (msg.type) {
          case "session_ready": {
            const ready = msg.payload as SessionReadyPayload;
            if (ready) {
              setIsConnected(true);
            }
            break;
          }

          case "pong": {
            const pong = msg.payload as PongPayload;
            if (pong?.clientTimestampMs) {
              setLatencyMs(Math.max(1, Date.now() - pong.clientTimestampMs));
            }
            break;
          }

          case "transcript_update": {
            const update = msg.payload as TranscriptUpdatePayload;
            if (update?.transcript) {
              setReconstructedText(update.transcript);
              setReconstructionConfidence(update.confidence);
              if (autoSpeak && update.transcript) {
                speakText(update.transcript);
              }
            }
            break;
          }

          case "translation_result": {
            const trans = msg.payload as TranslationResultPayload;
            if (trans?.direction === "speech_to_sign") {
              if (trans.representation) {
                setAvatarRepresentation(trans.representation);
              }
              if (trans.aslTokens) {
                setAslTokens(
                  trans.aslTokens.map((t) => ({
                    gloss: t.gloss,
                    pos: t.partOfSpeech,
                    marker: `EYEBROWS: ${t.nonManualMarkers.eyebrows.toUpperCase()}`,
                  })),
                );
              } else if (trans.tokens) {
                setAslTokens(
                  trans.tokens.map((t) => ({
                    gloss: t.gloss,
                    pos: "GLOSS",
                    marker: "NEUTRAL",
                  })),
                );
              }
            } else if (trans?.direction === "sign_to_speech" && trans.text) {
              setReconstructedText(trans.text);
              setReconstructionConfidence(trans.confidence);
            }
            break;
          }

          case "tts_audio": {
            // Server synthesized audio received
            break;
          }
        }
      } catch {
        // Safe parse skip
      }
    };

    socket.onclose = () => {
      setIsConnected(false);
      if (pingIntervalRef.current) {
        clearInterval(pingIntervalRef.current);
      }
    };

    socket.onerror = () => {
      setIsConnected(false);
    };

    return () => {
      if (pingIntervalRef.current) {
        clearInterval(pingIntervalRef.current);
      }
      socket.close();
    };
  }, [autoSpeak, clientId, sendRealtimeMessage, sessionId, speakText]);

  // Handle Webcam Feed toggle
  const toggleCamera = useCallback(async () => {
    if (cameraActive) {
      if (mediaStreamRef.current) {
        mediaStreamRef.current.getTracks().forEach((track) => track.stop());
        mediaStreamRef.current = null;
      }
      if (videoRef.current) {
        videoRef.current.srcObject = null;
      }
      setCameraActive(false);
    } else {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: { width: 640, height: 480, facingMode: "user" },
          audio: false,
        });
        mediaStreamRef.current = stream;
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          void videoRef.current.play();
        }
        setCameraActive(true);
      } catch {
        // Fallback gracefully if camera permission denied or unavailable
        setCameraActive(false);
      }
    }
  }, [cameraActive]);

  // Render Skeletal Landmark Overlay on Canvas
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) {
      return;
    }
    const ctx = canvas.getContext("2d");
    if (!ctx) {
      return;
    }

    let animationId: number;

    const drawSkeletonOverlay = () => {
      animationId = requestAnimationFrame(drawSkeletonOverlay);
      const w = canvas.width;
      const h = canvas.height;
      ctx.clearRect(0, 0, w, h);

      if (!overlayActive) {
        return;
      }

      const time = performance.now() * 0.002;
      const sway = Math.sin(time) * 12;

      // Draw Upper Body Skeletal Landmarks
      const headX = w * 0.5 + sway * 0.3;
      const headY = h * 0.28;
      const chestX = w * 0.5 + sway * 0.2;
      const chestY = h * 0.52;
      const leftShoulderX = chestX - 70;
      const leftShoulderY = chestY - 20;
      const rightShoulderX = chestX + 70;
      const rightShoulderY = chestY - 20;
      const leftElbowX = leftShoulderX - 45;
      const leftElbowY = leftShoulderY + 80;
      const rightElbowX = rightShoulderX + 45;
      const rightElbowY = rightShoulderY + 80;
      const leftWristX = leftElbowX + 25 + Math.sin(time * 1.5) * 15;
      const leftWristY = leftElbowY + 65;
      const rightWristX = rightElbowX - 25 - Math.sin(time * 1.5) * 15;
      const rightWristY = rightElbowY + 65;

      // Skeletal Bone Connections
      ctx.strokeStyle = "rgba(37, 99, 235, 0.75)";
      ctx.lineWidth = 3;
      ctx.lineCap = "round";

      ctx.beginPath();
      // Head to Chest
      ctx.moveTo(headX, headY + 25);
      ctx.lineTo(chestX, chestY);

      // Shoulders
      ctx.moveTo(leftShoulderX, leftShoulderY);
      ctx.lineTo(rightShoulderX, rightShoulderY);

      // Left Arm
      ctx.moveTo(leftShoulderX, leftShoulderY);
      ctx.lineTo(leftElbowX, leftElbowY);
      ctx.lineTo(leftWristX, leftWristY);

      // Right Arm
      ctx.moveTo(rightShoulderX, rightShoulderY);
      ctx.lineTo(rightElbowX, rightElbowY);
      ctx.lineTo(rightWristX, rightWristY);
      ctx.stroke();

      // Skeletal Joint Nodes
      const joints = [
        { x: headX, y: headY, r: 24, isHead: true },
        { x: chestX, y: chestY, r: 6 },
        { x: leftShoulderX, y: leftShoulderY, r: 6 },
        { x: rightShoulderX, y: rightShoulderY, r: 6 },
        { x: leftElbowX, y: leftElbowY, r: 5 },
        { x: rightElbowX, y: rightElbowY, r: 5 },
        { x: leftWristX, y: leftWristY, r: 7, isHand: true },
        { x: rightWristX, y: rightWristY, r: 7, isHand: true },
      ];

      for (const joint of joints) {
        ctx.beginPath();
        if (joint.isHead) {
          ctx.ellipse(joint.x, joint.y, joint.r, joint.r * 1.2, 0, 0, Math.PI * 2);
          ctx.strokeStyle = "rgba(13, 148, 136, 0.85)";
          ctx.lineWidth = 2;
          ctx.stroke();
        } else if (joint.isHand) {
          ctx.arc(joint.x, joint.y, joint.r, 0, Math.PI * 2);
          ctx.fillStyle = "#3b82f6";
          ctx.fill();
          ctx.strokeStyle = "#ffffff";
          ctx.lineWidth = 2;
          ctx.stroke();
        } else {
          ctx.arc(joint.x, joint.y, joint.r, 0, Math.PI * 2);
          ctx.fillStyle = "#10b981";
          ctx.fill();
        }
      }
    };

    drawSkeletonOverlay();

    return () => {
      cancelAnimationFrame(animationId);
    };
  }, [overlayActive]);

  // Trigger simulated/live sign detection
  const handleDetectSign = useCallback(
    (gloss: string) => {
      const updatedGlosses = [...activeGlosses, gloss];
      setActiveGlosses(updatedGlosses);

      const reconstruction = reconstructSentence(updatedGlosses, {
        confidence: 0.95,
      });

      setReconstructedText(reconstruction.englishText);
      setReconstructionConfidence(reconstruction.confidence);

      const now = Date.now();
      const signPayload: SignDetectedPayload = {
        gloss,
        confidence: 0.95,
        startTimeMs: now - 500,
        endTimeMs: now,
        durationMs: 500,
      };
      sendRealtimeMessage("sign_detected", signPayload);

      if (autoSpeak && reconstruction.englishText) {
        speakText(reconstruction.englishText);
      }
    },
    [activeGlosses, autoSpeak, sendRealtimeMessage, speakText],
  );

  // Clear Sign buffer
  const handleClearSigns = useCallback(() => {
    setActiveGlosses([]);
    setReconstructedText("");
  }, []);

  // Send hearing participant speech to sign translation
  const handleSendSpeech = useCallback(
    (customText?: string) => {
      const textToTranslate = (customText ?? speechInput).trim();
      if (!textToTranslate) {
        return;
      }

      setSpeechTranscript(textToTranslate);
      setSpeechInput("");

      const words = textToTranslate
        .toUpperCase()
        .replace(/[^A-Z\s]/g, "")
        .split(/\s+/)
        .filter(Boolean);

      const simulatedAsl: AslTokenDisplay[] = words.map((word) => ({
        gloss: word,
        pos: "NOUN",
        marker: word === "HOW" || word === "WHAT" ? "EYEBROWS: FURROW" : "NEUTRAL",
      }));
      setAslTokens(simulatedAsl);

      // Construct and trigger SignRepresentation on Three.js avatar
      const representation = buildFallbackSignRepresentation(words, sessionId);
      setAvatarRepresentation(representation);

      // Transmit transcript update over WebSocket
      const transcriptPayload: TranscriptUpdatePayload = {
        transcript: textToTranslate,
        isFinal: true,
        confidence: 0.96,
        durationMs: words.length * 600,
      };
      sendRealtimeMessage("transcript_update", transcriptPayload);
    },
    [sendRealtimeMessage, speechInput],
  );

  return (
    <div className={styles.pageContainer}>
      {/* Top Teleconference Navigation Bar */}
      <header className={styles.topBar}>
        <div className={styles.brandCluster}>
          <Link href="/" className={styles.brandTitle}>
            Converse Meeting
          </Link>
          <span className={styles.roomBadge}>{sessionId}</span>
        </div>

        <div className={styles.telemetryCluster}>
          <div
            className={`${styles.statusPill} ${
              isConnected
                ? styles.statusPillConnected
                : styles.statusPillDisconnected
            }`}
          >
            <div className={styles.statusIndicatorDot} />
            <span>{isConnected ? "LIVE GATEWAY" : "STANDALONE MODE"}</span>
          </div>

          <div className={styles.latencyBadge}>{latencyMs}ms</div>

          <Link href="/" className={styles.navActionBtn}>
            Exit Meeting
          </Link>
        </div>
      </header>

      {/* Main Teleconference Dual-Pane Layout */}
      <main className={styles.meetingGrid}>
        {/* LEFT PANE: Local Deaf Signer View */}
        <section
          className={styles.paneCard}
          aria-label="Local Deaf Signer (Sign-to-Speech)"
        >
          <div className={styles.paneHeader}>
            <div className={styles.paneTitleBlock}>
              <IconHandGesture className={styles.paneIcon} />
              <div>
                <h2 className={styles.paneTitle}>Local Deaf Signer</h2>
                <p className={styles.paneSubtitle}>
                  Vision landmark tracker and English speech synthesis
                </p>
              </div>
            </div>

            <div className={styles.controlGroup}>
              <button
                type="button"
                onClick={() => void toggleCamera()}
                className={`${styles.stageBtn} ${cameraActive ? styles.stageBtnActive : ""}`}
              >
                {cameraActive ? "Camera On" : "Enable Camera"}
              </button>
              <button
                type="button"
                onClick={() => setOverlayActive(!overlayActive)}
                className={`${styles.stageBtn} ${overlayActive ? styles.stageBtnActive : ""}`}
              >
                {overlayActive ? "Overlay On" : "Overlay Off"}
              </button>
            </div>
          </div>

          <div className={styles.paneContent}>
            {/* Webcam Video Stage & Canvas Landmark Overlay */}
            <div className={styles.videoStage}>
              <video
                ref={videoRef}
                playsInline
                muted
                className={styles.webcamVideo}
                style={{ display: cameraActive ? "block" : "none" }}
              />

              {!cameraActive && (
                <div className={styles.noCameraNotice}>
                  <p>Webcam preview is currently inactive.</p>
                  <button
                    type="button"
                    onClick={() => void toggleCamera()}
                    className={styles.stageBtn}
                  >
                    Start Video Stream
                  </button>
                </div>
              )}

              <canvas
                ref={canvasRef}
                width={640}
                height={480}
                className={styles.landmarkOverlayCanvas}
              />
            </div>

            {/* Real-time Sign Detection Preview */}
            <div className={styles.detectionSection}>
              <div
                style={{
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "space-between",
                }}
              >
                <span className={styles.sectionLabel}>
                  Active Detected Signs (Vision Pipeline)
                </span>
                <button
                  type="button"
                  onClick={handleClearSigns}
                  className={styles.sampleTriggerBtn}
                >
                  Clear Buffer
                </button>
              </div>

              <div className={styles.glossChipsRow}>
                {activeGlosses.length > 0 ? (
                  activeGlosses.map((gloss, idx) => (
                    <span
                      key={`${gloss}-${idx}`}
                      className={styles.glossChip}
                    >
                      {gloss}
                    </span>
                  ))
                ) : (
                  <span
                    style={{
                      fontSize: "0.8rem",
                      color: "var(--text-muted)",
                      fontStyle: "italic",
                    }}
                  >
                    Awaiting hand gesture input...
                  </span>
                )}
              </div>

              {/* Sample Signs Trigger Palette */}
              <div style={{ marginTop: "4px" }}>
                <span
                  className={styles.sectionLabel}
                  style={{ display: "block", marginBottom: "4px" }}
                >
                  Quick Sign Dispatcher:
                </span>
                <div className={styles.sampleTriggerRow}>
                  {SAMPLE_SIGNS.map((sign) => (
                    <button
                      key={sign}
                      type="button"
                      onClick={() => handleDetectSign(sign)}
                      className={styles.sampleTriggerBtn}
                    >
                      +{sign}
                    </button>
                  ))}
                </div>
              </div>
            </div>

            {/* Reconstructed English Speech Output */}
            <div className={styles.reconstructionBox}>
              <div className={styles.reconstructionHeader}>
                <span className={styles.sectionLabel}>
                  Reconstructed English (Spoken Output)
                </span>
                <span
                  style={{
                    fontSize: "0.72rem",
                    fontWeight: 600,
                    color: "var(--status-ok-text)",
                  }}
                >
                  {Math.round(reconstructionConfidence * 100)}% Confidence
                </span>
              </div>

              <p className={styles.reconstructionText}>
                {reconstructedText ? (
                  `"${reconstructedText}"`
                ) : (
                  <span style={{ color: "var(--text-muted)", fontStyle: "italic" }}>
                    Sentence will reconstruct here as signs arrive...
                  </span>
                )}
              </p>

              <div className={styles.reconstructionActions}>
                <button
                  type="button"
                  onClick={() => speakText(reconstructedText)}
                  disabled={!reconstructedText}
                  className={styles.speakBtn}
                >
                  Speak Output
                </button>

                <label
                  style={{
                    display: "flex",
                    alignItems: "center",
                    gap: "6px",
                    fontSize: "0.75rem",
                    color: "var(--text-secondary)",
                    cursor: "pointer",
                  }}
                >
                  <input
                    type="checkbox"
                    checked={autoSpeak}
                    onChange={(e) => setAutoSpeak(e.target.checked)}
                  />
                  Auto-speak on arrival
                </label>
              </div>
            </div>
          </div>
        </section>

        {/* RIGHT PANE: Hearing Participant View */}
        <section
          className={styles.paneCard}
          aria-label="Hearing Participant (Speech-to-Sign)"
        >
          <div className={styles.paneHeader}>
            <div className={styles.paneTitleBlock}>
              <IconAudioWave className={styles.paneIcon} />
              <div>
                <h2 className={styles.paneTitle}>Hearing Participant</h2>
                <p className={styles.paneSubtitle}>
                  ASL linguistic compiler and WebGL avatar signing
                </p>
              </div>
            </div>

            <div className={styles.speedSelector}>
              <span>Speed:</span>
              <button
                type="button"
                onClick={() => setPlaybackSpeed(0.75)}
                className={`${styles.sampleTriggerBtn} ${playbackSpeed === 0.75 ? styles.stageBtnActive : ""}`}
              >
                0.75x
              </button>
              <button
                type="button"
                onClick={() => setPlaybackSpeed(1.0)}
                className={`${styles.sampleTriggerBtn} ${playbackSpeed === 1.0 ? styles.stageBtnActive : ""}`}
              >
                1.0x
              </button>
              <button
                type="button"
                onClick={() => setPlaybackSpeed(1.5)}
                className={`${styles.sampleTriggerBtn} ${playbackSpeed === 1.5 ? styles.stageBtnActive : ""}`}
              >
                1.5x
              </button>
            </div>
          </div>

          <div className={styles.paneContent}>
            {/* Live Three.js WebGL Avatar Viewport */}
            <div className={styles.avatarViewport}>
              <WebglAvatar
                representation={avatarRepresentation}
                playbackSpeed={playbackSpeed}
                enableIdleMotion={true}
              />
            </div>

            {/* Spoken Speech Transcript & Interactive Speech Sender */}
            <div className={styles.detectionSection}>
              <div
                style={{
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "space-between",
                }}
              >
                <span className={styles.sectionLabel}>
                  Spoken English Transcript (Hearing Stream)
                </span>
                <span
                  style={{
                    fontSize: "0.72rem",
                    fontWeight: 600,
                    color: "var(--brand-accent)",
                  }}
                >
                  Captions Active
                </span>
              </div>

              <div className={styles.reconstructionText} style={{ fontSize: "0.95rem" }}>
                {speechTranscript ? `"${speechTranscript}"` : "Awaiting speech..."}
              </div>

              {/* Input for Hearing Participant Speech */}
              <div className={styles.speechInputBox}>
                <input
                  type="text"
                  value={speechInput}
                  onChange={(e) => setSpeechInput(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === "Enter") {
                      handleSendSpeech();
                    }
                  }}
                  placeholder="Type spoken sentence to translate to ASL..."
                  className={styles.speechInput}
                />
                <button
                  type="button"
                  onClick={() => handleSendSpeech()}
                  className={styles.sendBtn}
                >
                  Sign Text
                </button>
              </div>

              {/* Quick Speech Phrases */}
              <div className={styles.sampleTriggerRow} style={{ marginTop: "4px" }}>
                {SAMPLE_HEARING_PHRASES.map((phrase) => (
                  <button
                    key={phrase}
                    type="button"
                    onClick={() => handleSendSpeech(phrase)}
                    className={styles.sampleTriggerBtn}
                  >
                    {phrase}
                  </button>
                ))}
              </div>
            </div>

            {/* English-to-ASL Translation Token Rail */}
            <div className={styles.detectionSection}>
              <div
                style={{
                  display: "flex",
                  alignItems: "center",
                  gap: "6px",
                  marginBottom: "4px",
                }}
              >
                <IconActivity
                  style={{ width: "14px", height: "14px", color: "var(--brand-primary)" }}
                />
                <span className={styles.sectionLabel}>
                  Compiled ASL Gloss Tokens &amp; Non-Manual Markers
                </span>
              </div>

              <div className={styles.tokenRail}>
                {aslTokens.map((tok, idx) => (
                  <div key={`${tok.gloss}-${idx}`} className={styles.railTokenCard}>
                    <span className={styles.tokenGloss}>{tok.gloss}</span>
                    <span className={styles.tokenMetaTag}>{tok.pos}</span>
                    <span className={styles.tokenMetaTag}>{tok.marker}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </section>
      </main>
    </div>
  );
}
