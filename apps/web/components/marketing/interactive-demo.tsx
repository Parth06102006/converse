"use client";

import { useState, useEffect, useRef } from "react";
import styles from "../../app/marketing.module.css";
import {
  IconCamera,
  IconMic,
  IconPlay,
  IconPause,
  IconStepForward,
  IconRotateCcw,
  IconAudioWave,
  IconVolume2,
  IconCheck,
  IconCpu,
} from "./icons";

type PipelineDirection = "sign_to_speech" | "speech_to_sign";

interface Scenario {
  id: string;
  name: string;
  glosses: Array<{ gloss: string; confidence: number; timestampMs: number }>;
  englishText: string;
  nmm?: string;
  latencyBreakdown: {
    stage1: number;
    stage2: number;
    stage3: number;
    stage4: number;
  };
}

type ScenarioKey = "greeting" | "inquiry" | "gratitude";

const DEFAULT_SCENARIO: Scenario = {
  id: "greeting",
  name: "Greeting",
  glosses: [
    { gloss: "HELLO", confidence: 0.96, timestampMs: 120 },
    { gloss: "NICE", confidence: 0.91, timestampMs: 460 },
    { gloss: "MEET", confidence: 0.89, timestampMs: 820 },
    { gloss: "YOU", confidence: 0.94, timestampMs: 1140 },
  ],
  englishText: "Hello, nice to meet you.",
  nmm: "Head nod, neutral eyebrows",
  latencyBreakdown: { stage1: 32, stage2: 24, stage3: 42, stage4: 135 },
};

const SCENARIOS: Record<ScenarioKey, Scenario> = {
  greeting: DEFAULT_SCENARIO,
  inquiry: {
    id: "inquiry",
    name: "Inquiry (Wh-Question)",
    glosses: [
      { gloss: "LIBRARY", confidence: 0.93, timestampMs: 180 },
      { gloss: "WHERE", confidence: 0.95, timestampMs: 640 },
    ],
    englishText: "Where is the library?",
    nmm: "Eyebrows furrowed, head forward (Wh-Question marker)",
    latencyBreakdown: { stage1: 30, stage2: 22, stage3: 38, stage4: 128 },
  },
  gratitude: {
    id: "gratitude",
    name: "Assistance & Thanks",
    glosses: [
      { gloss: "THANK-YOU", confidence: 0.98, timestampMs: 140 },
      { gloss: "HELP", confidence: 0.92, timestampMs: 580 },
    ],
    englishText: "Thank you for your help.",
    nmm: "Affirmative smile, gentle head nod",
    latencyBreakdown: { stage1: 28, stage2: 20, stage3: 35, stage4: 122 },
  },
};

export function InteractiveDemo() {
  const [direction, setDirection] = useState<PipelineDirection>("sign_to_speech");
  const [activeScenarioKey, setActiveScenarioKey] = useState<ScenarioKey>("greeting");
  const [currentStage, setCurrentStage] = useState<number>(1);
  const [isPlaying, setIsPlaying] = useState<boolean>(false);
  const [playbackSpeed, setPlaybackSpeed] = useState<number>(1.0);
  const [isSpeaking, setIsSpeaking] = useState<boolean>(false);
  const [skeletonTick, setSkeletonTick] = useState<number>(0);

  const scenario: Scenario = SCENARIOS[activeScenarioKey] ?? DEFAULT_SCENARIO;

  // Animation frame ticker for visual skeleton motion
  useEffect(() => {
    let animId: number;
    const animate = () => {
      setSkeletonTick((prev) => (prev + 1) % 360);
      animId = requestAnimationFrame(animate);
    };
    animId = requestAnimationFrame(animate);
    return () => cancelAnimationFrame(animId);
  }, []);

  // Auto-progression when playing
  useEffect(() => {
    if (!isPlaying) return;

    const intervalTime = 1800 / playbackSpeed;
    const timer = setInterval(() => {
      setCurrentStage((prev) => {
        if (prev >= 4) {
          setIsPlaying(false);
          return 4;
        }
        return prev + 1;
      });
    }, intervalTime);

    return () => clearInterval(timer);
  }, [isPlaying, playbackSpeed]);

  // Audio synthesis trigger when arriving at stage 4
  const speakUtterance = (text: string) => {
    if (typeof window === "undefined" || !("speechSynthesis" in window)) return;
    window.speechSynthesis.cancel();
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.rate = playbackSpeed;
    utterance.onstart = () => setIsSpeaking(true);
    utterance.onend = () => setIsSpeaking(false);
    utterance.onerror = () => setIsSpeaking(false);
    window.speechSynthesis.speak(utterance);
  };

  const handlePlayToggle = () => {
    if (isPlaying) {
      setIsPlaying(false);
    } else {
      if (currentStage >= 4) {
        setCurrentStage(1);
      }
      setIsPlaying(true);
    }
  };

  const handleStepForward = () => {
    setIsPlaying(false);
    setCurrentStage((prev) => Math.min(4, prev + 1));
  };

  const handleReset = () => {
    setIsPlaying(false);
    setCurrentStage(1);
    if (typeof window !== "undefined" && "speechSynthesis" in window) {
      window.speechSynthesis.cancel();
      setIsSpeaking(false);
    }
  };

  // Dynamic skeletal keypoint coordinates based on tick
  const handOffset = Math.sin((skeletonTick * Math.PI) / 180) * 12;
  const wristY = 220 + handOffset;
  const palmY = 175 + handOffset * 1.3;

  return (
    <section id="demo" className={styles.section} aria-labelledby="demo-heading">
      <div className={styles.container}>
        <div className={styles.sectionHeader}>
          <span className={styles.sectionPretitle}>Interactive System Simulation</span>
          <h2 id="demo-heading" className={styles.sectionTitle}>
            Experience the Bidirectional Pipeline
          </h2>
          <p className={styles.sectionSubtitle}>
            Test canonical conversational scenarios across each transformation stage. Scrub
            through the pipeline, inspect landmark feature extraction, and listen to synthesized speech.
          </p>
        </div>

        <div className={styles.demoCard}>
          {/* Top Control Bar */}
          <div className={styles.demoTopBar}>
            <div className={styles.directionToggle} role="tablist" aria-label="Pipeline Direction">
              <button
                type="button"
                role="tab"
                aria-selected={direction === "sign_to_speech"}
                className={`${styles.toggleBtn} ${direction === "sign_to_speech" ? styles.toggleBtnActive : ""}`}
                onClick={() => {
                  setDirection("sign_to_speech");
                  handleReset();
                }}
              >
                <IconCamera style={{ width: "16px", height: "16px" }} aria-hidden="true" />
                <span>Sign to Speech</span>
              </button>
              <button
                type="button"
                role="tab"
                aria-selected={direction === "speech_to_sign"}
                className={`${styles.toggleBtn} ${direction === "speech_to_sign" ? styles.toggleBtnActive : ""}`}
                onClick={() => {
                  setDirection("speech_to_sign");
                  handleReset();
                }}
              >
                <IconMic style={{ width: "16px", height: "16px" }} aria-hidden="true" />
                <span>Speech to Sign</span>
              </button>
            </div>

            <div className={styles.scenarioSelectGroup} role="group" aria-label="Scenario Selection">
              <span className={styles.scenarioLabel}>Preset Scenario:</span>
              {(Object.entries(SCENARIOS) as Array<[ScenarioKey, Scenario]>).map(([key, s]) => (
                <button
                  key={key}
                  type="button"
                  className={`${styles.scenarioBtn} ${activeScenarioKey === key ? styles.scenarioBtnActive : ""}`}
                  onClick={() => {
                    setActiveScenarioKey(key);
                    handleReset();
                  }}
                >
                  {s.name}
                </button>
              ))}
            </div>
          </div>

          {/* Stepper Navigation */}
          <div className={styles.stepperBar} role="tablist" aria-label="Pipeline Stages">
            {direction === "sign_to_speech" ? (
              <>
                <button
                  type="button"
                  role="tab"
                  aria-selected={currentStage === 1}
                  className={`${styles.stepItem} ${currentStage === 1 ? styles.stepItemActive : ""}`}
                  onClick={() => setCurrentStage(1)}
                >
                  <span className={`${styles.stepBadge} ${currentStage >= 1 ? styles.stepBadgeActive : ""}`}>1</span>
                  <div>
                    <span className={styles.stepTextTitle}>3D Landmarks</span>
                    <span className={styles.stepTextDesc}>Camera & Keypoint Mesh</span>
                  </div>
                </button>
                <button
                  type="button"
                  role="tab"
                  aria-selected={currentStage === 2}
                  className={`${styles.stepItem} ${currentStage === 2 ? styles.stepItemActive : ""}`}
                  onClick={() => setCurrentStage(2)}
                >
                  <span className={`${styles.stepBadge} ${currentStage >= 2 ? styles.stepBadgeActive : ""}`}>2</span>
                  <div>
                    <span className={styles.stepTextTitle}>Candidate Signs</span>
                    <span className={styles.stepTextDesc}>ST-GCN Spotting</span>
                  </div>
                </button>
                <button
                  type="button"
                  role="tab"
                  aria-selected={currentStage === 3}
                  className={`${styles.stepItem} ${currentStage === 3 ? styles.stepItemActive : ""}`}
                  onClick={() => setCurrentStage(3)}
                >
                  <span className={`${styles.stepBadge} ${currentStage >= 3 ? styles.stepBadgeActive : ""}`}>3</span>
                  <div>
                    <span className={styles.stepTextTitle}>Sentence Recovery</span>
                    <span className={styles.stepTextDesc}>Linguistic Grammar</span>
                  </div>
                </button>
                <button
                  type="button"
                  role="tab"
                  aria-selected={currentStage === 4}
                  className={`${styles.stepItem} ${currentStage === 4 ? styles.stepItemActive : ""}`}
                  onClick={() => {
                    setCurrentStage(4);
                    speakUtterance(scenario.englishText);
                  }}
                >
                  <span className={`${styles.stepBadge} ${currentStage >= 4 ? styles.stepBadgeActive : ""}`}>4</span>
                  <div>
                    <span className={styles.stepTextTitle}>Voice Output</span>
                    <span className={styles.stepTextDesc}>Web Audio TTS</span>
                  </div>
                </button>
              </>
            ) : (
              <>
                <button
                  type="button"
                  role="tab"
                  aria-selected={currentStage === 1}
                  className={`${styles.stepItem} ${currentStage === 1 ? styles.stepItemActive : ""}`}
                  onClick={() => setCurrentStage(1)}
                >
                  <span className={`${styles.stepBadge} ${currentStage >= 1 ? styles.stepBadgeActive : ""}`}>1</span>
                  <div>
                    <span className={styles.stepTextTitle}>Audio Ingestion</span>
                    <span className={styles.stepTextDesc}>16kHz VAD Sampling</span>
                  </div>
                </button>
                <button
                  type="button"
                  role="tab"
                  aria-selected={currentStage === 2}
                  className={`${styles.stepItem} ${currentStage === 2 ? styles.stepItemActive : ""}`}
                  onClick={() => setCurrentStage(2)}
                >
                  <span className={`${styles.stepBadge} ${currentStage >= 2 ? styles.stepBadgeActive : ""}`}>2</span>
                  <div>
                    <span className={styles.stepTextTitle}>Streaming ASR</span>
                    <span className={styles.stepTextDesc}>Continuous Transcript</span>
                  </div>
                </button>
                <button
                  type="button"
                  role="tab"
                  aria-selected={currentStage === 3}
                  className={`${styles.stepItem} ${currentStage === 3 ? styles.stepItemActive : ""}`}
                  onClick={() => setCurrentStage(3)}
                >
                  <span className={`${styles.stepBadge} ${currentStage >= 3 ? styles.stepBadgeActive : ""}`}>3</span>
                  <div>
                    <span className={styles.stepTextTitle}>ASL Compiler</span>
                    <span className={styles.stepTextDesc}>Topic-Comment Syntax</span>
                  </div>
                </button>
                <button
                  type="button"
                  role="tab"
                  aria-selected={currentStage === 4}
                  className={`${styles.stepItem} ${currentStage === 4 ? styles.stepItemActive : ""}`}
                  onClick={() => setCurrentStage(4)}
                >
                  <span className={`${styles.stepBadge} ${currentStage >= 4 ? styles.stepBadgeActive : ""}`}>4</span>
                  <div>
                    <span className={styles.stepTextTitle}>3D Avatar Rig</span>
                    <span className={styles.stepTextDesc}>SLERP Skeletal Render</span>
                  </div>
                </button>
              </>
            )}
          </div>

          {/* Stage Viewport */}
          <div className={styles.stageViewport}>
            {/* Visual Pane */}
            <div className={styles.stageVisualPane}>
              {direction === "sign_to_speech" ? (
                <div style={{ width: "100%", height: "280px", display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center" }}>
                  <svg viewBox="0 0 400 320" style={{ width: "100%", height: "100%", maxHeight: "280px" }}>
                    {/* Background bounding box */}
                    <rect x="40" y="20" width="320" height="280" rx="12" fill="#F4EFE6" stroke="#E8E3D7" strokeWidth="1.5" strokeDasharray="4 4" />
                    
                    {/* Head / Face keypoints */}
                    <circle cx="200" cy="80" r="32" fill="#FAF7F2" stroke="#2563EB" strokeWidth="2" />
                    {/* Eyes and Eyebrow non-manual markers */}
                    <path d="M184 72 Q190 68 196 72" stroke="#1D4ED8" strokeWidth="2.5" fill="none" />
                    <path d="M204 72 Q210 68 216 72" stroke="#1D4ED8" strokeWidth="2.5" fill="none" />
                    <circle cx="190" cy="82" r="3" fill="#2563EB" />
                    <circle cx="210" cy="82" r="3" fill="#2563EB" />
                    <path d="M192 98 Q200 104 208 98" stroke="#2563EB" strokeWidth="2" fill="none" />

                    {/* Torso Skeleton */}
                    <line x1="200" y1="112" x2="200" y2="180" stroke="#0D9488" strokeWidth="3" />
                    {/* Shoulders */}
                    <line x1="140" y1="130" x2="260" y2="130" stroke="#0D9488" strokeWidth="3" />
                    <circle cx="140" cy="130" r="5" fill="#0D9488" />
                    <circle cx="260" cy="130" r="5" fill="#0D9488" />

                    {/* Left Arm & Forearm */}
                    <line x1="140" y1="130" x2="110" y2="190" stroke="#2563EB" strokeWidth="2.5" />
                    <circle cx="110" cy="190" r="4" fill="#2563EB" />
                    <line x1="110" y1="190" x2="130" y2="230" stroke="#2563EB" strokeWidth="2" />
                    <circle cx="130" cy="230" r="4" fill="#2563EB" />

                    {/* Right Active Signing Arm & Animated Wrist */}
                    <line x1="260" y1="130" x2="290" y2="180" stroke="#2563EB" strokeWidth="2.5" />
                    <circle cx="290" cy="180" r="4" fill="#2563EB" />
                    <line x1="290" y1="180" x2="270" y2={wristY} stroke="#2563EB" strokeWidth="2.5" />
                    <circle cx="270" cy={wristY} r="5" fill="#2563EB" />

                    {/* 21-point Hand Landmark Web */}
                    <line x1="270" y1={wristY} x2="270" y2={palmY} stroke="#2563EB" strokeWidth="1.5" />
                    <circle cx="270" cy={palmY} r="3" fill="#2563EB" />
                    {/* Fingers */}
                    <line x1="270" y1={palmY} x2="252" y2={palmY - 20} stroke="#2563EB" strokeWidth="1.5" />
                    <line x1="270" y1={palmY} x2="264" y2={palmY - 30} stroke="#2563EB" strokeWidth="1.5" />
                    <line x1="270" y1={palmY} x2="276" y2={palmY - 32} stroke="#2563EB" strokeWidth="1.5" />
                    <line x1="270" y1={palmY} x2="288" y2={palmY - 25} stroke="#2563EB" strokeWidth="1.5" />
                    <circle cx="252" cy={palmY - 20} r="2.5" fill="#0D9488" />
                    <circle cx="264" cy={palmY - 30} r="2.5" fill="#0D9488" />
                    <circle cx="276" cy={palmY - 32} r="2.5" fill="#0D9488" />
                    <circle cx="288" cy={palmY - 25} r="2.5" fill="#0D9488" />

                    {/* Landmark tracking status tag */}
                    <rect x="52" y="32" width="130" height="24" rx="4" fill="#FFFFFF" stroke="#E8E3D7" />
                    <text x="60" y="48" fontSize="11" fontWeight="600" fill="#2563EB">
                      Landmarks: 54 tracked
                    </text>
                  </svg>
                </div>
              ) : (
                <div style={{ width: "100%", height: "280px", display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center" }}>
                  <div className={styles.audioWaveformBox}>
                    {[18, 35, 62, 85, 45, 70, 92, 60, 40, 75, 55, 30, 68, 88, 50].map((h, idx) => (
                      <div
                        key={idx}
                        className={styles.waveBar}
                        style={{
                          height: `${Math.max(12, Math.sin((skeletonTick + idx * 20) * 0.05) * h)}px`,
                        }}
                      />
                    ))}
                  </div>
                  <span style={{ fontSize: "0.85rem", color: "var(--text-secondary)", fontWeight: 500 }}>
                    Streaming 16kHz PCM Audio Stream
                  </span>
                </div>
              )}
            </div>

            {/* Data & State Pane */}
            <div className={styles.stageDataPane}>
              <div>
                <div className={dataPaneHeaderClass(styles)}>
                  <span className={styles.dataPaneTitle}>
                    {direction === "sign_to_speech"
                      ? getSignToSpeechStageTitle(currentStage)
                      : getSpeechToSignStageTitle(currentStage)}
                  </span>
                  <span className={styles.metricPill}>
                    {getStageLatency(scenario, currentStage)}
                  </span>
                </div>

                {direction === "sign_to_speech" ? (
                  <div>
                    {currentStage >= 2 && (
                      <div>
                        <div style={{ fontSize: "0.8rem", color: "var(--text-muted)", fontWeight: 600, marginBottom: "6px" }}>
                          Recognized ASL Gloss Stream:
                        </div>
                        <div className={styles.glossList}>
                          {scenario.glosses.map((item, idx) => (
                            <span key={idx} className={styles.glossTag}>
                              <span>{item.gloss}</span>
                              <span className={styles.glossConfidence}>
                                {Math.round(item.confidence * 100)}%
                              </span>
                            </span>
                          ))}
                        </div>
                      </div>
                    )}

                    {currentStage >= 3 && (
                      <div className={styles.sentenceDisplay}>
                        <div className={styles.sentenceLabel}>Reconstructed English Sentence:</div>
                        <div className={styles.sentenceText}>&ldquo;{scenario.englishText}&rdquo;</div>
                        {scenario.nmm && (
                          <div style={{ fontSize: "0.78rem", color: "var(--text-secondary)", marginTop: "6px" }}>
                            <strong>Facial Cues (NMM):</strong> {scenario.nmm}
                          </div>
                        )}
                      </div>
                    )}

                    {currentStage >= 4 && (
                      <div>
                        <button
                          type="button"
                          className={styles.btnSecondary}
                          style={{ width: "100%", marginTop: "8px", justifyContent: "center" }}
                          onClick={() => speakUtterance(scenario.englishText)}
                        >
                          <IconVolume2 style={{ width: "18px", height: "18px", color: "var(--brand-primary)" }} aria-hidden="true" />
                          <span>{isSpeaking ? "Speaking via Web Audio..." : "Replay Spoken Audio"}</span>
                        </button>
                      </div>
                    )}

                    {currentStage === 1 && (
                      <div style={{ fontSize: "0.875rem", color: "var(--text-secondary)", lineHeight: 1.6 }}>
                        <p style={{ marginBottom: "8px" }}>
                          <strong>MediaPipe Tasks 3D Landmarker</strong> extracts 21 keypoints per hand
                          and 33 skeletal body landmarks at 30 FPS.
                        </p>
                        <p>
                          Coordinates are centered around the shoulder root and normalized against palm
                          length, guaranteeing complete distance and perspective invariance.
                        </p>
                      </div>
                    )}
                  </div>
                ) : (
                  <div>
                    {currentStage >= 2 && (
                      <div className={styles.sentenceDisplay}>
                        <div className={styles.sentenceLabel}>Transcribed Speech:</div>
                        <div className={styles.sentenceText}>&ldquo;{scenario.englishText}&rdquo;</div>
                      </div>
                    )}

                    {currentStage >= 3 && (
                      <div>
                        <div style={{ fontSize: "0.8rem", color: "var(--text-muted)", fontWeight: 600, marginBottom: "6px" }}>
                          Compiled ASL Topic-Comment Sequence:
                        </div>
                        <div className={styles.glossList}>
                          {scenario.glosses.map((item, idx) => (
                            <span key={idx} className={styles.glossTag}>
                              <span>{item.gloss}</span>
                            </span>
                          ))}
                        </div>
                        {scenario.nmm && (
                          <div style={{ fontSize: "0.8rem", color: "var(--text-secondary)", marginTop: "8px" }}>
                            <strong>Active Non-Manual Marker:</strong> {scenario.nmm}
                          </div>
                        )}
                      </div>
                    )}

                    {currentStage >= 4 && (
                      <div style={{ fontSize: "0.875rem", color: "var(--text-secondary)", lineHeight: 1.6 }}>
                        <p style={{ marginBottom: "8px" }}>
                          <strong>Humanoid 3D Skeletal Rigging:</strong> Generated animation tokens are
                          interpolated via Quaternion SLERP to eliminate mechanical jerks.
                        </p>
                        <p>
                          Blendshape morph targets drive subtle facial expressions and head turns at
                          a sustained 60 FPS in WebGL.
                        </p>
                      </div>
                    )}

                    {currentStage === 1 && (
                      <div style={{ fontSize: "0.875rem", color: "var(--text-secondary)", lineHeight: 1.6 }}>
                        <p style={{ marginBottom: "8px" }}>
                          <strong>Silero VAD (Voice Activity Detection)</strong> continuously monitors
                          the 16kHz audio stream, discarding background room noise.
                        </p>
                        <p>
                          Utterances are dynamically segmented upon 300ms acoustic silence boundaries.
                        </p>
                      </div>
                    )}
                  </div>
                )}
              </div>

              {/* Status footer inside data pane */}
              <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", borderTop: "1px solid var(--border-subtle)", paddingTop: "12px", marginTop: "16px" }}>
                <span style={{ fontSize: "0.8rem", color: "var(--text-muted)" }}>
                  Step {currentStage} of 4 Complete
                </span>
                <span style={{ fontSize: "0.8rem", color: "var(--status-ok-text)", fontWeight: 600, display: "flex", alignItems: "center", gap: "4px" }}>
                  <IconCheck style={{ width: "14px", height: "14px" }} aria-hidden="true" />
                  Verified Contract
                </span>
              </div>
            </div>
          </div>

          {/* Bottom Control Bar */}
          <div className={styles.demoControls}>
            <div className={styles.controlButtonGroup}>
              <button
                type="button"
                className={`${styles.iconBtn} ${isPlaying ? styles.iconBtnActive : ""}`}
                title={isPlaying ? "Pause Pipeline" : "Run Pipeline"}
                onClick={handlePlayToggle}
              >
                {isPlaying ? (
                  <IconPause style={{ width: "16px", height: "16px" }} aria-hidden="true" />
                ) : (
                  <IconPlay style={{ width: "16px", height: "16px" }} aria-hidden="true" />
                )}
              </button>
              <button
                type="button"
                className={styles.iconBtn}
                title="Step Forward"
                onClick={handleStepForward}
                disabled={currentStage >= 4}
              >
                <IconStepForward style={{ width: "16px", height: "16px" }} aria-hidden="true" />
              </button>
              <button
                type="button"
                className={styles.iconBtn}
                title="Reset Simulation"
                onClick={handleReset}
              >
                <IconRotateCcw style={{ width: "16px", height: "16px" }} aria-hidden="true" />
              </button>
            </div>

            <div className={styles.speedSelector}>
              <span>Speed:</span>
              {[0.5, 1.0, 1.5].map((speed) => (
                <button
                  key={speed}
                  type="button"
                  className={`${styles.speedBtn} ${playbackSpeed === speed ? styles.speedBtnActive : ""}`}
                  onClick={() => setPlaybackSpeed(speed)}
                >
                  {speed}x
                </button>
              ))}
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}

function dataPaneHeaderClass(s: Record<string, string>) {
  return s.dataPaneHeader || "";
}

function getSignToSpeechStageTitle(stage: number): string {
  switch (stage) {
    case 1:
      return "Stage 1: 3D Landmark Extraction";
    case 2:
      return "Stage 2: Spatiotemporal Sign Spotting";
    case 3:
      return "Stage 3: Sentence Reconstruction";
    case 4:
      return "Stage 4: Neural TTS & Web Audio";
    default:
      return "Pipeline Processing";
  }
}

function getSpeechToSignStageTitle(stage: number): string {
  switch (stage) {
    case 1:
      return "Stage 1: Acoustic Ingestion & VAD";
    case 2:
      return "Stage 2: Continuous Speech Recognition";
    case 3:
      return "Stage 3: ASL Grammar Compiler";
    case 4:
      return "Stage 4: WebGL 3D Avatar Rendering";
    default:
      return "Pipeline Processing";
  }
}

function getStageLatency(scenario: Scenario, stage: number): string {
  const b = scenario.latencyBreakdown;
  switch (stage) {
    case 1:
      return `${b.stage1}ms Latency`;
    case 2:
      return `${b.stage2}ms Latency`;
    case 3:
      return `${b.stage3}ms Latency`;
    case 4:
      return `${b.stage4}ms TTFB`;
    default:
      return "Active";
  }
}
