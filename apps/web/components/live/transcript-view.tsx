"use client";

import styles from "../../app/marketing.module.css";
import { IconHandGesture, IconAudioWave, IconActivity } from "../marketing/icons";

export interface TranscriptViewProps {
  /** The sequence of raw stabilized ASL gloss tokens */
  activeGlosses: readonly string[];
  /** The reconstructed fluent English sentence */
  reconstructedText: string;
  /** Whether the sentence boundary pause has finalized this sentence */
  isFinal: boolean;
  /** Processing latency in milliseconds */
  latencyMs?: number;
  /** Overall confidence score (0.0 to 1.0) */
  confidence?: number;
  /** Callback to clear the transcript buffer */
  onClear?: () => void;
}

export function TranscriptView({
  activeGlosses,
  reconstructedText,
  isFinal,
  latencyMs,
  confidence,
  onClear,
}: TranscriptViewProps) {
  const confidencePercent = confidence !== undefined ? Math.round(confidence * 100) : null;

  return (
    <div
      className={styles.transcriptContainer}
      role="region"
      aria-label="Real-time ASL and English Dual Transcript"
    >
      {/* Header Metadata */}
      <div className={styles.transcriptHeader}>
        <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
          <IconActivity style={{ width: "16px", height: "16px", color: "var(--brand-primary)" }} aria-hidden="true" />
          <span style={{ fontSize: "0.825rem", fontWeight: 700, color: "var(--text-primary)" }}>
            Dual-Stream Linguistic Pipeline
          </span>
        </div>

        <div style={{ display: "flex", alignItems: "center", gap: "10px" }}>
          {confidencePercent !== null && (
            <span
              style={{
                fontSize: "0.75rem",
                fontWeight: 600,
                color: "var(--status-ok-text)",
                backgroundColor: "var(--status-ok-bg)",
                padding: "2px 8px",
                borderRadius: "var(--radius-full)",
                border: "1px solid var(--status-ok-border)",
              }}
            >
              {confidencePercent}% Confidence
            </span>
          )}

          {latencyMs !== undefined && (
            <span
              style={{
                fontSize: "0.75rem",
                fontWeight: 600,
                color: "var(--status-blue-text)",
                backgroundColor: "var(--status-blue-bg)",
                padding: "2px 8px",
                borderRadius: "var(--radius-full)",
                border: "1px solid var(--status-blue-border)",
              }}
            >
              {latencyMs}ms
            </span>
          )}

          {onClear && (
            <button
              type="button"
              onClick={onClear}
              className={styles.stageControlBtn}
              style={{ padding: "3px 10px", fontSize: "0.75rem" }}
            >
              Clear
            </button>
          )}
        </div>
      </div>

      {/* Tier 1: Stabilized ASL Gloss Sequence Stream */}
      <div className={styles.transcriptGlossPane}>
        <div style={{ display: "flex", alignItems: "center", gap: "6px", marginBottom: "6px" }}>
          <IconHandGesture style={{ width: "14px", height: "14px", color: "var(--brand-primary)" }} aria-hidden="true" />
          <span style={{ fontSize: "0.72rem", fontWeight: 700, textTransform: "uppercase", color: "var(--text-muted)", letterSpacing: "0.05em" }}>
            Stabilized ASL Glosses (Vision Stream)
          </span>
        </div>

        <div className={styles.glossRow} style={{ justifyContent: "flex-start", minHeight: "36px" }}>
          {activeGlosses.length > 0 ? (
            activeGlosses.map((gloss, idx) => (
              <span
                key={`${gloss}-${idx}`}
                className={`${styles.glossChip} ${styles.glossChipActive}`}
              >
                {gloss}
              </span>
            ))
          ) : (
            <span style={{ fontSize: "0.825rem", color: "var(--text-muted)", fontStyle: "italic" }}>
              Awaiting sign gesture input...
            </span>
          )}
        </div>
      </div>

      {/* Tier 2: Fluent English Reconstruction Output */}
      <div className={styles.transcriptEnglishPane}>
        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: "6px" }}>
          <div style={{ display: "flex", alignItems: "center", gap: "6px" }}>
            <IconAudioWave style={{ width: "14px", height: "14px", color: "var(--brand-accent)" }} aria-hidden="true" />
            <span style={{ fontSize: "0.72rem", fontWeight: 700, textTransform: "uppercase", color: "var(--text-muted)", letterSpacing: "0.05em" }}>
              Reconstructed English (Spoken Output)
            </span>
          </div>

          <span
            style={{
              fontSize: "0.7rem",
              fontWeight: 600,
              color: isFinal ? "var(--brand-accent)" : "var(--text-muted)",
            }}
          >
            {isFinal ? "● Finalized Sentence" : "○ Streaming Preview"}
          </span>
        </div>

        <div
          className={styles.transcriptSentenceText}
          aria-live="polite"
          aria-atomic="true"
        >
          {reconstructedText ? (
            <span>&ldquo;{reconstructedText}&rdquo;</span>
          ) : (
            <span style={{ color: "var(--text-muted)", fontStyle: "italic", fontSize: "0.95rem" }}>
              English sentence will assemble here as signs arrive...
            </span>
          )}
        </div>
      </div>
    </div>
  );
}
