"use client";

import styles from "../../app/marketing.module.css";
import { IconGauge } from "./icons";

const LATENCY_BUDGET_SEGMENTS = [
  { name: "Frame Capture & Normalization", timeMs: 32, percentage: 11, color: "#2563EB" },
  { name: "3D Landmark Extraction", timeMs: 18, percentage: 6, color: "#0D9488" },
  { name: "ST-GCN Sign Spotting", timeMs: 25, percentage: 9, color: "#6366F1" },
  { name: "Gloss Debounce & Reconstruction", timeMs: 42, percentage: 14, color: "#D97706" },
  { name: "Piper Neural TTS TTFB", timeMs: 135, percentage: 46, color: "#DC2626" },
  { name: "Web Audio Jitter Buffer", timeMs: 40, percentage: 14, color: "#7C3AED" },
];

export function ArchitectureBreakdown() {
  const totalMeasuredMs = LATENCY_BUDGET_SEGMENTS.reduce((acc, curr) => acc + curr.timeMs, 0);

  return (
    <section id="architecture" className={`${styles.section} ${styles.sectionAlt}`} aria-labelledby="arch-title">
      <div className={styles.container}>
        <div className={styles.sectionHeader}>
          <span className={styles.sectionPretitle}>Performance Engineering</span>
          <h2 id="arch-title" className={styles.sectionTitle}>
            Sub-500ms Glass-to-Ear Latency Budget
          </h2>
          <p className={styles.sectionSubtitle}>
            Every microsecond matters in real-time conversational exchange. Converse eliminates
            batch store-and-forward delays with a strictly streaming, chunk-based architecture.
          </p>
        </div>

        {/* Stacked Latency Budget Bar */}
        <div className={styles.latencyBarCard}>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "8px", flexWrap: "wrap", gap: "12px" }}>
            <h3 className={styles.latencyTitle}>Measured End-to-End Latency Profile</h3>
            <div className={styles.badgeLatency}>
              <IconGauge style={{ width: "14px", height: "14px" }} aria-hidden="true" />
              <span>{totalMeasuredMs}ms Total (Target &lt; 500ms)</span>
            </div>
          </div>
          <p className={styles.latencyDesc}>
            Time elapsed from physical hand movement capture to spoken audio playback in the receiver&apos;s headphones.
          </p>

          <div className={styles.budgetStackedBar} role="progressbar" aria-valuenow={totalMeasuredMs} aria-valuemin={0} aria-valuemax={500} aria-label="Latency breakdown">
            {LATENCY_BUDGET_SEGMENTS.map((seg, idx) => (
              <div
                key={idx}
                className={styles.budgetSegment}
                style={{
                  width: `${(seg.timeMs / totalMeasuredMs) * 100}%`,
                  backgroundColor: seg.color,
                }}
                title={`${seg.name}: ${seg.timeMs}ms`}
              >
                <span>{seg.timeMs}ms</span>
              </div>
            ))}
          </div>

          <div className={styles.budgetLegendGrid}>
            {LATENCY_BUDGET_SEGMENTS.map((seg, idx) => (
              <div key={idx} className={styles.budgetLegendItem}>
                <div className={styles.legendColorBox} style={{ backgroundColor: seg.color }} aria-hidden="true" />
                <div>
                  <div className={styles.legendLabel}>{seg.name}</div>
                  <div className={styles.legendTime}>{seg.timeMs}ms ({Math.round((seg.timeMs / totalMeasuredMs) * 100)}%)</div>
                </div>
              </div>
            ))}
          </div>
        </div>


      </div>
    </section>
  );
}
