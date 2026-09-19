"use client";

import styles from "../../app/marketing.module.css";
import {
  IconArrowRight,
  IconShieldCheck,
  IconGauge,
  IconAccessibility,
  IconCpu,
  IconExchange,
  IconHandGesture,
  IconAudioWave,
  IconActivity,
} from "./icons";

export function Hero() {
  return (
    <section className={styles.hero} aria-labelledby="hero-title">
      <div className={styles.container}>
        <div className={styles.heroTag}>
          <IconExchange style={{ width: "14px", height: "14px", color: "var(--brand-primary)" }} aria-hidden="true" />
          <span>Real-Time Bidirectional ASL &amp; English Engine</span>
        </div>

        <h1 id="hero-title" className={styles.heroTitle}>
          Sign to Voice. <br />
          <span className={styles.heroHighlight}>Voice to Sign.</span>
        </h1>

        <p className={styles.heroDescription}>
          Converse bridges American Sign Language and spoken English in real time with sub-500ms latency, local-first edge privacy, and zero video cloud storage.
        </p>

        {/* Action Buttons */}
        <div className={styles.heroActions}>
          <a href="#demo" className={styles.btnPrimary}>
            <span>Explore Full Pipeline Demo</span>
            <IconArrowRight style={{ width: "16px", height: "16px" }} aria-hidden="true" />
          </a>
          <a href="#architecture" className={styles.btnSecondary}>
            <span>View Architecture</span>
          </a>
        </div>

        {/* 3 Visual Pillars */}
        <div className={styles.visualCardsGrid}>
          <div className={styles.visualHeroCard}>
            <div className={styles.visualCardIcon} aria-hidden="true">
              <IconHandGesture style={{ width: "22px", height: "22px" }} />
            </div>
            <div>
              <div className={styles.visualCardTitle}>Sign to Speech</div>
              <div className={styles.visualCardDesc}>Continuous ASL vision recognition to natural voice output</div>
            </div>
          </div>

          <div className={styles.visualHeroCard}>
            <div className={styles.visualCardIcon} aria-hidden="true">
              <IconActivity style={{ width: "22px", height: "22px" }} />
            </div>
            <div>
              <div className={styles.visualCardTitle}>Under 450ms Latency</div>
              <div className={styles.visualCardDesc}>Local neural models running directly on device edge</div>
            </div>
          </div>

          <div className={styles.visualHeroCard}>
            <div className={styles.visualCardIcon} aria-hidden="true">
              <IconAudioWave style={{ width: "22px", height: "22px" }} />
            </div>
            <div>
              <div className={styles.visualCardTitle}>Speech to Sign</div>
              <div className={styles.visualCardDesc}>Real-time spoken English converted to visual ASL gestures</div>
            </div>
          </div>
        </div>

        {/* Trust Guarantees */}
        <div className={styles.trustBar} role="list" aria-label="System Guarantees">
          <div className={styles.trustItem} role="listitem">
            <IconShieldCheck className={styles.trustIcon} aria-hidden="true" />
            <span>Local-First Edge Privacy</span>
          </div>
          <div className={styles.trustItem} role="listitem">
            <IconGauge className={styles.trustIcon} aria-hidden="true" />
            <span>Sub-Second Response</span>
          </div>
          <div className={styles.trustItem} role="listitem">
            <IconAccessibility className={styles.trustIcon} aria-hidden="true" />
            <span>WCAG AAA Accessible Design</span>
          </div>
          <div className={styles.trustItem} role="listitem">
            <IconCpu className={styles.trustIcon} aria-hidden="true" />
            <span>Zero Video Cloud Storage</span>
          </div>
        </div>
      </div>
    </section>
  );
}
