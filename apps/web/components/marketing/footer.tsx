"use client";

import styles from "../../app/marketing.module.css";
import { IconLogo, IconCheck } from "./icons";

export function Footer() {
  return (
    <footer className={styles.footer}>
      <div className={styles.container}>
        {/* Main Footer Links */}
        <div className={styles.footerGrid}>
          <div>
            <div style={{ display: "flex", alignItems: "center", gap: "10px", marginBottom: "14px" }}>
              <IconLogo style={{ width: "24px", height: "24px", color: "var(--brand-primary)" }} aria-hidden="true" />
              <span style={{ fontWeight: 700, fontSize: "1.1rem", color: "var(--text-primary)" }}>Converse</span>
            </div>
            <p style={{ fontSize: "0.875rem", color: "var(--text-secondary)", lineHeight: 1.6, maxWidth: "380px", marginBottom: "16px" }}>
              A real-time bidirectional ASL and English communication engine delivering equal access through
              open-source edge machine learning and accessible web engineering.
            </p>
            <div style={{ display: "flex", alignItems: "center", gap: "6px", fontSize: "0.78rem", color: "var(--status-ok-text)", fontWeight: 600 }}>
              <IconCheck style={{ width: "14px", height: "14px" }} aria-hidden="true" />
              <span>WCAG AAA Compliant Contrast & Focus Navigation</span>
            </div>
          </div>

          <div>
            <h4 className={styles.footerColTitle}>Architecture</h4>
            <ul className={styles.footerColLinks}>
              <li><a href="#architecture" className={styles.footerLink}>Latency Budget</a></li>
              <li><a href="#benchmarks" className={styles.footerLink}>Model Benchmarks</a></li>
              <li><a href="#demo" className={styles.footerLink}>Pipeline Simulation</a></li>
              <li><a href="#features" className={styles.footerLink}>Privacy Invariants</a></li>
            </ul>
          </div>

          <div>
            <h4 className={styles.footerColTitle}>Engineering</h4>
            <ul className={styles.footerColLinks}>
              <li><span className={styles.footerLink}>@converse/contracts</span></li>
              <li><span className={styles.footerLink}>Next.js 16 Client</span></li>
              <li><span className={styles.footerLink}>Express Real-time API</span></li>
              <li><span className={styles.footerLink}>MediaPipe 3D Tasks</span></li>
            </ul>
          </div>
        </div>

        {/* Bottom Bar */}
        <div className={styles.footerBottom}>
          <span>Converse Open Source Communication Platform</span>
          <span>Zero Emoji Standard Enforced | Strictly Typed Monorepo</span>
        </div>
      </div>
    </footer>
  );
}
