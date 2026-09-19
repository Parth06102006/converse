"use client";

import { useState } from "react";
import styles from "../../app/marketing.module.css";
import { IconLogo, IconAccessibility, IconArrowRight } from "./icons";

interface NavbarProps {
  onFontScaleChange?: (scale: "normal" | "large" | "xlarge") => void;
  onHighContrastToggle?: (enabled: boolean) => void;
}

export function Navbar({ onFontScaleChange, onHighContrastToggle }: NavbarProps) {
  const [activeFont, setActiveFont] = useState<"normal" | "large" | "xlarge">("normal");
  const [highContrast, setHighContrast] = useState<boolean>(false);

  const handleFontChange = (scale: "normal" | "large" | "xlarge") => {
    setActiveFont(scale);
    onFontScaleChange?.(scale);
    if (typeof document !== "undefined") {
      document.documentElement.classList.remove("font-scale-large", "font-scale-xlarge");
      if (scale === "large") document.documentElement.classList.add("font-scale-large");
      if (scale === "xlarge") document.documentElement.classList.add("font-scale-xlarge");
    }
  };

  const handleContrastToggle = () => {
    const next = !highContrast;
    setHighContrast(next);
    onHighContrastToggle?.(next);
    if (typeof document !== "undefined") {
      if (next) {
        document.documentElement.classList.add("high-contrast");
      } else {
        document.documentElement.classList.remove("high-contrast");
      }
    }
  };

  return (
    <header className={styles.header}>
      <div className={`${styles.container} ${styles.headerInner}`}>
        <div style={{ display: "flex", alignItems: "center", gap: "16px" }}>
          <a href="#" className={styles.brand} aria-label="Converse Home">
            <IconLogo className={styles.brandIcon} aria-hidden="true" />
            <span>Converse</span>
          </a>
        </div>

        {/* Accessibility Preferences in Navbar */}
        <div
          role="region"
          aria-label="Accessibility Preferences"
          style={{ display: "flex", alignItems: "center", gap: "14px", flexWrap: "wrap" }}
        >
          <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
            <IconAccessibility
              style={{ width: "18px", height: "18px", color: "var(--brand-primary)" }}
              aria-hidden="true"
            />
            <span style={{ fontSize: "0.8rem", color: "var(--text-secondary)", fontWeight: 600 }}>
              Text:
            </span>
            <div className={styles.fontScaleButtonGroup} role="group" aria-label="Font Sizing">
              <button
                type="button"
                className={`${styles.fontBtn} ${activeFont === "normal" ? styles.fontBtnActive : ""}`}
                onClick={() => handleFontChange("normal")}
                aria-pressed={activeFont === "normal"}
              >
                Normal
              </button>
              <button
                type="button"
                className={`${styles.fontBtn} ${activeFont === "large" ? styles.fontBtnActive : ""}`}
                onClick={() => handleFontChange("large")}
                aria-pressed={activeFont === "large"}
              >
                Large
              </button>
              <button
                type="button"
                className={`${styles.fontBtn} ${activeFont === "xlarge" ? styles.fontBtnActive : ""}`}
                onClick={() => handleFontChange("xlarge")}
                aria-pressed={activeFont === "xlarge"}
              >
                X-Large
              </button>
            </div>
          </div>

          <button
            type="button"
            className={styles.btnSecondary}
            style={{
              padding: "6px 12px",
              fontSize: "0.8rem",
              backgroundColor: highContrast ? "var(--text-primary)" : "var(--bg-card)",
              color: highContrast ? "#ffffff" : "var(--text-primary)",
            }}
            onClick={handleContrastToggle}
            aria-pressed={highContrast}
          >
            <span>{highContrast ? "High Contrast: ON" : "High Contrast"}</span>
          </button>
        </div>

        <div>
          <a href="#demo" className={styles.btnPrimary} style={{ padding: "8px 16px", fontSize: "0.85rem" }}>
            <span>Launch Demo</span>
            <IconArrowRight style={{ width: "14px", height: "14px" }} aria-hidden="true" />
          </a>
        </div>
      </div>
    </header>
  );
}
