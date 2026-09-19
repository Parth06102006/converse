"use client";

import styles from "../../app/marketing.module.css";
import {
  IconShieldCheck,
  IconGauge,
  IconCamera,
  IconLayers,
  IconAudioWave,
  IconCpu,
} from "./icons";

const FEATURES = [
  {
    icon: IconShieldCheck,
    title: "Zero Video Cloud Storage",
    description:
      "All video frames are processed in volatile memory. 3D keypoint normalization occurs client-side, guaranteeing that your camera feed never leaves your personal device.",
  },
  {
    icon: IconGauge,
    title: "Sub-500ms Glass-to-Ear Response",
    description:
      "A causal, streaming-first architecture processes video frames and audio packets in continuous sliding windows, enabling natural conversational interruptions and fast turn-taking.",
  },
  {
    icon: IconCamera,
    title: "VoIP Companion Extension",
    description:
      "A Chrome Manifest V3 companion injects real-time captions and an avatar HUD directly into Google Meet and Zoom, routing synthesized speech into the virtual microphone.",
  },
  {
    icon: IconLayers,
    title: "Anatomically Rigged 3D Avatar",
    description:
      "Humanoid armatures driven by Quaternion SLERP transitions and ARKit-compatible facial blendshapes accurately replicate Non-Manual Markers like eyebrow deflection and head tilts.",
  },
  {
    icon: IconCpu,
    title: "Cascading Sentence Reconstruction",
    description:
      "Temporal debouncing buffers absorb sustained signs, while a linguistic translation engine restores missing copulas ('is', 'are') and determiners ('the', 'a') from ASL glosses.",
  },
  {
    icon: IconAudioWave,
    title: "Gapless Web Audio Playback",
    description:
      "A dynamic 40ms-80ms jitter buffer schedules streaming PCM audio chunks via AudioBufferSourceNode, preventing buffer underruns and clicks during continuous speech playback.",
  },
];

export function FeaturesGrid() {
  return (
    <section id="features" className={styles.section} aria-labelledby="features-title">
      <div className={styles.container}>
        <div className={styles.sectionHeader}>
          <span className={styles.sectionPretitle}>Core Engineering Pillars</span>
          <h2 id="features-title" className={styles.sectionTitle}>
            Architected for True Conversational Parity
          </h2>
          <p className={styles.sectionSubtitle}>
            Engineered from first principles to overcome the linguistic and acoustic
            asymmetries between sign and spoken languages.
          </p>
        </div>

        <div className={styles.featuresGrid}>
          {FEATURES.map((feat, idx) => {
            const Icon = feat.icon;
            return (
              <div key={idx} className={styles.featureCard}>
                <div className={styles.featureIconWrapper}>
                  <Icon style={{ width: "24px", height: "24px" }} aria-hidden="true" />
                </div>
                <h3 className={styles.featureTitle}>{feat.title}</h3>
                <p className={styles.featureDesc}>{feat.description}</p>
              </div>
            );
          })}
        </div>
      </div>
    </section>
  );
}
