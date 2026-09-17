import { Button } from "@converse/ui/button";
import { Card } from "@converse/ui/card";
import { Code } from "@converse/ui/code";
import type {
  HealthCheckResponse,
  RealtimeDirection,
} from "@converse/contracts";
import styles from "./page.module.css";

const systemPipelines: Array<{
  direction: RealtimeDirection;
  title: string;
  flow: string;
  description: string;
}> = [
  {
    direction: "sign_to_speech",
    title: "Sign -> Speech Pipeline",
    flow: "Camera -> ASL Vision -> Representation -> Translation -> English Text -> TTS -> Audio",
    description:
      "Captures continuous sign gestures, extracts hand/body landmarks, translates to English syntax, and synthesizes audio speech.",
  },
  {
    direction: "speech_to_sign",
    title: "Speech -> Sign Pipeline",
    flow: "Audio -> ASR -> English Text -> Translation -> ASL Representation -> Sign Renderer",
    description:
      "Captures spoken voice, performs automatic speech recognition, transforms English grammar into ASL glosses, and drives sign visualization.",
  },
];

const mockHealth: HealthCheckResponse = {
  status: "ok",
  service: "@converse/api",
  version: "0.1.0",
  timestamp: new Date().toISOString(),
};

export default function Home() {
  return (
    <div className={styles.page}>
      <main className={styles.main}>
        <header
          style={{ display: "flex", flexDirection: "column", gap: "8px" }}
        >
          <h1 style={{ fontSize: "2rem", fontWeight: 700, margin: 0 }}>
            Converse
          </h1>
          <p style={{ margin: 0, opacity: 0.8 }}>
            Real-time bidirectional ASL and English communication platform
          </p>
          <div
            style={{
              display: "flex",
              alignItems: "center",
              gap: "8px",
              marginTop: "4px",
            }}
          >
            <span style={{ fontSize: "0.875rem", opacity: 0.7 }}>
              API Status:
            </span>
            <Code className={styles.secondary}>
              {mockHealth.status.toUpperCase()}
            </Code>
            <span style={{ fontSize: "0.875rem", opacity: 0.7 }}>
              via @converse/contracts
            </span>
          </div>
        </header>

        <section
          style={{
            display: "grid",
            gridTemplateColumns: "repeat(auto-fit, minmax(320px, 1fr))",
            gap: "16px",
          }}
        >
          {systemPipelines.map((pipeline) => (
            <Card
              key={pipeline.direction}
              title={pipeline.title}
              href="/#pipeline"
            >
              <span
                style={{
                  display: "block",
                  marginBottom: "8px",
                  fontSize: "0.85rem",
                  opacity: 0.7,
                }}
              >
                {pipeline.flow}
              </span>
              <span>{pipeline.description}</span>
            </Card>
          ))}
        </section>

        <section
          style={{ display: "flex", flexDirection: "column", gap: "12px" }}
        >
          <h2 style={{ fontSize: "1.25rem", margin: 0 }}>
            Monorepo Architecture
          </h2>
          <ol>
            <li>
              Shared Contracts: <Code>packages/contracts</Code> (TypeScript
              types for Vision, ASR, Translation, TTS)
            </li>
            <li>
              Backend API: <Code>services/api</Code> (Express on port 4000 wired
              to Turbo)
            </li>
            <li>
              Frontend: <Code>apps/web</Code> (Next.js 16 with Turbopack)
            </li>
            <li>
              Design System: <Code>packages/ui</Code> (Shared React components)
            </li>
          </ol>
        </section>

        <div className={styles.ctas}>
          <Button appName="web" className={styles.primary}>
            Launch Session
          </Button>
        </div>
      </main>

      <footer className={styles.footer}>
        <span style={{ fontSize: "0.875rem", opacity: 0.6 }}>
          Converse ASL Communication Platform
        </span>
      </footer>
    </div>
  );
}
