"use client";

import { Navbar } from "../components/marketing/navbar";
import { Hero } from "../components/marketing/hero";
import { InteractiveDemo } from "../components/marketing/interactive-demo";
import { FeaturesGrid } from "../components/marketing/features-grid";
import { ArchitectureBreakdown } from "../components/marketing/architecture-breakdown";
import { Footer } from "../components/marketing/footer";

export default function Home() {
  return (
    <div style={{ minHeight: "100vh", display: "flex", flexDirection: "column" }}>
      <Navbar />
      <main id="main-content" style={{ flex: 1 }}>
        <Hero />
        <InteractiveDemo />
        <FeaturesGrid />
        <ArchitectureBreakdown />
      </main>
      <Footer />
    </div>
  );
}
