import type { SignDetection } from "@converse/contracts";

export interface GlossStabilizerOptions {
  /** Minimum confidence threshold to consider a sign valid (0.0 to 1.0). Default: 0.65 */
  minConfidence?: number;
  /** Maximum time gap (in ms) to merge identical consecutive signs. Default: 350ms */
  debounceWindowMs?: number;
  /** Inactivity pause duration (in ms) that triggers a sentence boundary flush. Default: 800ms */
  boundaryPauseMs?: number;
}

export interface StabilizedGlossEvent {
  gloss: string;
  confidence: number;
  startTimeMs: number;
  endTimeMs: number;
}

export class GlossStabilizer {
  private readonly minConfidence: number;
  private readonly debounceWindowMs: number;
  private readonly boundaryPauseMs: number;

  private buffer: string[] = [];
  private lastDetection: SignDetection | null = null;
  private lastEmitTimeMs = 0;
  private boundaryTimer: ReturnType<typeof setTimeout> | null = null;

  public onStabilizedGloss?: (event: StabilizedGlossEvent) => void;
  public onBoundaryDetected?: (glosses: string[]) => void;

  constructor(options: GlossStabilizerOptions = {}) {
    this.minConfidence = options.minConfidence ?? 0.65;
    this.debounceWindowMs = options.debounceWindowMs ?? 350;
    this.boundaryPauseMs = options.boundaryPauseMs ?? 800;
  }

  /**
   * Ingest a single candidate SignDetection from the vision model.
   * Handles confidence filtering, temporal sliding-window debouncing, and boundary spotting.
   */
  public processDetection(detection: SignDetection): {
    accepted: boolean;
    isDuplicate: boolean;
    stabilizedGloss?: string;
  } {
    // 1. Filter low-confidence noise
    if (detection.confidence < this.minConfidence) {
      return { accepted: false, isDuplicate: false };
    }

    const normalizedGloss = detection.gloss.trim().toUpperCase();
    if (!normalizedGloss) {
      return { accepted: false, isDuplicate: false };
    }

    // Reset boundary pause timer on any high-confidence sign detection
    this.clearBoundaryTimer();

    // 2. Sliding window debouncing for identical consecutive signs
    if (
      this.lastDetection &&
      this.lastDetection.gloss === normalizedGloss &&
      detection.startTimeMs - this.lastDetection.endTimeMs <= this.debounceWindowMs
    ) {
      // Sustained sign detection: extend boundary window and update peak confidence
      this.lastDetection.endTimeMs = Math.max(this.lastDetection.endTimeMs, detection.endTimeMs);
      this.lastDetection.confidence = Math.max(this.lastDetection.confidence, detection.confidence);
      this.lastEmitTimeMs = detection.endTimeMs;

      this.scheduleBoundaryCheck();
      return { accepted: true, isDuplicate: true };
    }

    // 3. New unique sign transition
    this.lastDetection = {
      ...detection,
      gloss: normalizedGloss,
    };
    this.lastEmitTimeMs = detection.endTimeMs;
    this.buffer.push(normalizedGloss);

    const event: StabilizedGlossEvent = {
      gloss: normalizedGloss,
      confidence: detection.confidence,
      startTimeMs: detection.startTimeMs,
      endTimeMs: detection.endTimeMs,
    };

    if (this.onStabilizedGloss) {
      this.onStabilizedGloss(event);
    }

    // 4. Arm the boundary rest detection timer
    this.scheduleBoundaryCheck();

    return {
      accepted: true,
      isDuplicate: false,
      stabilizedGloss: normalizedGloss,
    };
  }

  /**
   * Explicitly check if a boundary should be flushed based on a provided timestamp.
   * Useful in simulated or offline batch evaluation.
   */
  public checkBoundaryAt(currentTimeMs: number): boolean {
    if (
      this.buffer.length > 0 &&
      this.lastEmitTimeMs > 0 &&
      currentTimeMs - this.lastEmitTimeMs >= this.boundaryPauseMs
    ) {
      this.flushBoundary();
      return true;
    }
    return false;
  }

  /**
   * Flush the current stabilized buffer and signal sentence boundary completion.
   */
  public flushBoundary(): string[] {
    this.clearBoundaryTimer();
    const flushedGlosses = [...this.buffer];
    this.buffer = [];
    this.lastDetection = null;
    this.lastEmitTimeMs = 0;

    if (flushedGlosses.length > 0 && this.onBoundaryDetected) {
      this.onBoundaryDetected(flushedGlosses);
    }

    return flushedGlosses;
  }

  /**
   * Retrieve an immutable snapshot of the current active gloss buffer.
   */
  public getBuffer(): readonly string[] {
    return [...this.buffer];
  }

  /**
   * Reset all internal state and cancel pending timers.
   */
  public reset(): void {
    this.clearBoundaryTimer();
    this.buffer = [];
    this.lastDetection = null;
    this.lastEmitTimeMs = 0;
  }

  private scheduleBoundaryCheck(): void {
    if (typeof setTimeout === "undefined") return;
    this.boundaryTimer = setTimeout(() => {
      if (this.buffer.length > 0) {
        this.flushBoundary();
      }
    }, this.boundaryPauseMs);
  }

  private clearBoundaryTimer(): void {
    if (this.boundaryTimer) {
      clearTimeout(this.boundaryTimer);
      this.boundaryTimer = null;
    }
  }
}
