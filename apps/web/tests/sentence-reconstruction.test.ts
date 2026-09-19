import { describe, it } from "node:test";
import assert from "node:assert/strict";
import { GlossStabilizer } from "../lib/gloss-stabilizer.ts";
import {
  reconstructSentence,
  generatePartialPreview,
} from "../lib/sentence-reconstructor.ts";
import type { SignDetection } from "@converse/contracts";

describe("GlossStabilizer", () => {
  it("should debounce repeated detections of the same sign within the temporal window", () => {
    const stabilizer = new GlossStabilizer({
      minConfidence: 0.65,
      debounceWindowMs: 350,
      boundaryPauseMs: 800,
    });

    const emittedGlosses: string[] = [];
    stabilizer.onStabilizedGloss = (event) => {
      emittedGlosses.push(event.gloss);
    };

    // Simulate 15 consecutive detections of "STORE" over 500ms (typical 30 FPS camera feed)
    for (let i = 0; i < 15; i++) {
      const detection: SignDetection = {
        gloss: "STORE",
        confidence: 0.75 + (i % 3) * 0.05,
        startTimeMs: i * 33,
        endTimeMs: (i + 1) * 33,
      };
      const result = stabilizer.processDetection(detection);
      assert.equal(result.accepted, true);
      if (i === 0) {
        assert.equal(result.isDuplicate, false);
        assert.equal(result.stabilizedGloss, "STORE");
      } else {
        assert.equal(result.isDuplicate, true);
      }
    }

    // Only 1 unique stabilized gloss should have been emitted
    assert.equal(emittedGlosses.length, 1);
    assert.equal(emittedGlosses[0], "STORE");
    assert.deepEqual(stabilizer.getBuffer(), ["STORE"]);
  });

  it("should filter out low-confidence detections below the minimum threshold", () => {
    const stabilizer = new GlossStabilizer({ minConfidence: 0.65 });

    const lowConfDetection: SignDetection = {
      gloss: "RANDOM_NOISE",
      confidence: 0.42,
      startTimeMs: 100,
      endTimeMs: 133,
    };

    const result = stabilizer.processDetection(lowConfDetection);
    assert.equal(result.accepted, false);
    assert.equal(stabilizer.getBuffer().length, 0);
  });

  it("should retain distinct sequential signs in proper chronological order", () => {
    const stabilizer = new GlossStabilizer({ debounceWindowMs: 350 });

    const sequence = ["YESTERDAY", "STORE", "ME", "GO"];
    let time = 0;

    for (const gloss of sequence) {
      // 3 detections per sign
      for (let f = 0; f < 3; f++) {
        stabilizer.processDetection({
          gloss,
          confidence: 0.88,
          startTimeMs: time,
          endTimeMs: time + 33,
        });
        time += 33;
      }
      // Small inter-sign transition gap of 100ms
      time += 100;
    }

    assert.deepEqual(stabilizer.getBuffer(), sequence);
  });

  it("should trigger boundary flush upon detecting an inactivity pause >= 800ms", () => {
    const stabilizer = new GlossStabilizer({
      debounceWindowMs: 350,
      boundaryPauseMs: 800,
    });

    let flushedBoundary: string[] | null = null;
    stabilizer.onBoundaryDetected = (glosses) => {
      flushedBoundary = glosses;
    };

    stabilizer.processDetection({
      gloss: "HELLO",
      confidence: 0.9,
      startTimeMs: 0,
      endTimeMs: 300,
    });

    stabilizer.processDetection({
      gloss: "NICE",
      confidence: 0.92,
      startTimeMs: 400,
      endTimeMs: 700,
    });

    assert.equal(flushedBoundary, null);

    // Simulate clock advancing past the 800ms rest boundary threshold (700 + 850 = 1550ms)
    const boundaryTriggered = stabilizer.checkBoundaryAt(1550);
    assert.equal(boundaryTriggered, true);
    assert.deepEqual(flushedBoundary, ["HELLO", "NICE"]);
    assert.equal(stabilizer.getBuffer().length, 0);
  });
});

describe("SentenceReconstructor", () => {
  describe("Canonical Idioms & Greetings", () => {
    it("should reconstruct 'HELLO' into 'Hello!'", () => {
      const res = reconstructSentence(["HELLO"]);
      assert.equal(res.englishText, "Hello!");
      assert.equal(res.confidence >= 0.9, true);
    });

    it("should reconstruct 'HELLO NICE MEET' into 'Hello, nice to meet you.'", () => {
      const res = reconstructSentence(["HELLO", "NICE", "MEET"]);
      assert.equal(res.englishText, "Hello, nice to meet you.");
    });

    it("should reconstruct 'HELLO NICE MEET YOU' into 'Hello, nice to meet you.'", () => {
      const res = reconstructSentence(["HELLO", "NICE", "MEET", "YOU"]);
      assert.equal(res.englishText, "Hello, nice to meet you.");
    });

    it("should reconstruct 'THANK-YOU HELP' into 'Thank you for your help.'", () => {
      const res = reconstructSentence(["THANK-YOU", "HELP"]);
      assert.equal(res.englishText, "Thank you for your help.");
    });

    it("should reconstruct 'SEE YOU LATER' into 'See you later.'", () => {
      const res = reconstructSentence(["SEE", "YOU", "LATER"]);
      assert.equal(res.englishText, "See you later.");
    });

    it("should reconstruct 'GOOD MORNING' into 'Good morning.'", () => {
      const res = reconstructSentence(["GOOD", "MORNING"]);
      assert.equal(res.englishText, "Good morning.");
    });
  });

  describe("Wh-Questions and Interrogative Inversion", () => {
    it("should invert ASL 'NAME YOU WHAT' into 'What is your name?'", () => {
      const res = reconstructSentence(["NAME", "YOU", "WHAT"]);
      assert.equal(res.englishText, "What is your name?");
    });

    it("should invert ASL 'BATHROOM WHERE' into 'Where is the bathroom?'", () => {
      const res = reconstructSentence(["BATHROOM", "WHERE"]);
      assert.equal(res.englishText, "Where is the bathroom?");
    });

    it("should reconstruct 'STORE WHERE' into 'Where is the store?'", () => {
      const res = reconstructSentence(["STORE", "WHERE"]);
      assert.equal(res.englishText, "Where is the store?");
    });

    it("should reconstruct 'TIME WHAT' into 'What time is it?'", () => {
      const res = reconstructSentence(["TIME", "WHAT"]);
      assert.equal(res.englishText, "What time is it?");
    });

    it("should reconstruct 'HOW YOU' into 'How are you?'", () => {
      const res = reconstructSentence(["HOW", "YOU"]);
      assert.equal(res.englishText, "How are you?");
    });

    it("should reconstruct 'WHEN YOU ARRIVE' into 'When will you arrive?'", () => {
      const res = reconstructSentence(["WHEN", "YOU", "ARRIVE"]);
      assert.equal(res.englishText, "When will you arrive?");
    });
  });

  describe("Time-Topic-Comment Tense Shifting", () => {
    it("should inflect past tense verb for 'YESTERDAY ME STORE GO'", () => {
      const res = reconstructSentence(["YESTERDAY", "ME", "STORE", "GO"]);
      assert.equal(res.englishText, "I went to the store yesterday.");
    });

    it("should inflect future tense verb for 'TOMORROW WE MEET'", () => {
      const res = reconstructSentence(["TOMORROW", "WE", "MEET"]);
      assert.equal(res.englishText, "We will meet tomorrow.");
    });

    it("should handle past tense with negation: 'YESTERDAY HE NOT COME'", () => {
      const res = reconstructSentence(["YESTERDAY", "HE", "NOT", "COME"]);
      assert.equal(res.englishText, "He did not come yesterday.");
    });
  });

  describe("Copula & Adjective Insertion", () => {
    it("should insert copula 'am' for 'ME HUNGRY'", () => {
      const res = reconstructSentence(["ME", "HUNGRY"]);
      assert.equal(res.englishText, "I am hungry.");
    });

    it("should insert copula 'is' for 'HE MY FRIEND'", () => {
      const res = reconstructSentence(["HE", "MY", "FRIEND"]);
      assert.equal(res.englishText, "He is my friend.");
    });

    it("should handle negation with copula: 'ME NOT TIRED'", () => {
      const res = reconstructSentence(["ME", "NOT", "TIRED"]);
      assert.equal(res.englishText, "I am not tired.");
    });

    it("should insert copula 'are' for 'WE READY'", () => {
      const res = reconstructSentence(["WE", "READY"]);
      assert.equal(res.englishText, "We are ready.");
    });
  });

  describe("Partial Streaming Preview", () => {
    it("should generate instant streaming preview with ellipsis", () => {
      const preview = generatePartialPreview(["STORE", "GO"]);
      assert.equal(preview, "the store go...");
    });

    it("should return canonical preview if available", () => {
      const preview = generatePartialPreview(["HELLO", "NICE", "MEET"]);
      assert.equal(preview, "Hello, nice to meet you.");
    });
  });

  describe("Performance & Latency", () => {
    it("should execute sentence reconstruction in under 15ms", () => {
      const res = reconstructSentence(["YESTERDAY", "ME", "STORE", "GO"]);
      assert.equal(res.latencyMs < 15, true);
    });
  });

    describe("Additional Sentence Reconstruction Cases", () => {
    it("should reconstruct 'I LOVE YOU' into 'I love you.'", () => {
      const res = reconstructSentence(["I", "LOVE", "YOU"]);
      assert.equal(res.englishText, "I love you.");
    });

    it("should reconstruct 'YOU HELP ME' into 'You help me.'", () => {
      const res = reconstructSentence(["YOU", "HELP", "ME"]);
      assert.equal(res.englishText, "You help me.");
    });

    it("should reconstruct 'SHE MY SISTER' into 'She is my sister.'", () => {
      const res = reconstructSentence(["SHE", "MY", "SISTER"]);
      assert.equal(res.englishText, "She is my sister.");
    });

    it("should reconstruct 'HE WORK' into 'He works.'", () => {
      const res = reconstructSentence(["HE", "WORK"]);
      assert.equal(res.englishText, "He works.");
    });

    it("should reconstruct 'I WANT WATER' into 'I want water.'", () => {
      const res = reconstructSentence(["I", "WANT", "WATER"]);
      assert.equal(res.englishText, "I want water.");
    });

    it("should reconstruct 'I NEED HELP' into 'I need help.'", () => {
      const res = reconstructSentence(["I", "NEED", "HELP"]);
      assert.equal(res.englishText, "I need help.");
    });

    it("should reconstruct 'YOU WANT COFFEE' into 'You want coffee.'", () => {
      const res = reconstructSentence(["YOU", "WANT", "COFFEE"]);
      assert.equal(res.englishText, "You want coffee.");
    });

    it("should reconstruct 'WE GO STORE' into 'We go to the store.'", () => {
      const res = reconstructSentence(["WE", "GO", "STORE"]);
      assert.equal(res.englishText, "We go to the store.");
    });

    it("should reconstruct 'I GO SCHOOL TODAY' into 'I go to school today.'", () => {
      const res = reconstructSentence(["I", "GO", "SCHOOL", "TODAY"]);
      assert.equal(res.englishText, "I go to school today.");
    });

    it("should reconstruct 'YESTERDAY I GO SCHOOL' into 'I went to school yesterday.'", () => {
      const res = reconstructSentence(["YESTERDAY", "I", "GO", "SCHOOL"]);
      assert.equal(res.englishText, "I went to school yesterday.");
    });

    it("should reconstruct 'TOMORROW I GO STORE' into 'I will go to the store tomorrow.'", () => {
      const res = reconstructSentence(["TOMORROW", "I", "GO", "STORE"]);
      assert.equal(res.englishText, "I will go to the store tomorrow.");
    });

    it("should reconstruct 'TODAY I WORK' into 'I work today.'", () => {
      const res = reconstructSentence(["TODAY", "I", "WORK"]);
      assert.equal(res.englishText, "I work today.");
    });

    it("should reconstruct 'YESTERDAY SHE WORK' into 'She worked yesterday.'", () => {
      const res = reconstructSentence(["YESTERDAY", "SHE", "WORK"]);
      assert.equal(res.englishText, "She worked yesterday.");
    });

    it("should reconstruct 'TOMORROW WE MEET' into 'We will meet tomorrow.'", () => {
      const res = reconstructSentence(["TOMORROW", "WE", "MEET"]);
      assert.equal(res.englishText, "We will meet tomorrow.");
    });

    it("should reconstruct 'YOU WHERE LIVE' into 'Where do you live?'", () => {
      const res = reconstructSentence(["YOU", "WHERE", "LIVE"]);
      assert.equal(res.englishText, "Where do you live?");
    });

    it("should reconstruct 'YOU WHAT WANT' into 'What do you want?'", () => {
      const res = reconstructSentence(["YOU", "WHAT", "WANT"]);
      assert.equal(res.englishText, "What do you want?");
    });

    it("should reconstruct 'YOU WHERE GO' into 'Where are you going?'", () => {
      const res = reconstructSentence(["YOU", "WHERE", "GO"]);
      assert.equal(res.englishText, "Where are you going?");
    });

    it("should reconstruct 'YOU WHEN ARRIVE' into 'When will you arrive?'", () => {
      const res = reconstructSentence(["YOU", "WHEN", "ARRIVE"]);
      assert.equal(res.englishText, "When will you arrive?");
    });

    it("should reconstruct 'WHO YOUR FRIEND' into 'Who is your friend?'", () => {
      const res = reconstructSentence(["WHO", "YOUR", "FRIEND"]);
      assert.equal(res.englishText, "Who is your friend?");
    });

    it("should reconstruct 'WHY YOU LATE' into 'Why are you late?'", () => {
      const res = reconstructSentence(["WHY", "YOU", "LATE"]);
      assert.equal(res.englishText, "Why are you late?");
    });

    it("should reconstruct 'ME NOT KNOW' into 'I do not know.'", () => {
      const res = reconstructSentence(["ME", "NOT", "KNOW"]);
      assert.equal(res.englishText, "I do not know.");
    });

    it("should reconstruct 'I NOT WANT COFFEE' into 'I do not want coffee.'", () => {
      const res = reconstructSentence(["I", "NOT", "WANT", "COFFEE"]);
      assert.equal(res.englishText, "I do not want coffee.");
    });

    it("should reconstruct 'HE NOT COME' into 'He did not come.'", () => {
      const res = reconstructSentence(["HE", "NOT", "COME"]);
      assert.equal(res.englishText, "He did not come.");
    });

    it("should reconstruct 'WE NOT READY' into 'We are not ready.'", () => {
      const res = reconstructSentence(["WE", "NOT", "READY"]);
      assert.equal(res.englishText, "We are not ready.");
    });

    it("should reconstruct 'I LIKE MUSIC' into 'I like music.'", () => {
      const res = reconstructSentence(["I", "LIKE", "MUSIC"]);
      assert.equal(res.englishText, "I like music.");
    });

    it("should reconstruct 'SHE LIKE DANCE' into 'She likes dance.'", () => {
      const res = reconstructSentence(["SHE", "LIKE", "DANCE"]);
      assert.equal(res.englishText, "She likes dance.");
    });

    it("should reconstruct 'HE HAVE CAR' into 'He has a car.'", () => {
      const res = reconstructSentence(["HE", "HAVE", "CAR"]);
      assert.equal(res.englishText, "He has a car.");
    });

    it("should reconstruct 'I HAVE QUESTION' into 'I have a question.'", () => {
      const res = reconstructSentence(["I", "HAVE", "QUESTION"]);
      assert.equal(res.englishText, "I have a question.");
    });

    it("should reconstruct 'MY NAME KANISHKA' into 'My name is Kanishka.'", () => {
      const res = reconstructSentence(["MY", "NAME", "KANISHKA"]);
      assert.equal(res.englishText, "My name is Kanishka.");
    });

    it("should reconstruct 'MY FRIEND NAME JOHN' into 'My friend's name is John.'", () => {
      const res = reconstructSentence(["MY", "FRIEND", "NAME", "JOHN"]);
      assert.equal(res.englishText, "My friend's name is John.");
    });

    it("should reconstruct 'GOOD NIGHT' into 'Good night.'", () => {
      const res = reconstructSentence(["GOOD", "NIGHT"]);
      assert.equal(res.englishText, "Good night.");
    });

    it("should reconstruct 'THANK-YOU VERY MUCH' into 'Thank you very much.'", () => {
      const res = reconstructSentence(["THANK-YOU", "VERY", "MUCH"]);
      assert.equal(res.englishText, "Thank you very much.");
    });

    it("should reconstruct 'SORRY I LATE' into 'Sorry, I am late.'", () => {
      const res = reconstructSentence(["SORRY", "I", "LATE"]);
      assert.equal(res.englishText, "Sorry, I am late.");
    });

    it("should reconstruct 'PLEASE HELP ME' into 'Please help me.'", () => {
      const res = reconstructSentence(["PLEASE", "HELP", "ME"]);
      assert.equal(res.englishText, "Please help me.");
    });
  });
});
