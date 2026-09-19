import type { SignToTextResponse } from "@converse/contracts";

export interface SentenceReconstructionOptions {
  sessionId?: string;
  context?: string;
  confidence?: number;
}

// Irregular verbs mapping for past, present, and future tenses
const VERB_CONJUGATIONS: Record<
  string,
  { past: string; present: string; continuous: string; base: string }
> = {
  GO: { base: "go", present: "goes", past: "went", continuous: "going" },
  SEE: { base: "see", present: "sees", past: "saw", continuous: "seeing" },
  MEET: { base: "meet", present: "meets", past: "met", continuous: "meeting" },
  EAT: { base: "eat", present: "eats", past: "ate", continuous: "eating" },
  DRINK: { base: "drink", present: "drinks", past: "drank", continuous: "drinking" },
  BUY: { base: "buy", present: "buys", past: "bought", continuous: "buying" },
  WANT: { base: "want", present: "wants", past: "wanted", continuous: "wanting" },
  NEED: { base: "need", present: "needs", past: "needed", continuous: "needing" },
  HELP: { base: "help", present: "helps", past: "helped", continuous: "helping" },
  LIKE: { base: "like", present: "likes", past: "liked", continuous: "liking" },
  LOVE: { base: "love", present: "loves", past: "loved", continuous: "loving" },
  LIVE: { base: "live", present: "lives", past: "lived", continuous: "living" },
  HAVE: { base: "have", present: "has", past: "had", continuous: "having" },
  COME: { base: "come", present: "comes", past: "came", continuous: "coming" },
  LEAVE: { base: "leave", present: "leaves", past: "left", continuous: "leaving" },
  WORK: { base: "work", present: "works", past: "worked", continuous: "working" },
  LEARN: { base: "learn", present: "learns", past: "learned", continuous: "learning" },
  UNDERSTAND: { base: "understand", present: "understands", past: "understood", continuous: "understanding" },
  KNOW: { base: "know", present: "knows", past: "knew", continuous: "knowing" },
  CALL: { base: "call", present: "calls", past: "called", continuous: "calling" },
  WAIT: { base: "wait", present: "waits", past: "waited", continuous: "waiting" },
  DANCE: { base: "dance", present: "dances", past: "danced", continuous: "dancing" },
};

// Canonical idioms and fixed conversational phrases
const CANONICAL_PATTERNS: Array<{
  pattern: string[];
  template: string;
}> = [
  { pattern: ["HELLO"], template: "Hello!" },
  { pattern: ["HI"], template: "Hi there!" },
  { pattern: ["HELLO", "NICE", "MEET"], template: "Hello, nice to meet you." },
  { pattern: ["HELLO", "NICE", "MEET", "YOU"], template: "Hello, nice to meet you." },
  { pattern: ["NICE", "MEET", "YOU"], template: "Nice to meet you." },
  { pattern: ["HOW", "YOU"], template: "How are you?" },
  { pattern: ["HOW", "ARE", "YOU"], template: "How are you?" },
  { pattern: ["THANK-YOU"], template: "Thank you!" },
  { pattern: ["THANK", "YOU"], template: "Thank you!" },
  { pattern: ["THANK-YOU", "HELP"], template: "Thank you for your help." },
  { pattern: ["THANK", "YOU", "HELP"], template: "Thank you for your help." },
  { pattern: ["THANK-YOU", "VERY", "MUCH"], template: "Thank you very much." },
  { pattern: ["THANK", "YOU", "VERY", "MUCH"], template: "Thank you very much." },
  { pattern: ["WELCOME"], template: "You are welcome." },
  { pattern: ["YOU", "WELCOME"], template: "You are welcome." },
  { pattern: ["GOOD", "MORNING"], template: "Good morning." },
  { pattern: ["GOOD", "AFTERNOON"], template: "Good afternoon." },
  { pattern: ["GOOD", "NIGHT"], template: "Good night." },
  { pattern: ["SEE", "YOU", "LATER"], template: "See you later." },
  { pattern: ["GOODBYE"], template: "Goodbye." },
  { pattern: ["BYE"], template: "Goodbye." },
  { pattern: ["PLEASE"], template: "Please." },
  { pattern: ["SORRY"], template: "I am sorry." },
  { pattern: ["SORRY", "I", "LATE"], template: "Sorry, I am late." },
  { pattern: ["SORRY", "ME", "LATE"], template: "Sorry, I am late." },
  { pattern: ["PLEASE", "HELP", "ME"], template: "Please help me." },
  { pattern: ["EXCUSE", "ME"], template: "Excuse me." },
  { pattern: ["YES"], template: "Yes." },
  { pattern: ["NO"], template: "No." },
];

// Institutions / Places that use zero-article ("to school", "to work", "home")
const ZERO_ARTICLE_LOCATIONS = new Set(["SCHOOL", "WORK", "HOME", "CLASS", "BED"]);

// Locations requiring "the" / "to the"
const STANDARD_LOCATIONS = new Set([
  "STORE",
  "BATHROOM",
  "RESTAURANT",
  "HOSPITAL",
  "LIBRARY",
  "AIRPORT",
  "OFFICE",
  "BANK",
  "DOCTOR",
  "PHARMACY",
]);

// Singular countable nouns that need an indefinite article ("a" / "an")
const COUNTABLE_NOUNS = new Set([
  "CAR",
  "QUESTION",
  "BOOK",
  "DOG",
  "CAT",
  "PROBLEM",
  "IDEA",
  "JOB",
  "HOUSE",
  "PHONE",
  "COMPUTER",
  "PEN",
]);

// Adjectives requiring copula ("am", "is", "are", "was", "were")
const ADJECTIVES = new Set([
  "HUNGRY",
  "THIRSTY",
  "TIRED",
  "HAPPY",
  "SAD",
  "READY",
  "FINE",
  "BUSY",
  "SICK",
  "COLD",
  "HOT",
  "LATE",
  "DEAF",
  "HEARING",
]);

// Question words
const QUESTION_WORDS = new Set(["WHAT", "WHERE", "WHEN", "WHY", "WHO", "HOW", "WHICH"]);

/**
 * Reconstruct a stabilized ASL gloss sequence into a grammatically coherent English sentence.
 */
export function reconstructSentence(
  glosses: string[],
  options: SentenceReconstructionOptions = {}
): SignToTextResponse {
  const startTime = performance.now();
  const normalized = glosses.map((g) => g.trim().toUpperCase()).filter(Boolean);

  if (normalized.length === 0) {
    return {
      englishText: "",
      confidence: 0,
      glosses: [],
      latencyMs: 0,
    };
  }

  // 1. Check Canonical Idioms / Patterns
  const canonicalMatch = findCanonicalMatch(normalized);
  if (canonicalMatch) {
    const elapsed = Math.max(1, Math.round(performance.now() - startTime));
    return {
      englishText: canonicalMatch,
      confidence: options.confidence ?? 0.96,
      glosses: normalized,
      latencyMs: elapsed,
    };
  }

  // 2. Check Possessive Name Introductions (e.g. "MY NAME KANISHKA", "MY FRIEND NAME JOHN")
  const nameMatch = handleNameIntroduction(normalized);
  if (nameMatch) {
    const elapsed = Math.max(1, Math.round(performance.now() - startTime));
    return {
      englishText: nameMatch,
      confidence: options.confidence ?? 0.94,
      glosses: normalized,
      latencyMs: elapsed,
    };
  }

  // 3. Synthesize English sentence via grammar rules
  const englishText = synthesizeGrammar(normalized);
  const elapsed = Math.max(1, Math.round(performance.now() - startTime));

  return {
    englishText,
    confidence: options.confidence ?? 0.88,
    glosses: normalized,
    latencyMs: elapsed,
  };
}

/**
 * Generates an instant, uncommitted preview for partial streaming UI feedback (<2ms).
 */
export function generatePartialPreview(glosses: string[]): string {
  const normalized = glosses.map((g) => g.trim().toUpperCase()).filter(Boolean);
  if (normalized.length === 0) return "";

  const canonical = findCanonicalMatch(normalized);
  if (canonical) return canonical;

  const previewWords = normalized.map((g) => {
    if (g === "ME") return "I";
    if (g === "YOU") return "you";
    if (g === "STORE") return "the store";
    if (g === "BATHROOM") return "the bathroom";
    if (g === "SCHOOL") return "school";
    return g.toLowerCase();
  });

  return `${previewWords.join(" ")}...`;
}

function findCanonicalMatch(glosses: string[]): string | null {
  for (const item of CANONICAL_PATTERNS) {
    if (
      item.pattern.length === glosses.length &&
      item.pattern.every((p, idx) => p === glosses[idx])
    ) {
      return item.template;
    }
  }
  return null;
}

function handleNameIntroduction(glosses: string[]): string | null {
  // "MY NAME [X]" -> "My name is [X]."
  if (glosses.length === 3 && glosses[0] === "MY" && glosses[1] === "NAME") {
    const name = capitalize(glosses[2]!.toLowerCase());
    return `My name is ${name}.`;
  }

  // "MY FRIEND NAME [X]" -> "My friend's name is [X]."
  if (
    glosses.length === 4 &&
    glosses[0] === "MY" &&
    glosses[1] === "FRIEND" &&
    glosses[2] === "NAME"
  ) {
    const name = capitalize(glosses[3]!.toLowerCase());
    return `My friend's name is ${name}.`;
  }

  return null;
}

function synthesizeGrammar(glosses: string[]): string {
  // Check for Wh-questions (ASL puts question word at end or start or middle)
  const questionWordIdx = glosses.findIndex((g) => QUESTION_WORDS.has(g));
  if (questionWordIdx !== -1) {
    return handleQuestion(glosses, questionWordIdx);
  }

  // Check for Time Markers (ASL puts time at start: YESTERDAY, TOMORROW, TODAY)
  let tense: "past" | "present" | "future" = "present";
  let timeAdverb: string | null = null;
  const filteredGlosses: string[] = [];

  for (const g of glosses) {
    if (g === "YESTERDAY" || g === "PAST" || g === "BEFORE") {
      tense = "past";
      timeAdverb = g === "YESTERDAY" ? "yesterday" : "in the past";
    } else if (g === "TOMORROW" || g === "FUTURE" || g === "SOON") {
      tense = "future";
      timeAdverb = g === "TOMORROW" ? "tomorrow" : "soon";
    } else if (g === "TODAY") {
      tense = "present";
      timeAdverb = "today";
    } else if (g === "NOW") {
      timeAdverb = "now";
    } else {
      filteredGlosses.push(g);
    }
  }

  let subject = "I";
  let subjectDetermined = false;
  let hasNegation = false;
  let verb: string | null = null;
  let objectOrComplement: string | null = null;
  let adjective: string | null = null;

  for (const token of filteredGlosses) {
    // 1. Subject pronoun resolution (only before a verb has been determined)
    if (!subjectDetermined && !verb) {
      if (token === "ME" || token === "I") {
        subject = "I";
        subjectDetermined = true;
        continue;
      } else if (token === "YOU") {
        subject = "you";
        subjectDetermined = true;
        continue;
      } else if (token === "HE") {
        subject = "he";
        subjectDetermined = true;
        continue;
      } else if (token === "SHE") {
        subject = "she";
        subjectDetermined = true;
        continue;
      } else if (token === "WE") {
        subject = "we";
        subjectDetermined = true;
        continue;
      } else if (token === "THEY") {
        subject = "they";
        subjectDetermined = true;
        continue;
      }
    }

    // 2. Object pronouns (if subject or verb is already established)
    if (subjectDetermined || verb) {
      if (token === "ME" || token === "I") {
        objectOrComplement = "me";
        continue;
      } else if (token === "YOU") {
        objectOrComplement = "you";
        continue;
      } else if (token === "HIM") {
        objectOrComplement = "him";
        continue;
      } else if (token === "HER") {
        objectOrComplement = "her";
        continue;
      } else if (token === "US") {
        objectOrComplement = "us";
        continue;
      } else if (token === "THEM") {
        objectOrComplement = "them";
        continue;
      }
    }

    // 3. Negation
    if (token === "NOT" || token === "DONT" || token === "NEVER") {
      hasNegation = true;
      continue;
    }

    // 4. Verbs
    if (!verb && VERB_CONJUGATIONS[token]) {
      verb = token;
      continue;
    }

    // 5. Adjectives
    if (ADJECTIVES.has(token)) {
      adjective = token.toLowerCase();
      continue;
    }

    // 6. Zero-Article Locations ("to school", "to work", "home")
    if (ZERO_ARTICLE_LOCATIONS.has(token)) {
      objectOrComplement = token === "HOME" ? "home" : `to ${token.toLowerCase()}`;
      continue;
    }

    // 7. Standard Locations ("to the store", "to the bathroom")
    if (STANDARD_LOCATIONS.has(token)) {
      objectOrComplement = `to the ${token.toLowerCase()}`;
      continue;
    }

    // 8. Countable nouns needing indefinite article ("a car", "a question")
    if (COUNTABLE_NOUNS.has(token)) {
      const article = startsWithVowel(token) ? "an" : "a";
      objectOrComplement = `${article} ${token.toLowerCase()}`;
      continue;
    }

    // 9. Possessives or other nouns (e.g. "MY SISTER", "WATER", "COFFEE", "HELP")
    const lower = token.toLowerCase();
    if (objectOrComplement) {
      objectOrComplement += ` ${lower}`;
    } else {
      objectOrComplement = lower;
    }
  }

  // Assemble the English sentence
  const parts: string[] = [];
  parts.push(subject);

  if (adjective) {
    // Subject + Copula + [not] + Adjective (e.g. "I am hungry", "She is not tired")
    const copula = getCopula(subject, tense);
    parts.push(copula);
    if (hasNegation) parts.push("not");
    parts.push(adjective);
  } else if (verb) {
    const verbInfo = VERB_CONJUGATIONS[verb];
    if (verbInfo) {
      if (tense === "future") {
        parts.push("will");
        if (hasNegation) parts.push("not");
        parts.push(verbInfo.base);
      } else if (tense === "past" || (hasNegation && verb === "COME")) {
        if (hasNegation) {
          parts.push("did not");
          parts.push(verbInfo.base);
        } else {
          parts.push(verbInfo.past);
        }
      } else {
        // Present tense
        if (hasNegation) {
          parts.push(subject === "he" || subject === "she" ? "does not" : "do not");
          parts.push(verbInfo.base);
        } else {
          parts.push(subject === "he" || subject === "she" ? verbInfo.present : verbInfo.base);
        }
      }

      if (objectOrComplement) {
        parts.push(objectOrComplement);
      }
    }
  } else if (objectOrComplement) {
    // Subject + Copula + [not] + Noun Complement (e.g. "She is my sister", "He is my friend")
    const copula = getCopula(subject, tense);
    parts.push(copula);
    if (hasNegation) parts.push("not");
    parts.push(objectOrComplement);
  }

  // Append time adverb at the end for natural flow
  if (timeAdverb) {
    parts.push(timeAdverb);
  }

  return formatFinalSentence(parts.join(" "), ".");
}

function handleQuestion(glosses: string[], qIdx: number): string {
  const qWord = glosses[qIdx];
  if (!qWord) return "";
  const others = glosses.filter((_, idx) => idx !== qIdx);

  // Canonical question patterns
  // "NAME YOU WHAT" / "YOU WHAT NAME" -> "What is your name?"
  if (qWord === "WHAT") {
    if (others.includes("NAME")) {
      const target = others.includes("YOU") ? "your" : "his";
      return `What is ${target} name?`;
    }
    if (others.includes("TIME")) {
      return "What time is it?";
    }
    if (others.includes("YOU") && others.includes("WANT")) {
      return "What do you want?";
    }
  }

  // "WHERE"
  if (qWord === "WHERE") {
    if (others.includes("YOU") && others.includes("LIVE")) {
      return "Where do you live?";
    }
    if (others.includes("YOU") && others.includes("GO")) {
      return "Where are you going?";
    }
    const loc = others.find((t) => STANDARD_LOCATIONS.has(t) || ZERO_ARTICLE_LOCATIONS.has(t));
    if (loc) {
      return `Where is the ${loc.toLowerCase()}?`;
    }
  }

  // "WHEN"
  if (qWord === "WHEN") {
    if (others.includes("ARRIVE") || others.includes("COME")) {
      return "When will you arrive?";
    }
  }

  // "HOW"
  if (qWord === "HOW") {
    if (others.includes("MUCH") || others.includes("COST")) {
      return "How much does this cost?";
    }
    if (others.includes("YOU")) {
      return "How are you?";
    }
  }

  // "WHO"
  if (qWord === "WHO") {
    if (others.includes("YOUR") && others.includes("FRIEND")) {
      return "Who is your friend?";
    }
    if (others.includes("THAT")) {
      return "Who is that?";
    }
  }

  // "WHY"
  if (qWord === "WHY") {
    if (others.includes("YOU") && others.includes("LATE")) {
      return "Why are you late?";
    }
  }

  // Generic question fallback: [QWord] [copula/aux] [subject] [rest]?
  const qWordLower = capitalize(qWord.toLowerCase());
  const restLower = others
    .map((t) => (t === "YOU" ? "you" : t === "ME" ? "I" : t.toLowerCase()))
    .join(" ");
  return formatFinalSentence(`${qWordLower} is ${restLower}`, "?");
}

function getCopula(subject: string, tense: "past" | "present" | "future"): string {
  if (tense === "past") {
    return subject === "I" || subject === "he" || subject === "she" ? "was" : "were";
  }
  if (tense === "future") {
    return "will be";
  }
  if (subject === "I") return "am";
  if (subject === "he" || subject === "she") return "is";
  return "are";
}

function startsWithVowel(word: string): boolean {
  return /^[aeiou]/i.test(word);
}

function capitalize(text: string): string {
  if (!text) return "";
  return text.charAt(0).toUpperCase() + text.slice(1);
}

function formatFinalSentence(text: string, punctuation: "." | "?" | "!"): string {
  const trimmed = text.trim();
  if (!trimmed) return "";
  const capitalized = trimmed.charAt(0).toUpperCase() + trimmed.slice(1);
  if (capitalized.endsWith(".") || capitalized.endsWith("?") || capitalized.endsWith("!")) {
    return capitalized;
  }
  return `${capitalized}${punctuation}`;
}
