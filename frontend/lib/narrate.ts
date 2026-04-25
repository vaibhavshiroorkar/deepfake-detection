/**
 * Narrative summary generator. Takes a finished detection result and
 * produces a multi-sentence explanation that reads like a careful
 * analyst wrote it. Pure prose generation from the actual signal
 * data, no LLM calls, no network.
 */
import type { DetectionResult, Signal } from "./api";

type Tone = "authentic" | "inconclusive" | "suspicious" | "manipulated";

function bucket(score: number): Tone {
  if (score < 0.3) return "authentic";
  if (score < 0.55) return "inconclusive";
  if (score < 0.75) return "suspicious";
  return "manipulated";
}

function pickStrong(signals: Signal[], threshold = 0.55): Signal[] {
  return signals
    .filter((s) => s.score >= threshold)
    .sort((a, b) => b.score - a.score)
    .slice(0, 3);
}

function pickQuiet(signals: Signal[], threshold = 0.25): Signal[] {
  return signals
    .filter((s) => s.score <= threshold)
    .sort((a, b) => a.score - b.score)
    .slice(0, 3);
}

function isLearned(name: string): boolean {
  const n = name.toLowerCase();
  return (
    n.includes("classifier") ||
    n.includes("detector") ||
    n.includes("ensemble") ||
    n.includes("transformer") ||
    n.includes("perplexity") ||
    n.includes("roberta") ||
    n.includes("whisper") ||
    n.includes("gpt-2") ||
    n.includes("dinov2")
  );
}

function isHeuristic(name: string): boolean {
  const n = name.toLowerCase();
  return (
    n.includes("focus") ||
    n.includes("chromatic") ||
    n.includes("noise") ||
    n.includes("ela") ||
    n.includes("error-level") ||
    n.includes("flicker") ||
    n.includes("burstiness") ||
    n.includes("rhythm") ||
    n.includes("punctuation") ||
    n.includes("silence") ||
    n.includes("pitch") ||
    n.includes("energy") ||
    n.includes("metadata") ||
    n.includes("exif")
  );
}

function listify(items: string[]): string {
  if (items.length === 0) return "";
  if (items.length === 1) return items[0];
  if (items.length === 2) return `${items[0]} and ${items[1]}`;
  return `${items.slice(0, -1).join(", ")}, and ${items[items.length - 1]}`;
}

function lower(name: string): string {
  // Keep model identifiers cased, lowercase common nouns.
  if (/^[A-Z]/.test(name) && /[A-Z]/.test(name.slice(1))) return name;
  return name.charAt(0).toLowerCase() + name.slice(1);
}

function intensifier(score: number): string {
  if (score >= 0.85) return "very strongly";
  if (score >= 0.7) return "strongly";
  if (score >= 0.55) return "moderately";
  if (score >= 0.4) return "mildly";
  return "weakly";
}

function pctText(score: number): string {
  return `${Math.round(score * 100)}%`;
}

function classifierLine(strong: Signal[], quiet: Signal[]): string {
  const strongLearned = strong.filter((s) => isLearned(s.name));
  const quietLearned = quiet.filter((s) => isLearned(s.name));

  if (strongLearned.length >= 2) {
    const names = strongLearned.map((s) => `${lower(s.name)} (${pctText(s.score)})`);
    return `${listify(names)} all flagged this as likely synthetic.`;
  }
  if (strongLearned.length === 1) {
    const s = strongLearned[0];
    return `The ${lower(s.name)} ${intensifier(s.score)} flagged this (${pctText(s.score)}).`;
  }
  if (quietLearned.length >= 2) {
    return `The learned classifiers came back clean.`;
  }
  if (quietLearned.length === 1) {
    return `The ${lower(quietLearned[0].name)} did not see a synthetic fingerprint.`;
  }
  return "";
}

function heuristicLine(strong: Signal[], quiet: Signal[]): string {
  const strongHeur = strong.filter((s) => isHeuristic(s.name));
  const quietHeur = quiet.filter((s) => isHeuristic(s.name));

  if (strongHeur.length > 0) {
    const names = strongHeur.map((s) => lower(s.name));
    return `Forensic checks also flagged ${listify(names)}.`;
  }
  if (quietHeur.length >= 2) {
    const names = quietHeur.slice(0, 2).map((s) => lower(s.name));
    return `${listify(names)} look normal, so the camera physics check out.`;
  }
  if (quietHeur.length === 1) {
    return `The ${lower(quietHeur[0].name)} signal looks unremarkable.`;
  }
  return "";
}

function disagreementLine(strong: Signal[], quiet: Signal[]): string {
  const strongLearned = strong.find((s) => isLearned(s.name));
  const quietLearned = quiet.find((s) => isLearned(s.name));
  const strongHeur = strong.find((s) => isHeuristic(s.name));
  const quietHeur = quiet.find((s) => isHeuristic(s.name));

  if (strongLearned && quietHeur) {
    return `The learned model and the camera-physics signals disagree, which is common when a generator handles optics convincingly but still leaves statistical fingerprints.`;
  }
  if (strongHeur && quietLearned) {
    return `The forensic signals raised a flag, but the learned classifier didn't, so this is more likely a heavy edit or unusual capture than a fully generated image.`;
  }
  return "";
}

function modalitySpecificCloser(result: DetectionResult, tone: Tone): string {
  const k = result.kind;
  if (k === "video") {
    if (tone === "manipulated") {
      return "Frames fail the same checks at multiple timestamps. Treat as synthetic.";
    }
    if (tone === "suspicious") {
      return "Suspicion spikes on certain frames more than others. Worth scrubbing the timeline below.";
    }
    if (tone === "inconclusive") {
      return "A few frames read oddly, but compression and ordinary noise can produce the same signals.";
    }
    return "Per-frame signals are stable across the sampled timeline.";
  }
  if (k === "audio") {
    if (tone === "manipulated") {
      return "Multiple tells of synthesised speech line up. Likely TTS or a voice clone.";
    }
    if (tone === "suspicious") {
      return "The voice is flatter or cleaner than a real recording, but a heavily processed studio capture can read the same way.";
    }
    if (tone === "inconclusive") {
      return "A couple of markers are mildly raised. Noise reduction or aggressive compression can mimic them.";
    }
    return "Pitch moves, silences carry room tone, the spectrum looks like a real recording.";
  }
  if (k === "text") {
    if (tone === "manipulated") {
      return "Strong statistical fingerprints of LLM writing throughout.";
    }
    if (tone === "suspicious") {
      return "Several markers line up: even sentence lengths, scaffolding phrases, a comma-heavy register. Likely a language model.";
    }
    if (tone === "inconclusive") {
      return "Some signs of machine drafting, but not enough to lean on. Could be a clean human writer or an edited AI draft.";
    }
    return "The prose reads like a person wrote it: uneven sentences, stray contractions, the occasional sharp turn.";
  }
  // image
  if (tone === "manipulated") {
    return "Fails multiple independent checks in ways that correlate. Likely manipulated.";
  }
  if (tone === "suspicious") {
    return "Several forensic channels disagree with the image's story. Common for composites and generator outputs.";
  }
  if (tone === "inconclusive") {
    return "A few signals are softly raised. Heavy editing or recompression can produce similar patterns.";
  }
  return "Noise is uniform, the spectrum looks like a normal capture, no seams around faces.";
}

function openingLine(tone: Tone, kind: DetectionResult["kind"]): string {
  const what =
    kind === "text" ? "writing" : kind === "audio" ? "clip" : kind;
  if (tone === "manipulated") {
    return `This ${what} fails enough checks that the verdict is confident: likely synthetic.`;
  }
  if (tone === "suspicious") {
    return `This ${what} reads as probably synthetic, with the caveat that one or two signals could still be ordinary artefacts.`;
  }
  if (tone === "inconclusive") {
    return `This ${what} sits in the middle. Some signals lean synthetic, others lean real, and the honest answer is that we can't tell.`;
  }
  return `This ${what} reads as probably authentic.`;
}

/**
 * Generate a narrative paragraph from a finished result. Returns
 * 2 to 4 sentences of prose explaining what happened.
 */
export function narrate(result: DetectionResult): string {
  const tone = bucket(result.suspicion);
  const signals = result.signals ?? [];
  const strong = pickStrong(signals);
  const quiet = pickQuiet(signals);

  const parts: string[] = [];
  parts.push(openingLine(tone, result.kind));

  // For media (image/video/audio), pull in classifier and forensic lines
  // if anything is interesting on either side.
  if (result.kind !== "text") {
    const cls = classifierLine(strong, quiet);
    if (cls) parts.push(cls);
    const heur = heuristicLine(strong, quiet);
    if (heur) parts.push(heur);
    const dis = disagreementLine(strong, quiet);
    if (dis) parts.push(dis);
  } else {
    // Text: name the strongest model that contributed.
    if (strong.length > 0) {
      const top = strong[0];
      parts.push(
        `The strongest signal was ${lower(top.name)} at ${pctText(top.score)}.`,
      );
    } else if (quiet.length > 0) {
      parts.push(`Nothing in the signal set rose above the noise floor.`);
    }
  }

  parts.push(modalitySpecificCloser(result, tone));

  // De-duplicate adjacent identical sentences (defensive)
  const seen = new Set<string>();
  const cleaned = parts.filter((p) => {
    if (seen.has(p)) return false;
    seen.add(p);
    return true;
  });

  return cleaned.join(" ");
}
