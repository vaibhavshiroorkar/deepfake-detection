"use client";

import { motion } from "framer-motion";
import { verdictTone, type DetectionResult, type Signal } from "@/lib/api";
import clsx from "clsx";

export default function ResultPanel({ result }: { result: DetectionResult }) {
  const tone = verdictTone(result.suspicion);
  const pct = Math.round(result.suspicion * 100);
  const conf = Math.round(result.confidence * 100);

  return (
    <article className="border border-ink/90 bg-paper shadow-paper">
      {/* Header strip */}
      <header className="flex items-center justify-between border-b border-rule px-6 py-4">
        <div className="flex items-center gap-3">
          <span
            className={clsx(
              "size-2 rounded-full",
              tone.color === "forest" && "bg-forest",
              tone.color === "amber" && "bg-amber",
              tone.color === "ember" && "bg-ember",
              tone.color === "alert" && "bg-alert",
            )}
          />
          <span className="font-mono text-[11px] uppercase tracking-[0.24em] text-smoke">
            Reading · {result.kind}
          </span>
        </div>
        <span className="font-mono text-[10px] uppercase tracking-[0.22em] text-mute">
          confidence · {conf}%
        </span>
      </header>

      <div className="grid md:grid-cols-5 gap-0">
        {/* Verdict column */}
        <div className="md:col-span-2 p-7 md:p-9 border-b md:border-b-0 md:border-r border-rule">
          <div className="font-mono text-[10px] uppercase tracking-[0.24em] text-mute">
            The verdict
          </div>
          <motion.h4
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.1, duration: 0.5 }}
            className={clsx(
              "mt-2 font-display font-normal tracking-tight",
              tone.color === "forest" && "text-forest",
              tone.color === "amber" && "text-ink",
              tone.color === "ember" && "text-ember",
              tone.color === "alert" && "text-alert",
            )}
            style={{ fontSize: "clamp(1.8rem, 3vw, 2.6rem)", lineHeight: 1.05 }}
          >
            {capitalize(result.verdict)}.
          </motion.h4>

          <Gauge suspicion={result.suspicion} />

          <div className="mt-6 space-y-2 text-sm leading-relaxed text-smoke">
            <p>
              {humanePreamble(result, pct)}
            </p>
            <p className="text-mute">{humaneAdvice(tone.tone)}</p>
          </div>

          <MetaDetails result={result} />
        </div>

        {/* Signals column */}
        <div className="md:col-span-3 p-7 md:p-9">
          <div className="font-mono text-[10px] uppercase tracking-[0.24em] text-mute mb-5">
            The working — signal by signal
          </div>
          <ul className="space-y-5">
            {result.signals.map((s, i) => (
              <SignalRow key={s.name} signal={s} index={i} />
            ))}
          </ul>

          {"timeline" in result && result.timeline && (
            <div className="mt-8">
              <div className="font-mono text-[10px] uppercase tracking-[0.24em] text-mute mb-3">
                Timeline (sampled)
              </div>
              <div className="grid grid-cols-8 gap-1">
                {result.timeline.map((t, i) => {
                  const v = verdictTone(t.suspicion);
                  return (
                    <div key={i} className="flex flex-col items-center gap-1">
                      <div
                        className={clsx(
                          "w-full",
                          v.color === "forest" && "bg-forest",
                          v.color === "amber" && "bg-amber",
                          v.color === "ember" && "bg-ember",
                          v.color === "alert" && "bg-alert",
                        )}
                        style={{ height: `${Math.max(6, Math.round(t.suspicion * 40))}px` }}
                      />
                      <span className="font-mono text-[9px] text-mute">
                        {t.timestamp}s
                      </span>
                    </div>
                  );
                })}
              </div>
            </div>
          )}
        </div>
      </div>

      <footer className="border-t border-rule px-6 py-4 flex items-center justify-between text-[10px] font-mono uppercase tracking-[0.22em] text-mute">
        <span>Issued {new Date().toLocaleTimeString()}</span>
        <span>Veritas · method cf. §III</span>
      </footer>
    </article>
  );
}

function Gauge({ suspicion }: { suspicion: number }) {
  const pct = Math.round(suspicion * 100);
  return (
    <div className="mt-6">
      <div className="flex items-baseline justify-between">
        <span className="font-mono text-[10px] uppercase tracking-[0.24em] text-mute">
          Suspicion
        </span>
        <span className="font-display text-3xl tracking-tight tabular-nums text-ink">
          {pct}
          <span className="text-mute text-lg">/100</span>
        </span>
      </div>
      <div className="mt-3 relative h-2">
        <div className="absolute inset-0 gauge-track" />
        <motion.div
          className="absolute top-1/2 -translate-y-1/2 w-[2px] h-5 bg-ink"
          initial={{ left: 0 }}
          animate={{ left: `calc(${pct}% - 1px)` }}
          transition={{ duration: 1.0, ease: [0.2, 0.8, 0.2, 1] }}
        />
      </div>
      <div className="mt-1.5 flex justify-between font-mono text-[9px] uppercase tracking-[0.22em] text-mute">
        <span>authentic</span>
        <span>inconclusive</span>
        <span>suspicious</span>
        <span>manipulated</span>
      </div>
    </div>
  );
}

function SignalRow({ signal, index }: { signal: Signal; index: number }) {
  const width = Math.round(signal.score * 100);
  const tone = verdictTone(signal.score);
  return (
    <motion.li
      initial={{ opacity: 0, x: -8 }}
      animate={{ opacity: 1, x: 0 }}
      transition={{ delay: 0.1 + index * 0.08, duration: 0.45 }}
    >
      <div className="flex items-baseline justify-between gap-4">
        <span className="font-display text-lg tracking-tight text-ink">
          {signal.name}
        </span>
        <span className="font-mono text-xs text-smoke tabular-nums">
          {width.toString().padStart(2, "0")}
        </span>
      </div>
      <div className="mt-1.5 h-[3px] bg-rule relative overflow-hidden">
        <motion.div
          initial={{ width: 0 }}
          animate={{ width: `${width}%` }}
          transition={{ delay: 0.2 + index * 0.08, duration: 0.7, ease: [0.2, 0.8, 0.2, 1] }}
          className={clsx(
            "absolute inset-y-0 left-0",
            tone.color === "forest" && "bg-forest",
            tone.color === "amber" && "bg-amber",
            tone.color === "ember" && "bg-ember",
            tone.color === "alert" && "bg-alert",
          )}
        />
      </div>
      <p className="mt-2.5 text-sm leading-relaxed text-smoke max-w-prose">
        {signal.detail}
      </p>
    </motion.li>
  );
}

function MetaDetails({ result }: { result: DetectionResult }) {
  const items: [string, string][] = [];
  if (result.kind === "image") {
    items.push(["Filename", result.filename]);
    items.push(["Dimensions", `${result.dimensions.width} × ${result.dimensions.height}`]);
  } else if (result.kind === "video") {
    items.push(["Filename", result.filename]);
    items.push(["Duration", `${result.duration_seconds}s`]);
    items.push(["Dimensions", `${result.dimensions.width} × ${result.dimensions.height}`]);
  } else {
    items.push(["Characters", String(result.length.characters)]);
    items.push(["Words", String(result.length.words)]);
    items.push(["Sentences", String(result.length.sentences)]);
  }

  return (
    <dl className="mt-8 border-t border-rule pt-5 space-y-2">
      {items.map(([k, v]) => (
        <div key={k} className="flex items-baseline justify-between gap-4">
          <dt className="font-mono text-[10px] uppercase tracking-[0.22em] text-mute">{k}</dt>
          <dd className="font-mono text-xs text-smoke truncate max-w-[240px]" title={v}>{v}</dd>
        </div>
      ))}
    </dl>
  );
}

function capitalize(s: string) {
  return s.charAt(0).toUpperCase() + s.slice(1);
}

function humanePreamble(r: DetectionResult, pct: number): string {
  if (r.kind === "text") {
    if (pct < 30) return "The prose moves the way a person's does — uneven sentences, stray contractions, the occasional sharp turn. The rhythm looks authentically human.";
    if (pct < 55) return "There are some signs of machine drafting, but not enough to lean on. A human may have edited an AI draft, or simply writes in a tidy register.";
    if (pct < 75) return "Several markers line up: even sentence lengths, scaffolding phrases, a comma-heavy, contraction-light voice. This reads like a language model at work.";
    return "The statistical fingerprints of LLM writing are strongly present. If a person wrote this, they wrote to sound like a model — which itself is worth noticing.";
  }
  if (r.kind === "video") {
    if (pct < 30) return "The clip's grain is stable, edges hold between frames, and faces sit cleanly on the bodies that carry them. Nothing here asks for a second look.";
    if (pct < 55) return "Some frames read oddly, but others look fine. Compression artifacts and ordinary video noise can produce signals like these.";
    if (pct < 75) return "The clip flickers where it shouldn't, and face boundaries drift across time. This is the shape a per-frame synthesis leaves behind.";
    return "Multiple frames fail the same tests, and the temporal signature is wrong. This clip should be treated as synthetic until proven otherwise.";
  }
  if (pct < 30) return "The image's noise is uniform, the spectrum falls off as a lens-and-sensor capture would, and no seam appears around the faces. It reads as authentic.";
  if (pct < 55) return "Some signals are softly raised. Heavy editing, re-compression, or an unusual sensor can all mimic manipulation signatures; don't over-read this.";
  if (pct < 75) return "Several forensic channels disagree with the image's story. That pattern is common for composites, face-swaps, and generator outputs.";
  return "The image fails multiple independent checks in ways that correlate. Short of a strong counter-explanation, treat it as manipulated.";
}

function humaneAdvice(tone: "authentic" | "inconclusive" | "suspicious" | "manipulated"): string {
  switch (tone) {
    case "authentic":
      return "Trust, but in the ordinary way — this reading is not a certificate of origin. Provenance still matters.";
    case "inconclusive":
      return "When the signal is mixed, the right move is often a second source, not a stronger reading.";
    case "suspicious":
      return "Look for a higher-resolution original, a first-party publisher, or a witness. Corroborate before you share.";
    case "manipulated":
      return "Consider the context before circulating. What's the claim attached to this file, and who benefits if you believe it?";
  }
}
