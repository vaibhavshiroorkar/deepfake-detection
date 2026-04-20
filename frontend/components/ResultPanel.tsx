"use client";

import { motion } from "framer-motion";
import { verdictTone, type DetectionResult, type Signal } from "@/lib/api";
import clsx from "clsx";

export default function ResultPanel({ result }: { result: DetectionResult }) {
  const tone = verdictTone(result.suspicion);
  const pct = Math.round(result.suspicion * 100);
  const conf = Math.round(result.confidence * 100);

  return (
    <article className="border border-rule bg-paper">
      <header className="flex items-center justify-between border-b border-rule px-5 py-3">
        <div className="flex items-center gap-2.5">
          <span
            className={clsx(
              "size-2 rounded-full",
              tone.color === "forest" && "bg-forest",
              tone.color === "amber" && "bg-amber",
              tone.color === "ember" && "bg-ember",
              tone.color === "alert" && "bg-alert",
            )}
          />
          <span className="text-xs text-smoke capitalize">
            {result.kind} reading
          </span>
        </div>
        <span className="text-xs text-mute">confidence {conf}%</span>
      </header>

      <div className="px-5 py-6 md:px-7 md:py-8 border-b border-rule">
        <div className="text-xs text-mute mb-2">Verdict</div>
        <motion.h3
          initial={{ opacity: 0, y: 6 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.4 }}
          className={clsx(
            "font-display tracking-tight",
            tone.color === "forest" && "text-forest",
            tone.color === "amber" && "text-ink",
            tone.color === "ember" && "text-ember",
            tone.color === "alert" && "text-alert",
          )}
          style={{ fontSize: "clamp(1.6rem, 3vw, 2.2rem)", lineHeight: 1.1 }}
        >
          {capitalize(result.verdict)}.
        </motion.h3>

        <Gauge suspicion={result.suspicion} />

        <p className="mt-5 text-sm leading-[1.65] text-smoke max-w-prose">
          {humanePreamble(result, pct)}
        </p>
      </div>

      <div className="px-5 py-6 md:px-7 md:py-8">
        <div className="text-xs text-mute mb-4">Signals examined</div>
        <ul className="space-y-4">
          {result.signals.map((s, i) => (
            <SignalRow key={s.name} signal={s} index={i} />
          ))}
        </ul>

        {"timeline" in result && result.timeline && (
          <div className="mt-8">
            <div className="text-xs text-mute mb-3">Timeline</div>
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
                    <span className="text-[10px] text-mute">{t.timestamp}s</span>
                  </div>
                );
              })}
            </div>
          </div>
        )}
      </div>
    </article>
  );
}

function Gauge({ suspicion }: { suspicion: number }) {
  const pct = Math.round(suspicion * 100);
  return (
    <div className="mt-5">
      <div className="flex items-center justify-between mb-2">
        <span className="text-xs text-mute">Suspicion</span>
        <span className="text-sm tabular-nums text-ink">{pct}/100</span>
      </div>
      <div className="relative h-1.5">
        <div className="absolute inset-0 gauge-track" />
        <motion.div
          className="absolute top-1/2 -translate-y-1/2 w-[2px] h-4 bg-ink"
          initial={{ left: 0 }}
          animate={{ left: `calc(${pct}% - 1px)` }}
          transition={{ duration: 0.8, ease: [0.2, 0.8, 0.2, 1] }}
        />
      </div>
      <div className="mt-1.5 flex justify-between text-[10px] text-mute">
        <span>authentic</span>
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
      initial={{ opacity: 0, x: -6 }}
      animate={{ opacity: 1, x: 0 }}
      transition={{ delay: 0.05 + index * 0.06, duration: 0.35 }}
    >
      <div className="flex items-baseline justify-between gap-4">
        <span className="text-sm text-ink">{signal.name}</span>
        <span className="text-xs text-mute tabular-nums">{width}%</span>
      </div>
      <div className="mt-1.5 h-[2px] bg-rule relative overflow-hidden">
        <motion.div
          initial={{ width: 0 }}
          animate={{ width: `${width}%` }}
          transition={{ delay: 0.1 + index * 0.06, duration: 0.6 }}
          className={clsx(
            "absolute inset-y-0 left-0",
            tone.color === "forest" && "bg-forest",
            tone.color === "amber" && "bg-amber",
            tone.color === "ember" && "bg-ember",
            tone.color === "alert" && "bg-alert",
          )}
        />
      </div>
      <p className="mt-2 text-xs leading-[1.6] text-smoke">{signal.detail}</p>
    </motion.li>
  );
}

function capitalize(s: string) {
  return s.charAt(0).toUpperCase() + s.slice(1);
}

function humanePreamble(r: DetectionResult, pct: number): string {
  if (r.kind === "text") {
    if (pct < 30) return "The prose reads like a person wrote it — uneven sentences, stray contractions, the occasional sharp turn.";
    if (pct < 55) return "Some signs of machine drafting, but not enough to lean on. Could be a human who writes cleanly, or an AI draft that's been edited.";
    if (pct < 75) return "Several markers line up: even sentence lengths, scaffolding phrases, a comma-heavy register. Likely a language model.";
    return "Strong statistical fingerprints of LLM writing throughout.";
  }
  if (r.kind === "video") {
    if (pct < 30) return "Grain is stable, edges hold between frames, faces sit cleanly on bodies. Nothing unusual.";
    if (pct < 55) return "A few frames read oddly. Compression and ordinary noise can produce signals like these.";
    if (pct < 75) return "The clip flickers where it shouldn't, and face boundaries drift across time. Signature of per-frame synthesis.";
    return "Multiple frames fail the same tests. Treat as synthetic.";
  }
  if (r.kind === "audio") {
    if (pct < 30) return "Pitch moves, silences carry room tone, the spectrum looks like a real recording.";
    if (pct < 55) return "A couple of markers are mildly raised. Heavy compression or noise reduction can mimic them.";
    if (pct < 75) return "The voice is too flat, or the silences are too clean, or the spectrum is missing what a microphone would give it.";
    return "Multiple tells of synthesized speech line up. Likely TTS or a voice clone.";
  }
  if (pct < 30) return "Noise is uniform, the spectrum looks like a normal camera capture, no seams around faces.";
  if (pct < 55) return "A few signals are softly raised. Heavy editing or re-compression can mimic manipulation.";
  if (pct < 75) return "Several forensic channels disagree with the image's story. Common for composites and generator outputs.";
  return "Fails multiple independent checks in ways that correlate. Likely manipulated.";
}
