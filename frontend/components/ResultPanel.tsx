"use client";

import { motion } from "framer-motion";
import { ShieldCheck, ShieldAlert, Sparkles } from "lucide-react";
import { verdictTone, type DetectionResult, type Signal } from "@/lib/api";
import { useNarrative } from "@/lib/use-narrative";
import HeatmapOverlay from "./HeatmapOverlay";
import clsx from "clsx";

export default function ResultPanel({
  result,
  previewUrl,
}: {
  result: DetectionResult;
  previewUrl?: string | null;
}) {
  const tone = verdictTone(result.suspicion);
  const conf = Math.round(result.confidence * 100);
  const narrative = useNarrative(result, previewUrl);

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

        {narrative.status === "loading" ? (
          <div className="mt-5 max-w-prose space-y-2" aria-busy="true">
            <div className="h-3 w-[92%] bg-ink/10 pulse-soft rounded-sm" />
            <div className="h-3 w-[80%] bg-ink/10 pulse-soft rounded-sm" />
            <div className="h-3 w-[60%] bg-ink/10 pulse-soft rounded-sm" />
            <div className="mt-3 flex items-center gap-1.5 text-[10px] tracking-widest uppercase text-mute font-mono">
              <Sparkles className="size-3 text-ember" strokeWidth={1.6} />
              <span>Writing it up</span>
            </div>
          </div>
        ) : (
          <>
            <motion.p
              key={narrative.text}
              initial={{ opacity: 0, y: 4 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.45 }}
              className="mt-5 text-sm leading-[1.65] text-smoke max-w-prose"
            >
              {narrative.text}
            </motion.p>
            {narrative.source === "ai" && (
              <div className="mt-2 flex items-center gap-1.5 text-[10px] tracking-widest uppercase text-mute font-mono">
                <Sparkles className="size-3 text-ember" strokeWidth={1.6} />
                <span>
                  {narrative.vision
                    ? "Written by Llama 4 (looking at the image)"
                    : "Written by Llama 3.3"}
                </span>
              </div>
            )}
          </>
        )}

        {result.kind === "image" && result.c2pa?.present && (
          <C2PABadge manifest={result.c2pa} />
        )}
      </div>

      {result.kind === "image" && previewUrl && result.heatmaps && (
        <div className="px-5 py-6 md:px-7 md:py-8 border-b border-rule">
          <div className="text-xs text-mute mb-3">Where the signal fired</div>
          <HeatmapOverlay base={previewUrl} heatmaps={result.heatmaps} />
        </div>
      )}

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

function splitDetail(detail: string): { summary: string; technical: string | null } {
  const trimmed = detail.trim();
  if (!trimmed) return { summary: "", technical: null };
  // Backend Signal details are typically "[numbers + model refs]. [plain
  // takeaway]." We split at the last ". " so the plain takeaway (summary)
  // goes first in big type and the numbers drop below in mono. If the
  // whole thing is a single sentence (e.g. error fallbacks), treat it
  // all as the summary.
  const lastSplit = trimmed.lastIndexOf(". ");
  if (lastSplit === -1) {
    return { summary: trimmed, technical: null };
  }
  const technical = trimmed.slice(0, lastSplit + 1).trim();
  const summary = trimmed.slice(lastSplit + 2).trim();
  if (!summary) return { summary: trimmed, technical: null };
  return { summary, technical };
}

function SignalRow({ signal, index }: { signal: Signal; index: number }) {
  const width = Math.round(signal.score * 100);
  const tone = verdictTone(signal.score);
  const { summary, technical } = splitDetail(signal.detail);
  return (
    <motion.li
      initial={{ opacity: 0, x: -6 }}
      animate={{ opacity: 1, x: 0 }}
      transition={{ delay: 0.05 + index * 0.06, duration: 0.35 }}
    >
      <div className="flex items-baseline justify-between gap-4">
        <span className="text-[0.95rem] text-ink font-semibold tracking-tight">
          {signal.name}
        </span>
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
      <p className="mt-2 text-[0.8125rem] leading-[1.55] text-smoke">
        {summary}
      </p>
      {technical && (
        <p className="mt-1 font-mono text-[10.5px] leading-[1.5] text-mute">
          {technical}
        </p>
      )}
    </motion.li>
  );
}

function C2PABadge({
  manifest,
}: {
  manifest: NonNullable<Extract<DetectionResult, { kind: "image" }>["c2pa"]>;
}) {
  const trusted = manifest.trusted;
  return (
    <div
      className={clsx(
        "mt-5 flex items-start gap-3 px-4 py-3 border",
        trusted ? "border-forest/40 bg-forest/5" : "border-amber/40 bg-amber/5",
      )}
    >
      {trusted ? (
        <ShieldCheck className="size-4 text-forest mt-0.5 shrink-0" />
      ) : (
        <ShieldAlert className="size-4 text-amber mt-0.5 shrink-0" />
      )}
      <div className="text-xs leading-relaxed">
        <div className="text-ink">
          Content Credentials present
          {manifest.signed_by ? ` (signed by ${manifest.signed_by})` : " (unsigned)"}
        </div>
        <div className="text-mute mt-0.5">
          {manifest.claim_generator || "unknown generator"}
          {manifest.actions && manifest.actions.length > 0 && (
            <> · {manifest.actions.join(", ")}</>
          )}
        </div>
      </div>
    </div>
  );
}

function capitalize(s: string) {
  return s.charAt(0).toUpperCase() + s.slice(1);
}

// humanePreamble has been replaced by narrate() in lib/narrate.ts.
