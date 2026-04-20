"use client";

import { useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { Image as ImageIcon, Film, AlignLeft } from "lucide-react";
import clsx from "clsx";
import Uploader from "./Uploader";
import TextPane from "./TextPane";
import ResultPanel from "./ResultPanel";
import type { DetectionResult } from "@/lib/api";

type Tab = "image" | "video" | "text";

const TABS: { id: Tab; label: string; kicker: string; Icon: typeof ImageIcon }[] = [
  { id: "image", label: "Image", kicker: "Stills", Icon: ImageIcon },
  { id: "video", label: "Video", kicker: "Moving picture", Icon: Film },
  { id: "text", label: "Text", kicker: "Prose", Icon: AlignLeft },
];

export default function DetectorConsole() {
  const [tab, setTab] = useState<Tab>("image");
  const [result, setResult] = useState<DetectionResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  function reset() {
    setResult(null);
    setError(null);
  }

  return (
    <section id="console" className="relative">
      <div className="mx-auto max-w-[1400px] px-6 py-16 md:py-20">
        <div className="grid md:grid-cols-12 gap-6 md:gap-12 items-start">
          <div className="md:col-span-4">
            <div className="font-mono text-[11px] uppercase tracking-[0.28em] text-ember">
              §II — The Desk
            </div>
            <h3 className="mt-4 font-display tracking-tight text-ink" style={{ fontSize: "clamp(1.8rem, 3.2vw, 2.6rem)", lineHeight: 1.05 }}>
              Submit a specimen.
              <br />
              <span className="italic text-smoke">We'll return a reading.</span>
            </h3>
            <p className="mt-5 text-smoke leading-relaxed max-w-sm">
              Choose a medium. Drop a file, or paste your text. The desk works
              on your machine's side — your upload goes to our service, a
              reading comes back, nothing is retained.
            </p>

            <ul className="mt-8 space-y-3 text-sm text-smoke">
              {[
                ["Image", "JPEG · PNG · WebP · up to 25 MB"],
                ["Video", "MP4 · WebM · MOV · up to 200 MB"],
                ["Text", "Plain prose, 40–50 000 characters"],
              ].map(([k, v]) => (
                <li key={k} className="flex items-baseline gap-4">
                  <span className="font-mono text-[10px] uppercase tracking-[0.2em] text-mute w-14">{k}</span>
                  <span className="flex-1 border-b border-dotted border-rule translate-y-[-3px]" />
                  <span className="font-mono text-xs text-smoke">{v}</span>
                </li>
              ))}
            </ul>
          </div>

          <div className="md:col-span-8">
            <div className="border border-rule bg-paper shadow-paper relative">
              {/* Tabs */}
              <div className="flex border-b border-rule">
                {TABS.map(({ id, label, kicker, Icon }) => {
                  const active = tab === id;
                  return (
                    <button
                      key={id}
                      onClick={() => {
                        setTab(id);
                        reset();
                      }}
                      className={clsx(
                        "group relative flex-1 px-5 py-5 text-left border-r border-rule last:border-r-0 transition-colors",
                        active ? "bg-bone" : "bg-paper hover:bg-bone/60",
                      )}
                    >
                      <div className="flex items-center gap-3">
                        <Icon className={clsx("size-4", active ? "text-ember" : "text-mute group-hover:text-smoke")} />
                        <div>
                          <div className="font-mono text-[10px] uppercase tracking-[0.22em] text-mute">
                            {kicker}
                          </div>
                          <div className={clsx("font-display text-xl tracking-tight", active ? "text-ink" : "text-smoke")}>
                            {label}
                          </div>
                        </div>
                      </div>
                      {active && (
                        <motion.div
                          layoutId="tab-underline"
                          className="absolute inset-x-0 -bottom-px h-[2px] bg-ember"
                        />
                      )}
                    </button>
                  );
                })}
              </div>

              {/* Body */}
              <div className="p-6 md:p-8 relative">
                <AnimatePresence mode="wait">
                  <motion.div
                    key={tab}
                    initial={{ opacity: 0, y: 6 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, y: -6 }}
                    transition={{ duration: 0.3, ease: [0.2, 0.8, 0.2, 1] }}
                  >
                    {tab === "text" ? (
                      <TextPane
                        loading={loading}
                        onSubmit={async (text) => {
                          setLoading(true);
                          setError(null);
                          setResult(null);
                          try {
                            const { detectText } = await import("@/lib/api");
                            const r = await detectText(text);
                            setResult(r);
                          } catch (e) {
                            setError(e instanceof Error ? e.message : "Something went wrong.");
                          } finally {
                            setLoading(false);
                          }
                        }}
                      />
                    ) : (
                      <Uploader
                        kind={tab}
                        loading={loading}
                        onSubmit={async (file) => {
                          setLoading(true);
                          setError(null);
                          setResult(null);
                          try {
                            const { detectImage, detectVideo } = await import("@/lib/api");
                            const r = tab === "image" ? await detectImage(file) : await detectVideo(file);
                            setResult(r);
                          } catch (e) {
                            setError(e instanceof Error ? e.message : "Something went wrong.");
                          } finally {
                            setLoading(false);
                          }
                        }}
                      />
                    )}
                  </motion.div>
                </AnimatePresence>
              </div>
            </div>

            {/* Result or error */}
            <AnimatePresence>
              {error && (
                <motion.div
                  initial={{ opacity: 0, y: 6 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0 }}
                  className="mt-5 border border-alert/50 bg-alert/10 px-5 py-4 text-sm text-alert font-mono"
                >
                  {error}
                </motion.div>
              )}
              {result && !loading && (
                <motion.div
                  key={result.kind + String(result.suspicion)}
                  initial={{ opacity: 0, y: 12 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0 }}
                  transition={{ duration: 0.5, ease: [0.2, 0.8, 0.2, 1] }}
                  className="mt-8"
                >
                  <ResultPanel result={result} />
                </motion.div>
              )}
            </AnimatePresence>
          </div>
        </div>
      </div>
    </section>
  );
}
