"use client";

import { useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { Image as ImageIcon, Film, AudioLines, AlignLeft } from "lucide-react";
import clsx from "clsx";
import Uploader from "./Uploader";
import TextPane from "./TextPane";
import ResultPanel from "./ResultPanel";
import type { DetectionResult } from "@/lib/api";

type Tab = "text" | "image" | "audio" | "video";

const TABS: { id: Tab; label: string; Icon: typeof ImageIcon }[] = [
  { id: "text", label: "Text", Icon: AlignLeft },
  { id: "image", label: "Image", Icon: ImageIcon },
  { id: "audio", label: "Audio", Icon: AudioLines },
  { id: "video", label: "Video", Icon: Film },
];

export default function DetectorConsole() {
  const [tab, setTab] = useState<Tab>("text");
  const [result, setResult] = useState<DetectionResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  function reset() {
    setResult(null);
    setError(null);
  }

  return (
    <section id="console" className="mx-auto max-w-3xl px-6 pb-20">
      <div className="border border-rule bg-paper rounded-sm">
        <div className="flex border-b border-rule">
          {TABS.map(({ id, label, Icon }) => {
            const active = tab === id;
            return (
              <button
                key={id}
                onClick={() => {
                  setTab(id);
                  reset();
                }}
                className={clsx(
                  "relative flex-1 px-5 py-4 flex items-center justify-center gap-2 border-r border-rule last:border-r-0 transition-colors",
                  active ? "text-ink" : "text-mute hover:text-smoke",
                )}
              >
                <Icon className="size-4" />
                <span className="text-sm">{label}</span>
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

        <div className="p-6 md:p-8">
          <AnimatePresence mode="wait">
            <motion.div
              key={tab}
              initial={{ opacity: 0, y: 4 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -4 }}
              transition={{ duration: 0.2 }}
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
                      const { detectImage, detectVideo, detectAudio } = await import("@/lib/api");
                      const r =
                        tab === "image"
                          ? await detectImage(file)
                          : tab === "video"
                          ? await detectVideo(file)
                          : await detectAudio(file);
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

      <AnimatePresence>
        {error && (
          <motion.div
            initial={{ opacity: 0, y: 6 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0 }}
            className="mt-5 border border-alert/40 bg-alert/5 px-5 py-3 text-sm text-alert"
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
            transition={{ duration: 0.4 }}
            className="mt-6"
          >
            <ResultPanel result={result} />
          </motion.div>
        )}
      </AnimatePresence>
    </section>
  );
}
