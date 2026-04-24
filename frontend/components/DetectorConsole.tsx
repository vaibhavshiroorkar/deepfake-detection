"use client";

import { useEffect, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { Image as ImageIcon, Film, AudioLines, AlignLeft } from "lucide-react";
import clsx from "clsx";
import Uploader from "./Uploader";
import TextPane from "./TextPane";
import ResultPanel from "./ResultPanel";
import PreviewPanel from "./PreviewPanel";
import ErrorBoundary from "./ErrorBoundary";
import { warmBackend, type DetectionResult } from "@/lib/api";

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
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [fileInfo, setFileInfo] = useState<{ name: string; size: number } | null>(
    null,
  );
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    warmBackend();
  }, []);

  function resetMediaPreview() {
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    setPreviewUrl(null);
    setFileInfo(null);
  }

  function switchTab(next: Tab) {
    setTab(next);
    setResult(null);
    setError(null);
    resetMediaPreview();
  }

  return (
    <section id="console" className="max-w-7xl">
      <div className="grid gap-8 lg:grid-cols-[minmax(0,1fr)_26rem]">
        <div>
          <div className="border border-ink bg-paper shadow-[8px_8px_0_rgba(20,20,19,0.08)]">
            <div className="flex border-b border-ink">
              {TABS.map(({ id, label, Icon }) => {
                const active = tab === id;
                return (
                  <button
                    key={id}
                    onClick={() => switchTab(id)}
                    className={clsx(
                      "relative flex-1 px-6 py-5 flex items-center justify-center gap-2.5 border-r border-rule last:border-r-0 transition-colors",
                      active ? "text-ink bg-bone" : "text-mute hover:text-smoke hover:bg-bone/40",
                    )}
                  >
                    <Icon className="size-4" strokeWidth={1.6} />
                    <span className="text-sm font-medium tracking-wide">
                      {label}
                    </span>
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

            <div className="p-7 md:p-10">
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
                          setError(
                            e instanceof Error ? e.message : "Something went wrong.",
                          );
                        } finally {
                          setLoading(false);
                        }
                      }}
                    />
                  ) : (
                    <Uploader
                      kind={tab}
                      loading={loading}
                      onPick={(info) => {
                        if (previewUrl) URL.revokeObjectURL(previewUrl);
                        if (info) {
                          setPreviewUrl(info.previewUrl);
                          setFileInfo({ name: info.file.name, size: info.file.size });
                        } else {
                          setPreviewUrl(null);
                          setFileInfo(null);
                        }
                      }}
                      onSubmit={async (file) => {
                        setLoading(true);
                        setError(null);
                        setResult(null);
                        try {
                          const { detectImage, detectVideo, detectAudio } = await import(
                            "@/lib/api"
                          );
                          const r =
                            tab === "image"
                              ? await detectImage(file)
                              : tab === "video"
                              ? await detectVideo(file)
                              : await detectAudio(file);
                          setResult(r);
                        } catch (e) {
                          setError(
                            e instanceof Error ? e.message : "Something went wrong.",
                          );
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
                className="mt-6 border-l-2 border-alert bg-alert/5 px-5 py-4"
              >
                <div
                  className="eyebrow eyebrow-ember mb-1"
                  style={{ color: "var(--alert)" }}
                >
                  Error
                </div>
                <div className="text-[0.95rem] text-ink">{error}</div>
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
                <ErrorBoundary>
                  <ResultPanel result={result} previewUrl={previewUrl} />
                </ErrorBoundary>
              </motion.div>
            )}
          </AnimatePresence>
        </div>

        {tab !== "text" && (
          <PreviewPanel
            kind={tab}
            previewUrl={previewUrl}
            fileName={fileInfo?.name ?? null}
            fileSize={fileInfo?.size ?? null}
          />
        )}
      </div>
    </section>
  );
}
