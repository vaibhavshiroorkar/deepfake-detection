"use client";

import { useEffect, useRef } from "react";
import clsx from "clsx";

export type TranscriptChunk = {
  start: number;
  end: number;
  text: string;
};

export type Transcript = {
  text: string;
  chunks: TranscriptChunk[];
};

type Props = {
  transcript: Transcript;
  currentTime?: number;
  onSeek?: (seconds: number) => void;
  title?: string;
};

function formatTime(seconds: number): string {
  if (!isFinite(seconds) || seconds < 0) return "0:00";
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${m}:${s.toString().padStart(2, "0")}`;
}

export default function SubtitleBox({
  transcript,
  currentTime,
  onSeek,
  title = "Transcript",
}: Props) {
  const chunks = transcript.chunks ?? [];
  const hasChunks = chunks.length > 0;

  // Find the active chunk based on currentTime, if provided.
  let activeIndex = -1;
  if (typeof currentTime === "number" && hasChunks) {
    for (let i = 0; i < chunks.length; i++) {
      const c = chunks[i];
      if (currentTime >= c.start && currentTime <= c.end + 0.05) {
        activeIndex = i;
        break;
      }
    }
    // If we're past the last chunk's end, keep the last one highlighted.
    if (activeIndex === -1 && currentTime > 0) {
      const last = chunks[chunks.length - 1];
      if (currentTime > last.end) activeIndex = chunks.length - 1;
    }
  }

  // Auto-scroll the active chunk into view.
  const containerRef = useRef<HTMLDivElement>(null);
  const activeRef = useRef<HTMLButtonElement>(null);
  useEffect(() => {
    if (activeIndex < 0) return;
    const el = activeRef.current;
    const container = containerRef.current;
    if (!el || !container) return;
    const elRect = el.getBoundingClientRect();
    const cRect = container.getBoundingClientRect();
    if (elRect.top < cRect.top || elRect.bottom > cRect.bottom) {
      el.scrollIntoView({ block: "center", behavior: "smooth" });
    }
  }, [activeIndex]);

  return (
    <div className="border border-rule bg-paper">
      <header className="px-4 py-2.5 border-b border-rule flex items-baseline justify-between">
        <span className="running-head">{title}</span>
        <span className="font-mono text-[10px] text-mute">
          {hasChunks ? `${chunks.length} segments` : "no timestamps"}
        </span>
      </header>

      {hasChunks ? (
        <div
          ref={containerRef}
          className="max-h-44 overflow-y-auto"
          aria-label="Transcript"
        >
          <div className="divide-y divide-rule/60">
            {chunks.map((c, i) => {
              const active = i === activeIndex;
              return (
                <button
                  key={i}
                  ref={active ? activeRef : undefined}
                  onClick={() => onSeek?.(c.start)}
                  className={clsx(
                    "w-full text-left px-4 py-2.5 flex gap-3 transition-colors",
                    onSeek && "cursor-pointer hover:bg-bone/40",
                    !onSeek && "cursor-default",
                    active && "bg-ember/10",
                  )}
                >
                  <span
                    className={clsx(
                      "font-mono text-[10px] tabular-nums shrink-0 mt-0.5 w-12",
                      active ? "text-ember" : "text-mute",
                    )}
                  >
                    {formatTime(c.start)}
                  </span>
                  <span
                    className={clsx(
                      "flex-1 text-sm leading-snug",
                      active ? "text-ink" : "text-smoke",
                    )}
                  >
                    {c.text}
                  </span>
                </button>
              );
            })}
          </div>
        </div>
      ) : (
        <p className="px-4 py-4 text-sm text-smoke leading-relaxed max-h-44 overflow-y-auto">
          {transcript.text || "No speech detected."}
        </p>
      )}
    </div>
  );
}
