"use client";

import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import { Play, Pause } from "lucide-react";
import clsx from "clsx";
import { verdictTone } from "@/lib/api";

type TimelinePoint = {
  timestamp: number;
  suspicion: number;
  verdict: string;
};

type Props = {
  src: string;
  timeline: TimelinePoint[];
  duration: number;
};

function formatTime(seconds: number): string {
  if (!isFinite(seconds) || seconds < 0) return "0:00";
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${m}:${s.toString().padStart(2, "0")}`;
}

/**
 * Build a continuous suspicion curve over the duration of the video.
 * The backend samples a small set of timestamps; for the timeline
 * strip we want a colour at every pixel, so we linearly interpolate
 * between sampled points.
 */
function suspicionAt(timeline: TimelinePoint[], t: number): number {
  if (!timeline.length) return 0;
  if (t <= timeline[0].timestamp) return timeline[0].suspicion;
  if (t >= timeline[timeline.length - 1].timestamp) {
    return timeline[timeline.length - 1].suspicion;
  }
  for (let i = 0; i < timeline.length - 1; i++) {
    const a = timeline[i];
    const b = timeline[i + 1];
    if (t >= a.timestamp && t <= b.timestamp) {
      const span = b.timestamp - a.timestamp;
      if (span <= 0) return a.suspicion;
      const ratio = (t - a.timestamp) / span;
      return a.suspicion + (b.suspicion - a.suspicion) * ratio;
    }
  }
  return timeline[timeline.length - 1].suspicion;
}

const TONE_TO_RGB: Record<string, string> = {
  forest: "47, 93, 69",
  amber: "232, 163, 61",
  ember: "216, 71, 39",
  alert: "184, 52, 31",
};

export default function VideoTimelinePlayer({
  src,
  timeline,
  duration,
}: Props) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const trackRef = useRef<HTMLDivElement>(null);
  const [playing, setPlaying] = useState(false);
  const [current, setCurrent] = useState(0);
  const [hoverTime, setHoverTime] = useState<number | null>(null);
  const [actualDuration, setActualDuration] = useState(duration);

  // Sort timeline by timestamp defensively — backend usually does but
  // this is cheap insurance.
  const sortedTimeline = useMemo(
    () => [...timeline].sort((a, b) => a.timestamp - b.timestamp),
    [timeline],
  );

  // Render the suspicion gradient as a tall narrow set of vertical
  // strips. Using ~120 strips across the strip width gives a smooth
  // gradient feel without a per-pixel canvas.
  const STRIPS = 120;
  const stripData = useMemo(() => {
    if (actualDuration <= 0 || sortedTimeline.length === 0) return [];
    const out: { tone: string; suspicion: number; t: number }[] = [];
    for (let i = 0; i < STRIPS; i++) {
      const t = ((i + 0.5) / STRIPS) * actualDuration;
      const sus = suspicionAt(sortedTimeline, t);
      const tone = verdictTone(sus).color;
      out.push({ tone, suspicion: sus, t });
    }
    return out;
  }, [sortedTimeline, actualDuration]);

  // Keep the actualDuration synced once the video metadata loads, in
  // case the backend's reported duration is slightly off.
  useEffect(() => {
    const v = videoRef.current;
    if (!v) return;
    const onMeta = () => {
      if (isFinite(v.duration) && v.duration > 0) {
        setActualDuration(v.duration);
      }
    };
    const onTime = () => setCurrent(v.currentTime);
    const onPlay = () => setPlaying(true);
    const onPause = () => setPlaying(false);
    const onEnded = () => setPlaying(false);
    v.addEventListener("loadedmetadata", onMeta);
    v.addEventListener("timeupdate", onTime);
    v.addEventListener("play", onPlay);
    v.addEventListener("pause", onPause);
    v.addEventListener("ended", onEnded);
    return () => {
      v.removeEventListener("loadedmetadata", onMeta);
      v.removeEventListener("timeupdate", onTime);
      v.removeEventListener("play", onPlay);
      v.removeEventListener("pause", onPause);
      v.removeEventListener("ended", onEnded);
    };
  }, []);

  const seekFromEvent = useCallback(
    (clientX: number): number | null => {
      const track = trackRef.current;
      if (!track || actualDuration <= 0) return null;
      const rect = track.getBoundingClientRect();
      const x = Math.min(Math.max(clientX - rect.left, 0), rect.width);
      return (x / rect.width) * actualDuration;
    },
    [actualDuration],
  );

  const handleSeek = useCallback(
    (clientX: number) => {
      const t = seekFromEvent(clientX);
      if (t === null) return;
      const v = videoRef.current;
      if (v) {
        v.currentTime = t;
        setCurrent(t);
      }
    },
    [seekFromEvent],
  );

  const togglePlay = () => {
    const v = videoRef.current;
    if (!v) return;
    if (v.paused) v.play().catch(() => {});
    else v.pause();
  };

  const playheadPct =
    actualDuration > 0 ? Math.min(100, (current / actualDuration) * 100) : 0;
  const hoverPct =
    hoverTime !== null && actualDuration > 0
      ? Math.min(100, (hoverTime / actualDuration) * 100)
      : null;
  const hoverSus =
    hoverTime !== null ? suspicionAt(sortedTimeline, hoverTime) : null;

  return (
    <div className="border border-rule bg-paper">
      <div className="bg-ink relative">
        <video
          ref={videoRef}
          src={src}
          className="w-full max-h-[480px] block bg-ink mx-auto"
          playsInline
          onClick={togglePlay}
        />
      </div>

      <div className="px-4 py-3 flex items-center gap-3 border-t border-rule">
        <button
          onClick={togglePlay}
          aria-label={playing ? "Pause" : "Play"}
          className="size-8 flex items-center justify-center bg-ink text-paper hover:bg-ember transition-colors shrink-0"
        >
          {playing ? (
            <Pause className="size-3.5" strokeWidth={2} />
          ) : (
            <Play className="size-3.5 translate-x-[1px]" strokeWidth={2} />
          )}
        </button>

        <div className="flex-1 min-w-0">
          <div
            ref={trackRef}
            role="slider"
            tabIndex={0}
            aria-label="Seek and suspicion timeline"
            aria-valuemin={0}
            aria-valuemax={actualDuration}
            aria-valuenow={current}
            className="relative h-7 cursor-pointer select-none"
            onPointerDown={(e) => {
              (e.target as HTMLElement).setPointerCapture?.(e.pointerId);
              handleSeek(e.clientX);
            }}
            onPointerMove={(e) => {
              const t = seekFromEvent(e.clientX);
              setHoverTime(t);
              if (e.buttons === 1) handleSeek(e.clientX);
            }}
            onPointerLeave={() => setHoverTime(null)}
            onKeyDown={(e) => {
              const v = videoRef.current;
              if (!v) return;
              if (e.key === "ArrowLeft") {
                v.currentTime = Math.max(0, v.currentTime - 1);
              } else if (e.key === "ArrowRight") {
                v.currentTime = Math.min(actualDuration, v.currentTime + 1);
              } else if (e.key === " ") {
                e.preventDefault();
                togglePlay();
              }
            }}
          >
            <div className="absolute inset-0 flex bg-bone/60 overflow-hidden">
              {stripData.map((strip, i) => {
                const rgb = TONE_TO_RGB[strip.tone] ?? TONE_TO_RGB.forest;
                // Mix base toward the tone colour by suspicion. Even
                // very low-suspicion regions get a faint forest tint
                // so the strip visibly tells the story across the
                // whole clip.
                const alpha = 0.25 + Math.min(0.7, strip.suspicion * 0.95);
                return (
                  <div
                    key={i}
                    className="flex-1 h-full"
                    style={{ backgroundColor: `rgba(${rgb}, ${alpha})` }}
                  />
                );
              })}
            </div>

            {sortedTimeline.map((p, i) => {
              const pct =
                actualDuration > 0
                  ? Math.min(100, (p.timestamp / actualDuration) * 100)
                  : 0;
              const tone = verdictTone(p.suspicion).color;
              const rgb = TONE_TO_RGB[tone] ?? TONE_TO_RGB.forest;
              return (
                <div
                  key={i}
                  className="absolute top-0 bottom-0 w-px pointer-events-none opacity-60"
                  style={{
                    left: `${pct}%`,
                    backgroundColor: `rgba(${rgb}, 0.95)`,
                  }}
                />
              );
            })}

            <div
              className="absolute top-0 bottom-0 w-[2px] bg-ink pointer-events-none"
              style={{ left: `${playheadPct}%` }}
            />

            {hoverPct !== null && (
              <div
                className="absolute top-0 bottom-0 w-px bg-ink/40 pointer-events-none"
                style={{ left: `${hoverPct}%` }}
              />
            )}
          </div>

          <div className="mt-1.5 flex items-center justify-between font-mono text-[10px] text-mute tabular-nums">
            <span>{formatTime(current)}</span>
            {hoverTime !== null && hoverSus !== null ? (
              <span
                className={clsx(
                  "text-ink",
                )}
              >
                {formatTime(hoverTime)} · suspicion {Math.round(hoverSus * 100)}
              </span>
            ) : (
              <span>{formatTime(actualDuration)}</span>
            )}
          </div>
        </div>
      </div>

      <div className="px-4 pb-3 flex items-center gap-4 text-[10px] text-mute font-mono uppercase tracking-widest">
        <span className="flex items-center gap-1.5">
          <span className="block w-3 h-2 bg-forest" /> low
        </span>
        <span className="flex items-center gap-1.5">
          <span className="block w-3 h-2 bg-amber" /> mid
        </span>
        <span className="flex items-center gap-1.5">
          <span className="block w-3 h-2 bg-ember" /> high
        </span>
        <span className="flex items-center gap-1.5">
          <span className="block w-3 h-2 bg-alert" /> very high
        </span>
        <span className="ml-auto normal-case tracking-normal text-mute/80">
          {sortedTimeline.length} sampled frames
        </span>
      </div>
    </div>
  );
}
