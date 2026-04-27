"use client";

import { useEffect, useRef, useState } from "react";
import { FileImage, FileVideo, FileAudio } from "lucide-react";
import SubtitleBox from "./SubtitleBox";
import HeatmapOverlay from "./HeatmapOverlay";
import VideoTimelinePlayer from "./VideoTimelinePlayer";
import type { DetectionResult, Transcript } from "@/lib/api";

type Kind = "image" | "video" | "audio";

type Props = {
  kind: Kind;
  previewUrl: string | null;
  fileName?: string | null;
  fileSize?: number | null;
  transcript?: Transcript | null;
  result?: DetectionResult | null;
};

function formatSize(bytes: number | null | undefined) {
  if (!bytes) return null;
  const kb = bytes / 1024;
  return kb > 1024 ? `${(kb / 1024).toFixed(1)} MB` : `${kb.toFixed(0)} KB`;
}

export default function PreviewPanel({
  kind,
  previewUrl,
  fileName,
  fileSize,
  transcript,
  result,
}: Props) {
  const prettySize = formatSize(fileSize);
  const hasFile = Boolean(previewUrl);
  const audioRef = useRef<HTMLAudioElement>(null);
  const [audioTime, setAudioTime] = useState(0);

  useEffect(() => {
    const a = audioRef.current;
    if (!a) return;
    const onTime = () => setAudioTime(a.currentTime);
    a.addEventListener("timeupdate", onTime);
    return () => a.removeEventListener("timeupdate", onTime);
  }, [previewUrl]);

  const imageResult = result?.kind === "image" ? result : null;
  const videoResult = result?.kind === "video" ? result : null;
  const showHeatmap = Boolean(imageResult?.heatmaps && previewUrl);
  const showTimeline = Boolean(videoResult?.timeline && previewUrl);

  const showAudioSubtitles =
    kind === "audio" &&
    previewUrl &&
    transcript &&
    (transcript.text || (transcript.chunks && transcript.chunks.length > 0));

  return (
    <aside className="border border-ink bg-paper lg:sticky lg:top-[5.5rem] lg:self-start">
      <header className="px-5 py-3 border-b border-ink flex items-baseline justify-between">
        <span className="running-head">
          {showHeatmap ? "Analysis" : showTimeline ? "Timeline" : "Preview"}
        </span>
        <span className="running-head text-mute capitalize">{kind}</span>
      </header>

      <div
        className={
          showTimeline
            ? "border-b border-rule"
            : "min-h-[18rem] bg-bone/40 border-b border-rule flex items-center justify-center overflow-hidden"
        }
      >
        {showHeatmap ? (
          <div className="w-full p-4">
            <HeatmapOverlay base={previewUrl!} heatmaps={imageResult!.heatmaps!} />
          </div>
        ) : showTimeline ? (
          <VideoTimelinePlayer
            src={previewUrl!}
            timeline={videoResult!.timeline}
            duration={videoResult!.duration_seconds || 0}
            transcript={videoResult!.transcript}
          />
        ) : kind === "image" && previewUrl ? (
          <img
            src={previewUrl}
            alt="Selected file"
            className="max-h-[26rem] w-full object-contain"
          />
        ) : kind === "video" && previewUrl ? (
          <video
            src={previewUrl}
            controls
            className="w-full max-h-[26rem] bg-ink"
          />
        ) : kind === "audio" && previewUrl ? (
          <div className="w-full p-6 flex flex-col items-center gap-5">
            <FileAudio className="size-10 text-ink" strokeWidth={1.2} />
            <audio
              ref={audioRef}
              src={previewUrl}
              controls
              className="w-full h-10"
            />
          </div>
        ) : (
          <EmptyState kind={kind} />
        )}
      </div>

      {showAudioSubtitles && (
        <div className="border-b border-rule">
          <SubtitleBox
            transcript={transcript!}
            currentTime={audioTime}
            onSeek={(t) => {
              const a = audioRef.current;
              if (!a) return;
              a.currentTime = Math.max(0, t);
              setAudioTime(a.currentTime);
            }}
            title="Spoken transcript"
          />
        </div>
      )}

      <footer className="px-5 py-3 flex items-center justify-between min-h-[2.75rem]">
        {hasFile && fileName ? (
          <>
            <span className="font-mono text-xs text-ink truncate pr-3">
              {fileName}
            </span>
            {prettySize && (
              <span className="font-mono text-xs text-mute shrink-0">
                {prettySize}
              </span>
            )}
          </>
        ) : (
          <span className="font-mono text-xs text-mute">
            Nothing selected yet
          </span>
        )}
      </footer>
    </aside>
  );
}

function EmptyState({ kind }: { kind: Kind }) {
  const Icon =
    kind === "image" ? FileImage : kind === "video" ? FileVideo : FileAudio;

  const hint: Record<Kind, string> = {
    image: "Your image will show up here.",
    video: "Your video will play here.",
    audio: "Your clip will show up here.",
  };

  return (
    <div className="w-full py-12 flex flex-col items-center gap-4 text-mute">
      <Icon className="size-8" strokeWidth={1.2} />
      <p className="body-sm max-w-[26ch] text-center">{hint[kind]}</p>
    </div>
  );
}
