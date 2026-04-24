"use client";

import { FileImage, FileVideo, FileAudio, AlignLeft } from "lucide-react";

type Kind = "image" | "video" | "audio" | "text";

type Props = {
  kind: Kind;
  previewUrl: string | null;
  fileName?: string | null;
  fileSize?: number | null;
  textSample?: string;
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
  textSample,
}: Props) {
  const prettySize = formatSize(fileSize);
  const hasFile = Boolean(previewUrl);
  const hasText = kind === "text" && Boolean(textSample && textSample.trim());

  return (
    <aside className="border border-ink bg-paper sticky top-[5.5rem] self-start">
      <header className="px-5 py-3 border-b border-ink flex items-baseline justify-between">
        <span className="running-head">Preview</span>
        <span className="running-head text-mute capitalize">{kind}</span>
      </header>

      <div className="min-h-[18rem] bg-bone/40 border-b border-rule flex items-center justify-center overflow-hidden">
        {kind === "image" && previewUrl ? (
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
            <audio src={previewUrl} controls className="w-full h-10" />
          </div>
        ) : hasText ? (
          <div className="w-full p-6">
            <p className="font-display italic text-xl leading-snug text-ink/80 line-clamp-[10]">
              &ldquo;{textSample}&rdquo;
            </p>
          </div>
        ) : (
          <EmptyState kind={kind} />
        )}
      </div>

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
        ) : hasText ? (
          <span className="font-mono text-xs text-mute">
            {textSample!.trim().split(/\s+/).length} words
          </span>
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
    kind === "image"
      ? FileImage
      : kind === "video"
      ? FileVideo
      : kind === "audio"
      ? FileAudio
      : AlignLeft;

  const hint: Record<Kind, string> = {
    image: "Your image will show up here.",
    video: "Your video will play here.",
    audio: "Your clip will show up here.",
    text: "Your writing will appear here as you type.",
  };

  return (
    <div className="w-full py-12 flex flex-col items-center gap-4 text-mute">
      <Icon className="size-8" strokeWidth={1.2} />
      <p className="body-sm max-w-[26ch] text-center">{hint[kind]}</p>
    </div>
  );
}
