"use client";

import { FileImage, FileVideo, FileAudio } from "lucide-react";

type Kind = "image" | "video" | "audio";

type Props = {
  kind: Kind;
  previewUrl: string | null;
  fileName?: string | null;
  fileSize?: number | null;
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
}: Props) {
  const prettySize = formatSize(fileSize);
  const hasFile = Boolean(previewUrl);

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
