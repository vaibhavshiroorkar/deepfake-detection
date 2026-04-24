"use client";

import { useCallback, useMemo, useState } from "react";
import { Upload, FileImage, FileVideo, FileAudio, X } from "lucide-react";
import clsx from "clsx";

type Kind = "image" | "video" | "audio";

type Props = {
  kind: Kind;
  loading: boolean;
  onSubmit: (file: File) => void | Promise<void>;
};

const ACCEPT: Record<Kind, string> = {
  image: "image/jpeg,image/png,image/webp,image/jpg",
  video: "video/mp4,video/webm,video/quicktime,video/x-matroska",
  audio: "audio/mpeg,audio/mp3,audio/wav,audio/wave,audio/x-wav,audio/flac,audio/ogg,audio/x-flac,audio/vorbis",
};

const LABEL: Record<Kind, { noun: string; hint: string }> = {
  image: { noun: "an image", hint: "JPEG · PNG · WebP · up to 25 MB" },
  video: { noun: "a video", hint: "MP4 · WebM · MOV · up to 200 MB" },
  audio: { noun: "a clip", hint: "MP3 · WAV · FLAC · OGG · up to 50 MB" },
};

export default function Uploader({ kind, loading, onSubmit }: Props) {
  const [file, setFile] = useState<File | null>(null);
  const [dragging, setDragging] = useState(false);
  const [preview, setPreview] = useState<string | null>(null);

  const pretty = useMemo(() => {
    if (!file) return null;
    const kb = file.size / 1024;
    return kb > 1024 ? `${(kb / 1024).toFixed(1)} MB` : `${kb.toFixed(0)} KB`;
  }, [file]);

  const onDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setDragging(false);
    const f = e.dataTransfer.files?.[0];
    if (f) pick(f);
  }, []);

  const pick = (f: File) => {
    const allowed = ACCEPT[kind].split(",");
    if (!allowed.some((a) => f.type === a || f.type.startsWith(a.split("/")[0] + "/"))) {
      alert(`That file type isn't supported for ${kind}.`);
      return;
    }
    setFile(f);
    if (preview) URL.revokeObjectURL(preview);
    setPreview(URL.createObjectURL(f));
  };

  const clear = () => {
    setFile(null);
    if (preview) URL.revokeObjectURL(preview);
    setPreview(null);
  };

  const Icon = kind === "image" ? FileImage : kind === "video" ? FileVideo : FileAudio;

  return (
    <div>
      {!file ? (
        <label
          onDragOver={(e) => {
            e.preventDefault();
            setDragging(true);
          }}
          onDragLeave={() => setDragging(false)}
          onDrop={onDrop}
          className={clsx(
            "relative block cursor-pointer border-2 border-dashed transition-colors",
            "px-10 py-20 text-center",
            dragging ? "border-ember bg-ember/5" : "border-rule hover:border-ink bg-bone/30",
          )}
        >
          <input
            type="file"
            accept={ACCEPT[kind]}
            className="sr-only"
            onChange={(e) => {
              const f = e.target.files?.[0];
              if (f) pick(f);
            }}
          />
          <div className="flex flex-col items-center gap-5">
            <Upload className="size-8 text-ink" strokeWidth={1.3} />
            <p className="font-display text-2xl tracking-tight text-ink">
              Drop {LABEL[kind].noun} here
            </p>
            <p className="body-sm text-smoke">
              or{" "}
              <span className="underline decoration-ember decoration-2 underline-offset-[6px] text-ink">
                browse your files
              </span>
            </p>
            <p className="font-mono text-xs tracking-widest text-mute uppercase mt-2">
              {LABEL[kind].hint}
            </p>
          </div>
        </label>
      ) : (
        <div className="border border-ink bg-bone/40 p-5 md:p-6">
          <div className="flex items-start gap-5">
            <div className="size-24 shrink-0 border border-ink bg-ink/5 overflow-hidden flex items-center justify-center">
              {preview && kind === "image" ? (
                <img src={preview} alt="preview" className="size-full object-cover" />
              ) : preview && kind === "video" ? (
                <video src={preview} className="size-full object-cover" muted />
              ) : (
                <Icon className="size-7 text-mute" strokeWidth={1.3} />
              )}
            </div>
            <div className="flex-1 min-w-0">
              <div className="font-display text-xl tracking-tight truncate">
                {file.name}
              </div>
              <div className="mt-1 font-mono text-xs text-mute tracking-wider">
                {pretty}
              </div>
              {preview && kind === "audio" && (
                <audio src={preview} controls className="mt-4 w-full h-9" />
              )}
              <div className="mt-6 flex items-center gap-6">
                <button
                  onClick={() => onSubmit(file)}
                  disabled={loading}
                  className={clsx(
                    "inline-flex items-center gap-2.5 px-6 py-3 text-sm tracking-wide transition-colors border",
                    loading
                      ? "bg-mute text-paper border-mute cursor-wait"
                      : "bg-ink text-paper border-ink hover:bg-ember hover:border-ember",
                  )}
                >
                  {loading ? (
                    <>
                      <span className="size-1.5 rounded-full bg-paper pulse-soft" />
                      Examining — this may take a moment on cold start
                    </>
                  ) : (
                    <>
                      Examine
                      <span aria-hidden>→</span>
                    </>
                  )}
                </button>
                <button
                  onClick={clear}
                  disabled={loading}
                  className="inline-flex items-center gap-1.5 text-sm text-smoke hover:text-alert transition-colors"
                >
                  <X className="size-3.5" />
                  Remove
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
