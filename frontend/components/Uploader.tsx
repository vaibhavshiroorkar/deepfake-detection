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
            "relative block cursor-pointer border border-dashed transition-colors",
            "px-8 py-12 text-center",
            dragging ? "border-ember bg-ember/5" : "border-rule hover:border-smoke bg-bone/40",
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
          <div className="flex flex-col items-center gap-3">
            <Upload className="size-5 text-smoke" />
            <p className="text-ink">
              Drop {LABEL[kind].noun} here,{" "}
              <span className="underline decoration-ember underline-offset-4">or browse</span>
            </p>
            <p className="text-xs text-mute">{LABEL[kind].hint}</p>
          </div>
        </label>
      ) : (
        <div className="border border-rule bg-bone/40 p-4">
          <div className="flex items-start gap-4">
            <div className="size-20 shrink-0 border border-rule bg-ink/5 overflow-hidden flex items-center justify-center">
              {preview && kind === "image" ? (
                <img src={preview} alt="preview" className="size-full object-cover" />
              ) : preview && kind === "video" ? (
                <video src={preview} className="size-full object-cover" muted />
              ) : (
                <Icon className="size-6 text-mute" />
              )}
            </div>
            <div className="flex-1 min-w-0">
              <div className="text-sm truncate">{file.name}</div>
              <div className="mt-0.5 text-xs text-mute">{pretty}</div>
              {preview && kind === "audio" && (
                <audio src={preview} controls className="mt-3 w-full h-8" />
              )}
              <div className="mt-4 flex items-center gap-3">
                <button
                  onClick={() => onSubmit(file)}
                  disabled={loading}
                  className={clsx(
                    "inline-flex items-center gap-2 px-4 py-2 text-sm transition-colors",
                    loading ? "bg-mute text-white cursor-wait" : "bg-ink text-white hover:bg-ember",
                  )}
                >
                  {loading ? (
                    <>
                      <span className="size-1.5 rounded-full bg-white pulse-soft" />
                      Examining
                    </>
                  ) : (
                    <>Examine<span>→</span></>
                  )}
                </button>
                <button
                  onClick={clear}
                  disabled={loading}
                  className="inline-flex items-center gap-1.5 text-sm text-smoke hover:text-alert"
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
