"use client";

import { useCallback, useMemo, useState } from "react";
import { Upload, FileImage, FileVideo, X } from "lucide-react";
import clsx from "clsx";

type Props = {
  kind: "image" | "video";
  loading: boolean;
  onSubmit: (file: File) => void | Promise<void>;
};

const ACCEPT = {
  image: "image/jpeg,image/png,image/webp,image/jpg",
  video: "video/mp4,video/webm,video/quicktime,video/x-matroska",
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
    if (!f) return;
    pick(f);
  }, []);

  const pick = (f: File) => {
    const allowed = ACCEPT[kind].split(",");
    if (!allowed.some((a) => f.type === a || f.type.startsWith(a.split("/")[0] + "/"))) {
      alert(`That file type isn't supported for ${kind} — please try another.`);
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

  const submit = async () => {
    if (!file) return;
    await onSubmit(file);
  };

  const Icon = kind === "image" ? FileImage : FileVideo;

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
            "relative block cursor-pointer border-2 border-dashed transition-all",
            "px-8 py-14 text-center",
            dragging
              ? "border-ember bg-ember/5"
              : "border-rule hover:border-smoke bg-bone/40 hover:bg-bone",
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
          <div className="flex flex-col items-center gap-4">
            <div className="relative">
              <div className="size-14 rounded-full border border-rule bg-paper flex items-center justify-center">
                <Upload className="size-5 text-smoke" />
              </div>
              <span className="absolute -right-1 -top-1 size-3 rounded-full bg-ember pulse-soft" />
            </div>
            <div>
              <p className="font-display text-2xl tracking-tight text-ink">
                {kind === "image" ? "Drop an image" : "Drop a clip"}
              </p>
              <p className="mt-2 text-sm text-smoke">
                or <span className="underline decoration-ember decoration-2 underline-offset-4">choose from your machine</span>
              </p>
            </div>
            <p className="font-mono text-[10px] uppercase tracking-[0.22em] text-mute mt-2">
              {kind === "image"
                ? "JPEG · PNG · WebP  —  up to 25 MB"
                : "MP4 · WebM · MOV  —  up to 200 MB"}
            </p>
          </div>
        </label>
      ) : (
        <div className="border border-rule bg-bone/40 p-5">
          <div className="flex items-start gap-5">
            <div className="size-24 shrink-0 border border-rule bg-ink/5 overflow-hidden flex items-center justify-center">
              {preview && kind === "image" ? (
                <img src={preview} alt="preview" className="size-full object-cover" />
              ) : preview && kind === "video" ? (
                <video src={preview} className="size-full object-cover" muted />
              ) : (
                <Icon className="size-8 text-mute" />
              )}
            </div>
            <div className="flex-1 min-w-0">
              <div className="font-mono text-[10px] uppercase tracking-[0.2em] text-mute">
                Specimen received
              </div>
              <div className="mt-1 font-display text-lg tracking-tight truncate">
                {file.name}
              </div>
              <div className="mt-1 font-mono text-xs text-smoke">
                {file.type || "unknown type"} · {pretty}
              </div>
              <div className="mt-5 flex items-center gap-3">
                <button
                  onClick={submit}
                  disabled={loading}
                  className={clsx(
                    "inline-flex items-center gap-3 px-5 py-2.5 font-mono text-xs uppercase tracking-[0.2em] transition-colors",
                    loading ? "bg-mute text-paper cursor-wait" : "bg-ink text-paper hover:bg-ember",
                  )}
                >
                  {loading ? (
                    <>
                      <span className="size-2 rounded-full bg-paper pulse-soft" />
                      Examining
                    </>
                  ) : (
                    <>
                      Examine
                      <span>→</span>
                    </>
                  )}
                </button>
                <button
                  onClick={clear}
                  disabled={loading}
                  className="inline-flex items-center gap-2 text-smoke hover:text-alert font-mono text-xs uppercase tracking-[0.2em]"
                >
                  <X className="size-3" />
                  Discard
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
