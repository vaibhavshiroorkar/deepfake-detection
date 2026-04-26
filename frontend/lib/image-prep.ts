/**
 * Client-side image preparation for the vision LLM. Downscales to a
 * sensible upper bound and re-encodes as JPEG so the data URL stays
 * well under any reasonable request body limit. Vision LLMs don't
 * need full resolution, ~768px on the long side is plenty.
 */

const MAX_SIDE = 768;
const JPEG_QUALITY = 0.85;

export async function imageToScaledDataURL(
  url: string,
  maxSide = MAX_SIDE,
): Promise<string | null> {
  if (typeof window === "undefined") return null;
  return new Promise((resolve) => {
    const img = new Image();
    img.crossOrigin = "anonymous";
    img.onload = () => {
      try {
        const longest = Math.max(img.naturalWidth, img.naturalHeight);
        const scale = longest > maxSide ? maxSide / longest : 1;
        const w = Math.max(1, Math.round(img.naturalWidth * scale));
        const h = Math.max(1, Math.round(img.naturalHeight * scale));
        const canvas = document.createElement("canvas");
        canvas.width = w;
        canvas.height = h;
        const ctx = canvas.getContext("2d");
        if (!ctx) return resolve(null);
        ctx.drawImage(img, 0, 0, w, h);
        resolve(canvas.toDataURL("image/jpeg", JPEG_QUALITY));
      } catch {
        resolve(null);
      }
    };
    img.onerror = () => resolve(null);
    img.src = url;
  });
}

export async function videoFrameToDataURL(
  url: string,
  maxSide = MAX_SIDE,
): Promise<string | null> {
  if (typeof window === "undefined") return null;
  return new Promise((resolve) => {
    const v = document.createElement("video");
    v.muted = true;
    v.playsInline = true;
    v.preload = "auto";
    v.crossOrigin = "anonymous";
    let settled = false;
    const cleanup = () => {
      v.removeAttribute("src");
      v.load();
    };
    const finish = (out: string | null) => {
      if (settled) return;
      settled = true;
      cleanup();
      resolve(out);
    };
    v.onloadedmetadata = () => {
      // Aim for the middle frame, capped at 1.5s for very long clips.
      const target = Math.min(v.duration / 2 || 0.5, 1.5);
      v.currentTime = isFinite(target) ? target : 0.1;
    };
    v.onseeked = () => {
      try {
        const longest = Math.max(v.videoWidth, v.videoHeight);
        if (!longest) return finish(null);
        const scale = longest > maxSide ? maxSide / longest : 1;
        const w = Math.max(1, Math.round(v.videoWidth * scale));
        const h = Math.max(1, Math.round(v.videoHeight * scale));
        const canvas = document.createElement("canvas");
        canvas.width = w;
        canvas.height = h;
        const ctx = canvas.getContext("2d");
        if (!ctx) return finish(null);
        ctx.drawImage(v, 0, 0, w, h);
        finish(canvas.toDataURL("image/jpeg", JPEG_QUALITY));
      } catch {
        finish(null);
      }
    };
    v.onerror = () => finish(null);
    // Some browsers won't seek without a play() nudge.
    setTimeout(() => finish(null), 6000);
    v.src = url;
  });
}
