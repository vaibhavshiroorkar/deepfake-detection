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
  const frames = await videoKeyframesToDataURLs(url, 1, maxSide);
  return frames[0] ?? null;
}

/**
 * Extract a small set of evenly-spaced keyframes from a video. Three
 * frames at 25%, 50%, 75% of duration is enough for a vision LLM to
 * understand what the clip is actually about, without bloating the
 * request body or burning vision tokens. A single middle frame is
 * unreliable (it can land on a black frame, motion blur, or a cut
 * that misrepresents the video).
 */
export async function videoKeyframesToDataURLs(
  url: string,
  count = 3,
  maxSide = MAX_SIDE,
): Promise<string[]> {
  if (typeof window === "undefined") return [];

  const targets = await new Promise<number[]>((resolve) => {
    const v = document.createElement("video");
    v.muted = true;
    v.playsInline = true;
    v.preload = "metadata";
    v.crossOrigin = "anonymous";
    v.onloadedmetadata = () => {
      const dur = isFinite(v.duration) && v.duration > 0 ? v.duration : 0;
      v.removeAttribute("src");
      v.load();
      if (!dur) return resolve([0.1]);
      // Avoid 0 and the very end so we don't catch boundary artefacts.
      const list: number[] = [];
      for (let i = 1; i <= count; i++) {
        list.push((dur * i) / (count + 1));
      }
      resolve(list);
    };
    v.onerror = () => resolve([0.1]);
    setTimeout(() => resolve([0.1]), 4000);
    v.src = url;
  });

  const out: string[] = [];
  for (const t of targets) {
    const frame = await captureVideoFrame(url, t, maxSide);
    if (frame) out.push(frame);
  }
  return out;
}

function captureVideoFrame(
  url: string,
  timestamp: number,
  maxSide: number,
): Promise<string | null> {
  if (typeof window === "undefined") return Promise.resolve(null);
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
      const dur = isFinite(v.duration) && v.duration > 0 ? v.duration : 0;
      const target = dur ? Math.min(timestamp, Math.max(0, dur - 0.1)) : 0.1;
      v.currentTime = target;
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
    setTimeout(() => finish(null), 6000);
    v.src = url;
  });
}
