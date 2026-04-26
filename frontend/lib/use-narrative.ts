"use client";

import { useEffect, useState } from "react";
import type { DetectionResult } from "./api";
import { narrate } from "./narrate";
import { imageToScaledDataURL, videoKeyframesToDataURLs } from "./image-prep";

type State =
  | { status: "loading"; text: null; source: null; vision: false }
  | { status: "ready"; text: string; source: "ai" | "template"; vision: boolean };

/**
 * Returns the narrative for a result. Behaviour:
 *
 *   1. While the API call is in flight, status is "loading" and the
 *      component should render a skeleton (no text yet).
 *   2. If the result is image/video and a previewUrl is available, the
 *      image (or middle video frame) is downscaled and sent along so
 *      the LLM can actually see what it's describing.
 *   3. If /api/narrate returns an LLM-written narrative, that text is
 *      shown with source="ai". `vision` indicates whether the model
 *      had eyes on the picture.
 *   4. If the API call fails or returns no narrative, the local
 *      template generator runs and that text is shown with
 *      source="template".
 */
export function useNarrative(
  result: DetectionResult,
  previewUrl?: string | null,
): State {
  const [state, setState] = useState<State>({
    status: "loading",
    text: null,
    source: null,
    vision: false,
  });

  useEffect(() => {
    let cancelled = false;
    setState({ status: "loading", text: null, source: null, vision: false });

    const ctrl = new AbortController();
    const timeout = setTimeout(() => ctrl.abort(), 16_000);

    const fallback = () => {
      if (cancelled) return;
      setState({
        status: "ready",
        text: narrate(result),
        source: "template",
        vision: false,
      });
    };

    (async () => {
      let imageDataUrl: string | null = null;
      let imageDataUrls: string[] | null = null;
      if (previewUrl) {
        try {
          if (result.kind === "image") {
            imageDataUrl = await imageToScaledDataURL(previewUrl);
          } else if (result.kind === "video") {
            // Three keyframes at 25/50/75% of duration so the model
            // sees the actual content, not a single frame that might
            // misrepresent what the clip is about.
            imageDataUrls = await videoKeyframesToDataURLs(previewUrl, 3);
            if (imageDataUrls.length === 0) imageDataUrls = null;
          }
        } catch {
          imageDataUrl = null;
          imageDataUrls = null;
        }
      }
      if (cancelled) return;

      try {
        const res = await fetch("/api/narrate", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            kind: result.kind,
            suspicion: result.suspicion,
            verdict: result.verdict,
            confidence: result.confidence,
            signals: result.signals,
            image_data_url: imageDataUrl,
            image_data_urls: imageDataUrls,
          }),
          signal: ctrl.signal,
        });
        if (!res.ok) return fallback();
        const data = (await res.json()) as {
          narrative?: string | null;
          vision?: boolean;
        };
        const narrative = data.narrative ?? null;
        if (cancelled) return;
        if (narrative && narrative.length > 0) {
          setState({
            status: "ready",
            text: narrative,
            source: "ai",
            vision: Boolean(data.vision),
          });
        } else {
          fallback();
        }
      } catch {
        fallback();
      } finally {
        clearTimeout(timeout);
      }
    })();

    return () => {
      cancelled = true;
      ctrl.abort();
      clearTimeout(timeout);
    };
    // Re-run when the result identity changes.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [result.kind, result.suspicion, result.verdict, result.signals.length, previewUrl]);

  return state;
}
