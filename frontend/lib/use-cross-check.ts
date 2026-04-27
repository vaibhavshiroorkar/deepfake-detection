"use client";

import { useEffect, useState } from "react";
import type { DetectionResult, Signal } from "./api";
import { imageToScaledDataURL } from "./image-prep";

type CrossCheck = {
  score: number;
  reason: string;
  realism?: number;
};

type State = {
  status: "idle" | "loading" | "ready" | "skipped";
  signals: Signal[];
};

/**
 * Runs an LLM-based cross-check after a detection result lands.
 *
 *   - For image: sends the picture to a vision model and asks for a
 *     0-1 score, a scene-realism score, and a short reason.
 *   - For text: sends the writing to a text model with the same ask.
 *   - For audio and video: skipped.
 *
 * Returns `signals` — a (possibly empty) array of synthetic Signals
 * the caller can splice into the rendered signal list.
 */
export function useCrossCheck(
  result: DetectionResult,
  previewUrl?: string | null,
  inputText?: string,
): State {
  const [state, setState] = useState<State>({ status: "idle", signals: [] });

  useEffect(() => {
    let cancelled = false;
    const ctrl = new AbortController();
    const timeout = setTimeout(() => ctrl.abort(), 16_000);

    const isImage = result.kind === "image" && Boolean(previewUrl);
    const isText = result.kind === "text" && Boolean(inputText);

    if (!isImage && !isText) {
      setState({ status: "skipped", signals: [] });
      return () => {
        cancelled = true;
        clearTimeout(timeout);
      };
    }

    setState({ status: "loading", signals: [] });

    (async () => {
      let body: { kind: "image" | "text"; text?: string; image_data_url?: string } | null = null;
      if (isImage) {
        const data = await imageToScaledDataURL(previewUrl!);
        if (!data) {
          if (!cancelled) setState({ status: "skipped", signals: [] });
          return;
        }
        body = { kind: "image", image_data_url: data };
      } else if (isText) {
        body = { kind: "text", text: inputText };
      }
      if (!body || cancelled) return;

      try {
        const res = await fetch("/api/cross-check", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(body),
          signal: ctrl.signal,
        });
        if (!res.ok) {
          if (!cancelled) setState({ status: "skipped", signals: [] });
          return;
        }
        const data = (await res.json()) as Partial<CrossCheck> & {
          error?: string;
        };
        if (cancelled) return;
        if (data.error || typeof data.score !== "number" || !data.reason) {
          setState({ status: "skipped", signals: [] });
          return;
        }

        const score = Math.max(0, Math.min(1, data.score));
        const tail =
          score > 0.7
            ? "Foundation model thinks this is likely AI."
            : score > 0.5
              ? "Foundation model leans toward AI."
              : score < 0.3
                ? "Foundation model thinks this is real."
                : "Foundation model is uncertain.";

        const signals: Signal[] = [
          {
            name:
              result.kind === "image"
                ? "Vision LLM cross-check"
                : "Language LLM cross-check",
            score,
            detail: `Llama vision/text model: rated ${(score * 100).toFixed(
              0,
            )}/100. ${data.reason} ${tail}`,
          },
        ];

        // Scene-realism signal — image only. An implausible scene (cat in space,
        // impossible physics) is itself evidence of synthetic content.
        if (result.kind === "image" && typeof data.realism === "number") {
          const realism = Math.max(0, Math.min(1, data.realism));
          const suspicion = 1 - realism;
          const realismTail =
            suspicion > 0.6
              ? "Scene depicts something implausible in the real world — strong indicator of synthetic content."
              : suspicion > 0.35
                ? "Scene is unusual but not physically impossible."
                : "Scene is plausible in the real world.";
          signals.push({
            name: "Scene realism",
            score: suspicion,
            detail: `LLM rated scene realism at ${(realism * 100).toFixed(0)}/100. ${realismTail}`,
          });
        }

        setState({ status: "ready", signals });
      } catch {
        if (!cancelled) setState({ status: "skipped", signals: [] });
      } finally {
        clearTimeout(timeout);
      }
    })();

    return () => {
      cancelled = true;
      ctrl.abort();
      clearTimeout(timeout);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [result.kind, result.suspicion, previewUrl, inputText]);

  return state;
}
