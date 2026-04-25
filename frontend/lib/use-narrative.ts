"use client";

import { useEffect, useState } from "react";
import type { DetectionResult } from "./api";
import { narrate } from "./narrate";

type State = {
  text: string;
  source: "template" | "ai";
  loading: boolean;
};

/**
 * Returns the best available narrative for a result. The template is
 * rendered immediately, then a background fetch hits /api/narrate
 * which calls Groq if a key is configured. When the LLM response
 * arrives, swap it in. If anything goes wrong, the template stays.
 */
export function useNarrative(result: DetectionResult): State {
  const fallback = narrate(result);
  const [state, setState] = useState<State>({
    text: fallback,
    source: "template",
    loading: true,
  });

  useEffect(() => {
    let cancelled = false;
    setState({ text: fallback, source: "template", loading: true });

    const ctrl = new AbortController();
    const timeout = setTimeout(() => ctrl.abort(), 13_000);

    fetch("/api/narrate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        kind: result.kind,
        suspicion: result.suspicion,
        verdict: result.verdict,
        confidence: result.confidence,
        signals: result.signals,
      }),
      signal: ctrl.signal,
    })
      .then(async (res) => {
        if (!res.ok) return null;
        const data = (await res.json()) as { narrative?: string | null };
        return data.narrative ?? null;
      })
      .then((narrative) => {
        if (cancelled) return;
        if (narrative && narrative.length > 0) {
          setState({ text: narrative, source: "ai", loading: false });
        } else {
          setState((s) => ({ ...s, loading: false }));
        }
      })
      .catch(() => {
        if (cancelled) return;
        setState((s) => ({ ...s, loading: false }));
      })
      .finally(() => clearTimeout(timeout));

    return () => {
      cancelled = true;
      ctrl.abort();
      clearTimeout(timeout);
    };
    // Re-run when the result identity changes. We hash a small subset
    // because the full result includes object refs.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [result.kind, result.suspicion, result.verdict, result.signals.length]);

  return state;
}
