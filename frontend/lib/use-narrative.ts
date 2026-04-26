"use client";

import { useEffect, useState } from "react";
import type { DetectionResult } from "./api";
import { narrate } from "./narrate";

type State =
  | { status: "loading"; text: null; source: null }
  | { status: "ready"; text: string; source: "ai" | "template" };

/**
 * Returns the narrative for a result. Behaviour:
 *
 *   1. While the API call is in flight, status is "loading" and the
 *      component should render a skeleton (no text yet).
 *   2. If /api/narrate returns an LLM-written narrative, that text is
 *      shown with source="ai".
 *   3. If the API call fails or returns no narrative (no key set,
 *      rate-limited, network error, timeout), the local template
 *      generator runs and that text is shown with source="template".
 *
 * The template never renders before the API has had a chance, so
 * users don't see a flash of generic text replaced a moment later
 * by the LLM output.
 */
export function useNarrative(result: DetectionResult): State {
  const [state, setState] = useState<State>({
    status: "loading",
    text: null,
    source: null,
  });

  useEffect(() => {
    let cancelled = false;
    setState({ status: "loading", text: null, source: null });

    const ctrl = new AbortController();
    const timeout = setTimeout(() => ctrl.abort(), 13_000);

    const fallback = () => {
      if (cancelled) return;
      setState({ status: "ready", text: narrate(result), source: "template" });
    };

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
          setState({ status: "ready", text: narrative, source: "ai" });
        } else {
          fallback();
        }
      })
      .catch(() => {
        fallback();
      })
      .finally(() => clearTimeout(timeout));

    return () => {
      cancelled = true;
      ctrl.abort();
      clearTimeout(timeout);
    };
    // Re-run when the result identity changes.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [result.kind, result.suspicion, result.verdict, result.signals.length]);

  return state;
}
