"use client";

import { useState } from "react";
import Uploader from "@/components/Uploader";
import ResultPanel from "@/components/ResultPanel";
import { detectImage, type ImageResult } from "@/lib/api";

type Side = "a" | "b";
type Slot = { result: ImageResult | null; preview: string | null; loading: boolean; error: string | null };

const EMPTY: Slot = { result: null, preview: null, loading: false, error: null };

export default function ComparePage() {
  const [a, setA] = useState<Slot>(EMPTY);
  const [b, setB] = useState<Slot>(EMPTY);

  const setSide = (side: Side, patch: Partial<Slot>) =>
    side === "a" ? setA((s) => ({ ...s, ...patch })) : setB((s) => ({ ...s, ...patch }));

  async function run(side: Side, file: File) {
    if (side === "a" && a.preview) URL.revokeObjectURL(a.preview);
    if (side === "b" && b.preview) URL.revokeObjectURL(b.preview);
    const preview = URL.createObjectURL(file);
    setSide(side, { loading: true, error: null, result: null, preview });
    try {
      const r = await detectImage(file);
      setSide(side, { loading: false, result: r });
    } catch (e) {
      setSide(side, {
        loading: false,
        error: e instanceof Error ? e.message : "Something went wrong.",
      });
    }
  }

  return (
    <>
      <section className="mx-auto max-w-6xl px-6 py-12">
        <h1 className="font-display text-3xl">Compare</h1>
        <p className="mt-2 text-sm text-smoke max-w-prose">
          Place a suspect image next to a known-real reference. The same forensic
          signals run on both — divergence is what to look for.
        </p>

        <div className="mt-8 grid md:grid-cols-2 gap-6">
          {(["a", "b"] as Side[]).map((side) => {
            const slot = side === "a" ? a : b;
            return (
              <div key={side} className="border border-rule bg-paper">
                <div className="px-5 py-3 border-b border-rule flex items-center justify-between">
                  <span className="text-xs text-mute uppercase tracking-wide">
                    Image {side.toUpperCase()}
                  </span>
                  {slot.result && (
                    <span className="text-xs text-mute tabular-nums">
                      suspicion {Math.round(slot.result.suspicion * 100)}
                    </span>
                  )}
                </div>
                <div className="p-5">
                  <Uploader
                    kind="image"
                    loading={slot.loading}
                    onSubmit={(f) => run(side, f)}
                  />
                  {slot.error && (
                    <div className="mt-4 text-xs text-alert border border-alert/40 bg-alert/5 px-3 py-2">
                      {slot.error}
                    </div>
                  )}
                </div>
                {slot.result && (
                  <div className="border-t border-rule">
                    <ResultPanel result={slot.result} previewUrl={slot.preview} />
                  </div>
                )}
              </div>
            );
          })}
        </div>

        {a.result && b.result && (
          <div className="mt-8 border border-rule bg-paper px-5 py-4">
            <div className="text-xs text-mute mb-2">Δ suspicion</div>
            <div className="text-3xl tabular-nums">
              {Math.round((a.result.suspicion - b.result.suspicion) * 100)}
            </div>
            <p className="mt-1 text-xs text-mute">
              Positive means A reads more suspicious than B.
            </p>
          </div>
        )}
      </section>
    </>
  );
}
