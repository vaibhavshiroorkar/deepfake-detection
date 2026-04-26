"use client";

import { useMemo, useState } from "react";
import clsx from "clsx";

type ActiveLayer = "ela" | "noise";
type Layer = ActiveLayer | "off";

const LABELS: Record<ActiveLayer, { name: string; hint: string }> = {
  ela: {
    name: "ELA",
    hint: "Bright = regions whose recompression residue exceeds the rest of the frame.",
  },
  noise: {
    name: "Noise",
    hint: "Bright = patches whose high-frequency noise diverges from the median.",
  },
};

export default function HeatmapOverlay({
  base,
  heatmaps,
}: {
  base: string;
  heatmaps: { ela?: string; noise?: string };
}) {
  const available = useMemo(() => {
    const out: ActiveLayer[] = [];
    if (heatmaps.ela) out.push("ela");
    if (heatmaps.noise) out.push("noise");
    return out;
  }, [heatmaps]);

  const [layer, setLayer] = useState<Layer>(available[0] ?? "off");
  const [opacity, setOpacity] = useState(0.7);

  if (available.length === 0) return null;
  const overlayUrl = layer === "off" ? null : heatmaps[layer];

  return (
    <div className="border border-rule bg-bone/30">
      <div className="flex items-center justify-between gap-3 px-4 py-2.5 border-b border-rule">
        <div className="flex items-center gap-1">
          {available.map((id) => (
            <button
              key={id}
              onClick={() => setLayer(id)}
              className={clsx(
                "px-2.5 py-1 text-xs transition-colors",
                layer === id ? "bg-ink text-white" : "text-smoke hover:text-ink",
              )}
            >
              {LABELS[id].name}
            </button>
          ))}
          <button
            onClick={() => setLayer("off")}
            className={clsx(
              "px-2.5 py-1 text-xs transition-colors",
              layer === "off" ? "bg-ink text-white" : "text-smoke hover:text-ink",
            )}
          >
            Off
          </button>
        </div>
        {layer !== "off" && (
          <label className="flex items-center gap-2 text-xs text-mute">
            <span>opacity</span>
            <input
              type="range"
              min={0.1}
              max={1}
              step={0.05}
              value={opacity}
              onChange={(e) => setOpacity(parseFloat(e.target.value))}
              className="w-24 accent-ember"
            />
          </label>
        )}
      </div>

      <div className="bg-ink/5 flex justify-center">
        <div className="relative inline-block">
          <img
            src={base}
            alt="scanned"
            className="block max-h-[480px] w-auto"
          />
          {overlayUrl && (
            <img
              src={overlayUrl}
              alt={`${layer} heatmap`}
              className="absolute inset-0 w-full h-full pointer-events-none mix-blend-screen"
              style={{ opacity }}
            />
          )}
        </div>
      </div>

      {layer !== "off" && (
        <p className="px-4 py-2 text-[11px] text-mute border-t border-rule">
          {LABELS[layer].hint}
        </p>
      )}
    </div>
  );
}
