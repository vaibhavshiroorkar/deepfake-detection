"use client";

import { useRouter, useSearchParams } from "next/navigation";
import { useEffect, useState } from "react";
import { Search, X } from "lucide-react";
import clsx from "clsx";

const KINDS = [
  { value: "", label: "All" },
  { value: "image", label: "Image" },
  { value: "video", label: "Video" },
  { value: "audio", label: "Audio" },
  { value: "text", label: "Text" },
];

const VERDICTS = [
  { value: "", label: "Any" },
  { value: "authentic", label: "Authentic" },
  { value: "inconclusive", label: "Inconclusive" },
  { value: "suspicious", label: "Suspicious" },
  { value: "manipulated", label: "Manipulated" },
];

export default function HistoryFilters() {
  const router = useRouter();
  const params = useSearchParams();

  const [q, setQ] = useState(params.get("q") ?? "");
  const kind = params.get("kind") ?? "";
  const verdict = params.get("verdict") ?? "";

  // Debounced commit of the text query to the URL.
  useEffect(() => {
    const current = params.get("q") ?? "";
    if (q === current) return;
    const handle = setTimeout(() => {
      const next = new URLSearchParams(params.toString());
      if (q) next.set("q", q);
      else next.delete("q");
      router.replace(`/history?${next.toString()}`, { scroll: false });
    }, 300);
    return () => clearTimeout(handle);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [q]);

  function setParam(key: string, value: string) {
    const next = new URLSearchParams(params.toString());
    if (value) next.set(key, value);
    else next.delete(key);
    router.replace(`/history?${next.toString()}`, { scroll: false });
  }

  function clearAll() {
    setQ("");
    router.replace("/history", { scroll: false });
  }

  const hasFilters = Boolean(kind || verdict || q);

  return (
    <div className="mb-6 border border-rule bg-paper">
      <div className="px-4 py-3 flex flex-wrap items-center gap-3">
        <div className="relative flex-1 min-w-[200px]">
          <Search className="absolute left-2.5 top-1/2 -translate-y-1/2 size-3.5 text-mute" />
          <input
            type="search"
            value={q}
            onChange={(e) => setQ(e.target.value)}
            placeholder="Search filenames"
            className="w-full pl-8 pr-3 py-1.5 text-sm bg-bone/40 border border-transparent focus:border-ink focus:outline-none"
          />
        </div>
        <FilterGroup
          label="Kind"
          options={KINDS}
          active={kind}
          onSelect={(v) => setParam("kind", v)}
        />
        <FilterGroup
          label="Verdict"
          options={VERDICTS}
          active={verdict}
          onSelect={(v) => setParam("verdict", v)}
        />
        {hasFilters && (
          <button
            onClick={clearAll}
            className="inline-flex items-center gap-1 text-xs text-smoke hover:text-ink"
          >
            <X className="size-3" /> Clear
          </button>
        )}
      </div>
    </div>
  );
}

function FilterGroup({
  label,
  options,
  active,
  onSelect,
}: {
  label: string;
  options: { value: string; label: string }[];
  active: string;
  onSelect: (value: string) => void;
}) {
  return (
    <div className="flex items-center gap-1.5">
      <span className="font-mono text-[10px] tracking-widest uppercase text-mute mr-0.5">
        {label}
      </span>
      {options.map((o) => (
        <button
          key={o.value || "__all__"}
          onClick={() => onSelect(o.value)}
          className={clsx(
            "px-2 py-1 text-xs transition-colors border",
            active === o.value
              ? "bg-ink text-paper border-ink"
              : "bg-transparent text-smoke border-transparent hover:border-rule hover:text-ink",
          )}
        >
          {o.label}
        </button>
      ))}
    </div>
  );
}
