"use client";

import { useState } from "react";
import clsx from "clsx";

const SAMPLES: { label: string; body: string }[] = [
  {
    label: "AI-ish",
    body:
      "In today's ever-evolving landscape, our organization is proud to announce a groundbreaking initiative that will leverage artificial intelligence to unlock unprecedented value for our stakeholders. Furthermore, this seamless integration represents a testament to our commitment to innovation. Moreover, it is important to note that the robust framework we have developed plays a crucial role in navigating the complexities of the modern marketplace.",
  },
  {
    label: "Human-ish",
    body:
      "Spent most of Tuesday trying to fix the porch light — thought it was the bulb, wasn't the bulb. Turned out the wire in the fixture had pulled loose from years of slamming the screen door. Dad used to just tape it back. I soldered it. Probably overkill but it felt good to sit on the step and actually finish something small.",
  },
];

export default function TextPane({
  loading,
  onSubmit,
}: {
  loading: boolean;
  onSubmit: (t: string) => void | Promise<void>;
}) {
  const [text, setText] = useState("");
  const wordCount = text.trim() ? text.trim().split(/\s+/).length : 0;
  const ok = text.trim().length >= 40 && !loading;

  return (
    <div>
      <textarea
        value={text}
        onChange={(e) => setText(e.target.value)}
        placeholder="Paste a paragraph. The longer, the steadier the reading — 40 words minimum."
        rows={8}
        className="w-full resize-y bg-bone/40 border border-rule p-4 text-[1rem] leading-[1.6] text-ink placeholder:text-mute focus:outline-none focus:border-ember"
      />

      <div className="mt-3 flex flex-wrap items-center gap-3">
        <button
          onClick={() => ok && onSubmit(text)}
          disabled={!ok}
          className={clsx(
            "inline-flex items-center gap-2 px-4 py-2 text-sm transition-colors",
            ok ? "bg-ink text-white hover:bg-ember" : "bg-rule text-mute cursor-not-allowed",
          )}
        >
          {loading ? (
            <>
              <span className="size-1.5 rounded-full bg-white pulse-soft" />
              Reading
            </>
          ) : (
            <>Examine<span>→</span></>
          )}
        </button>

        <span className="text-xs text-mute">{wordCount} words</span>

        <div className="ml-auto flex items-center gap-3 text-xs">
          <span className="text-mute">Try:</span>
          {SAMPLES.map((s) => (
            <button
              key={s.label}
              onClick={() => setText(s.body)}
              className="text-smoke hover:text-ember underline decoration-dotted underline-offset-4"
            >
              {s.label}
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}
