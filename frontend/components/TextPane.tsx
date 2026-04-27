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
      "Spent most of Tuesday trying to fix the porch light. Thought it was the bulb, wasn't the bulb. Turned out the wire in the fixture had pulled loose from years of slamming the screen door. Dad used to just tape it back. I soldered it. Probably overkill but it felt good to sit on the step and actually finish something small.",
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
  const ok = text.trim().length >= 1 && !loading;

  return (
    <div>
      <textarea
        value={text}
        onChange={(e) => setText(e.target.value)}
        placeholder="Paste any text. Longer gives a more reliable reading."
        rows={10}
        className="w-full resize-y bg-bone/30 border border-rule p-5 text-[1.05rem] leading-[1.7] text-ink placeholder:text-mute focus:outline-none focus:border-ink transition-colors"
      />

      <div className="mt-5 flex flex-wrap items-center gap-x-6 gap-y-3">
        <button
          onClick={() => ok && onSubmit(text)}
          disabled={!ok}
          className={clsx(
            "inline-flex items-center gap-2.5 px-6 py-3 text-sm tracking-wide transition-colors border",
            ok
              ? "bg-ink text-paper border-ink hover:bg-ember hover:border-ember"
              : "bg-rule/50 text-mute border-rule cursor-not-allowed",
          )}
        >
          {loading ? (
            <>
              <span className="size-1.5 rounded-full bg-paper pulse-soft" />
              Reading through it
            </>
          ) : (
            <>
              Check it
              <span aria-hidden>→</span>
            </>
          )}
        </button>

        <span className="font-mono text-xs text-mute tracking-wider uppercase">
          {wordCount} {wordCount === 1 ? "word" : "words"}
        </span>

        <div className="ml-auto flex items-center gap-4 text-sm">
          <span className="running-head">Sample:</span>
          {SAMPLES.map((s) => (
            <button
              key={s.label}
              onClick={() => setText(s.body)}
              className="text-smoke hover:text-ember border-b border-dotted border-mute hover:border-ember pb-0.5 transition-colors"
            >
              {s.label}
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}
