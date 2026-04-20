"use client";

import { useState } from "react";
import clsx from "clsx";
import { ClipboardPaste } from "lucide-react";

const SAMPLES: { label: string; body: string }[] = [
  {
    label: "Press release",
    body:
      "In today's ever-evolving landscape, our organization is proud to announce a groundbreaking initiative that will leverage the power of artificial intelligence to unlock unprecedented value for our stakeholders. Furthermore, this seamless integration represents a testament to our commitment to innovation. Moreover, it is important to note that the robust framework we have developed plays a crucial role in navigating the complexities of the modern marketplace.",
  },
  {
    label: "Personal note",
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
  const tooShort = text.trim().length > 0 && text.trim().length < 40;
  const ok = text.trim().length >= 40 && !loading;

  return (
    <div>
      <div className="relative">
        <textarea
          value={text}
          onChange={(e) => setText(e.target.value)}
          placeholder="Paste a paragraph here. The longer the passage, the steadier the reading — forty words is the floor, a few hundred is ideal."
          rows={10}
          className="w-full resize-y bg-bone/40 border border-rule p-5 font-serif text-[1.02rem] leading-[1.7] text-ink placeholder:text-mute focus:outline-none focus:border-ember"
          style={{ fontFamily: "Fraunces, serif" }}
        />
        <div className="absolute bottom-3 right-3 flex items-center gap-3 font-mono text-[10px] uppercase tracking-[0.2em] text-mute bg-paper/80 px-2 py-1">
          <span>{text.length} chars</span>
          <span className="text-rule">·</span>
          <span>{wordCount} words</span>
        </div>
      </div>

      <div className="mt-4 flex flex-wrap items-center gap-3">
        <button
          onClick={() => ok && onSubmit(text)}
          disabled={!ok}
          className={clsx(
            "inline-flex items-center gap-3 px-5 py-2.5 font-mono text-xs uppercase tracking-[0.2em] transition-colors",
            ok
              ? "bg-ink text-paper hover:bg-ember"
              : "bg-rule text-mute cursor-not-allowed",
          )}
        >
          {loading ? (
            <>
              <span className="size-2 rounded-full bg-paper pulse-soft" />
              Reading
            </>
          ) : (
            <>
              Examine the prose
              <span>→</span>
            </>
          )}
        </button>

        <div className="ml-auto flex items-center gap-2 text-xs text-smoke">
          <ClipboardPaste className="size-3.5 text-mute" />
          <span className="font-mono text-[10px] uppercase tracking-[0.2em] text-mute">
            Try a sample:
          </span>
          {SAMPLES.map((s) => (
            <button
              key={s.label}
              onClick={() => setText(s.body)}
              className="font-mono text-[11px] uppercase tracking-[0.18em] text-smoke hover:text-ember underline decoration-dotted underline-offset-4"
            >
              {s.label}
            </button>
          ))}
        </div>
      </div>

      {tooShort && (
        <p className="mt-3 font-mono text-[11px] text-alert">
          Add a few more sentences — short passages are too noisy to judge.
        </p>
      )}
    </div>
  );
}
