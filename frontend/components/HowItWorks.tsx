"use client";

const ITEMS = [
  {
    title: "Text",
    body:
      "Sentence burstiness, lexical rhythm, phrase tics, and punctuation patterns statistically over-represented in AI writing.",
  },
  {
    title: "Images",
    body:
      "Error-level analysis, noise consistency, frequency signatures, face-boundary continuity, and EXIF metadata.",
  },
  {
    title: "Audio",
    body:
      "Pitch variability, silence noise floor, high-frequency roll-off, and energy rhythm — the messy signatures a real microphone leaves.",
  },
  {
    title: "Video",
    body:
      "Frame-by-frame forensics plus a temporal flicker check for the per-frame wobble that generators leave behind.",
  },
];

export default function HowItWorks() {
  return (
    <section id="how" className="border-t border-rule bg-paper">
      <div className="mx-auto max-w-5xl px-6 py-16">
        <h2
          className="font-display tracking-tight text-center"
          style={{ fontSize: "clamp(1.6rem, 3vw, 2.2rem)" }}
        >
          How it works
        </h2>
        <p className="mt-3 max-w-lg mx-auto text-center text-sm text-smoke leading-relaxed">
          No single signal is decisive. The verdict is the agreement across independent channels.
        </p>

        <div className="mt-12 grid sm:grid-cols-2 lg:grid-cols-4 gap-8">
          {ITEMS.map((item, i) => (
            <div key={item.title}>
              <div className="text-xs text-ember mb-2">0{i + 1}</div>
              <h3 className="font-display text-xl tracking-tight">{item.title}</h3>
              <p className="mt-2 text-sm text-smoke leading-[1.6]">{item.body}</p>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}
