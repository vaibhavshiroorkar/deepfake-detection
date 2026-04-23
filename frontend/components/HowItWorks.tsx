"use client";

import { motion } from "framer-motion";

const ITEMS = [
  {
    n: "01",
    title: "Image",
    body:
      "A pretrained Swin-v2 classifier returns P(AI). MTCNN crops faces and re-runs the same head per face. Camera-physics signals (focus uniformity, chromatic aberration, sensor noise, ELA) act as supporting evidence, weighted by whether EXIF identifies a real camera.",
  },
  {
    n: "02",
    title: "Video",
    body:
      "Frames are sampled along the timeline, each one passes through the image pipeline, and a temporal-flicker check looks for the per-frame wobble that face-generators leave behind.",
  },
  {
    n: "03",
    title: "Audio",
    body:
      "The Whisper-base encoder produces a learned spectral profile. Adjacent-frame cosine and per-dimension variance are paired with classical pitch, silence-floor, and energy-rhythm heuristics.",
  },
  {
    n: "04",
    title: "Text",
    body:
      "Sentence burstiness, lexical rhythm, phrase tics, and punctuation patterns over-represented in AI writing.",
  },
];

export default function HowItWorks() {
  return (
    <section id="how" className="border-t border-rule bg-paper">
      <div className="mx-auto max-w-5xl px-6 py-24 md:py-32">
        <motion.p
          initial={{ opacity: 0, y: 8 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, margin: "-80px" }}
          transition={{ duration: 0.5 }}
          className="text-xs uppercase tracking-[0.22em] text-mute"
        >
          The method
        </motion.p>
        <motion.h2
          initial={{ opacity: 0, y: 10 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, margin: "-80px" }}
          transition={{ duration: 0.6, delay: 0.05 }}
          className="font-display tracking-tight mt-3"
          style={{ fontSize: "clamp(1.8rem, 3.6vw, 2.6rem)", lineHeight: 1.1 }}
        >
          No single signal is decisive.
        </motion.h2>
        <motion.p
          initial={{ opacity: 0, y: 6 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, margin: "-80px" }}
          transition={{ duration: 0.5, delay: 0.1 }}
          className="mt-4 max-w-xl text-sm text-smoke leading-[1.7]"
        >
          The verdict is the agreement across independent channels — a learned
          classifier reading the picture as a whole, plus targeted forensic
          checks for the artefacts each medium tends to leave behind.
        </motion.p>

        <div className="mt-16 grid md:grid-cols-2 gap-12 md:gap-x-16 md:gap-y-14">
          {ITEMS.map((item, i) => (
            <motion.div
              key={item.title}
              initial={{ opacity: 0, y: 12 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true, margin: "-60px" }}
              transition={{ duration: 0.5, delay: i * 0.06 }}
            >
              <div className="text-xs text-ember tracking-[0.2em]">{item.n}</div>
              <h3 className="font-display text-2xl tracking-tight mt-2">
                {item.title}
              </h3>
              <p className="mt-3 text-sm text-smoke leading-[1.75]">
                {item.body}
              </p>
            </motion.div>
          ))}
        </div>
      </div>
    </section>
  );
}
