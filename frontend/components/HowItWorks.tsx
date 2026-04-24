"use client";

import { motion } from "framer-motion";

const ITEMS = [
  {
    n: "01",
    title: "Image",
    body:
      "Two pretrained classifiers vote on synthesis: a Swin-v2 SDXL detector and an ensemble head trained on a different generative mix. MTCNN crops faces and re-runs the same classifiers per face. Camera-physics signals (focus uniformity, chromatic aberration, sensor noise, ELA) act as supporting evidence, weighted by whether EXIF identifies a real camera.",
  },
  {
    n: "02",
    title: "Video",
    body:
      "Frames sampled along the timeline each pass through the image pipeline. A temporal-flicker check looks for the per-frame wobble that face-generators leave behind. The verdict folds frame-level scores into a timeline chart, where spikes get weighted more than a uniformly-high average.",
  },
  {
    n: "03",
    title: "Audio",
    body:
      "A wav2vec2 binary classifier pairs with a Whisper encoder. The Whisper encoder gives learned spectral features read by an adjacent-frame cosine heuristic. Classical signals: pitch variability, silence-floor cleanliness, energy-envelope rhythm. TTS and voice clones tend to flatten all three at once.",
  },
  {
    n: "04",
    title: "Text",
    body:
      "RoBERTa binary discriminator plus GPT-2 perplexity scoring. One is a dedicated detector, the other measures language-model surprise. Supporting: sentence-length burstiness, lexical rhythm, phrase-tic hit rate, punctuation signatures over-represented in AI writing.",
  },
];

const EASE = [0.2, 0.8, 0.2, 1] as const;

export default function HowItWorks() {
  return (
    <section id="method" className="section-screen border-t border-rule bg-paper">
      <div className="page-frame flex-1 flex flex-col justify-center py-24">
        <div className="flex items-baseline justify-between mb-10">
          <motion.span
            initial={{ opacity: 0 }}
            whileInView={{ opacity: 1 }}
            viewport={{ once: true, margin: "-120px" }}
            transition={{ duration: 0.5 }}
            className="running-head"
          >
            Plate 003 · Method
          </motion.span>
          <motion.span
            initial={{ opacity: 0 }}
            whileInView={{ opacity: 1 }}
            viewport={{ once: true, margin: "-120px" }}
            transition={{ duration: 0.5, delay: 0.1 }}
            className="running-head hidden sm:inline"
          >
            Agreement across channels
          </motion.span>
        </div>

        <motion.h2
          initial={{ opacity: 0, y: 12 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, margin: "-120px" }}
          transition={{ duration: 0.7, ease: EASE }}
          className="display-xl max-w-[18ch]"
        >
          No single signal
          <br />
          <span className="italic text-ink/70">is decisive.</span>
        </motion.h2>

        <motion.p
          initial={{ opacity: 0, y: 8 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, margin: "-120px" }}
          transition={{ duration: 0.6, delay: 0.1 }}
          className="body mt-8 md:mt-10 max-w-[56ch]"
        >
          The verdict is the agreement across independent channels. A learned
          classifier reads the piece as a whole, and targeted forensic
          checks look for the artefacts each medium tends to leave behind.
        </motion.p>

        <div className="mt-14 md:mt-20 grid md:grid-cols-2 gap-x-16 md:gap-x-20 gap-y-14">
          {ITEMS.map((item, i) => (
            <motion.article
              key={item.title}
              initial={{ opacity: 0, y: 16 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true, margin: "-80px" }}
              transition={{ duration: 0.6, delay: i * 0.08, ease: EASE }}
              className="relative"
            >
              <div className="flex items-center gap-4 mb-4">
                <span className="font-mono text-sm text-ember tracking-[0.2em]">
                  {item.n}
                </span>
                <span className="h-px bg-rule flex-1" />
              </div>
              <h3 className="display-md mb-4">{item.title}</h3>
              <p className="body-sm text-smoke max-w-[48ch]">{item.body}</p>
            </motion.article>
          ))}
        </div>
      </div>
    </section>
  );
}
