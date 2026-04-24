"use client";

import { motion } from "framer-motion";

const ITEMS = [
  {
    n: "01",
    title: "Image",
    body:
      "Two pretrained classifiers weigh in on the whole image. MTCNN finds faces, if any, and we run the same check on each crop. Then we look at camera physics: focus uniformity, chromatic aberration, sensor noise, error-level analysis. If EXIF matches a real camera, those signals count for more.",
  },
  {
    n: "02",
    title: "Video",
    body:
      "We sample frames along the timeline and run each one through the image pipeline. A separate check looks for the frame-to-frame wobble face generators tend to leave behind. The timeline chart shows where suspicion spikes. Spikes count more than a flat, evenly-high average.",
  },
  {
    n: "03",
    title: "Audio",
    body:
      "A wav2vec2 classifier handles the main call. Whisper features give a second, learned read. On the classical side we watch pitch variability, silence-floor cleanliness, and energy-envelope rhythm. TTS and voice clones tend to flatten all three at once, which is a tell.",
  },
  {
    n: "04",
    title: "Text",
    body:
      "RoBERTa says whether the writing reads machine-made. GPT-2 measures how surprising the language is. Then we look at sentence-length burstiness, lexical rhythm, scaffolding-phrase frequency, and the punctuation patterns that show up more often in AI writing.",
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
            Agreement between checks
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
          <span className="italic text-ink/70">decides it.</span>
        </motion.h2>

        <motion.p
          initial={{ opacity: 0, y: 8 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, margin: "-120px" }}
          transition={{ duration: 0.6, delay: 0.1 }}
          className="body mt-8 md:mt-10 max-w-[56ch]"
        >
          A verdict is the agreement between independent checks. One
          classifier reads the thing as a whole. Smaller forensic passes
          look for the specific fingerprints each medium tends to leave
          behind. If they line up, we say so. If they don&rsquo;t, we say
          that too.
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
