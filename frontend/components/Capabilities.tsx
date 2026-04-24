"use client";

import { motion } from "framer-motion";
import { Image as ImageIcon, Film, AudioLines, AlignLeft } from "lucide-react";

const ITEMS = [
  {
    n: "01",
    Icon: ImageIcon,
    label: "Image",
    formats: "JPEG · PNG · WEBP",
    note:
      "Two pretrained classifiers vote on synthesis. MTCNN crops faces and re-runs them. Camera-physics signals and EXIF act as supporting evidence.",
  },
  {
    n: "02",
    Icon: Film,
    label: "Video",
    formats: "MP4 · WebM · MOV",
    note:
      "Sampled frames run through the image pipeline. A temporal-flicker check looks for the per-frame wobble face-generators leave behind.",
  },
  {
    n: "03",
    Icon: AudioLines,
    label: "Audio",
    formats: "WAV · MP3 · FLAC · OGG",
    note:
      "A wav2vec2 classifier pairs with Whisper encoder features. Classical pitch, silence-floor, and energy-rhythm signals corroborate.",
  },
  {
    n: "04",
    Icon: AlignLeft,
    label: "Text",
    formats: "any UTF-8 paragraph",
    note:
      "RoBERTa discriminator and GPT-2 perplexity together. Burstiness, lexical rhythm, and LLM tics provide corroborating reads.",
  },
];

const EASE = [0.2, 0.8, 0.2, 1] as const;

export default function Capabilities() {
  return (
    <section id="capabilities" className="section-screen border-t border-rule">
      <div className="page-frame flex-1 flex flex-col justify-center py-24">
        <div className="flex items-baseline justify-between mb-10">
          <motion.span
            initial={{ opacity: 0 }}
            whileInView={{ opacity: 1 }}
            viewport={{ once: true, margin: "-120px" }}
            transition={{ duration: 0.5 }}
            className="running-head"
          >
            Plate 002 — Modalities
          </motion.span>
          <motion.span
            initial={{ opacity: 0 }}
            whileInView={{ opacity: 1 }}
            viewport={{ once: true, margin: "-120px" }}
            transition={{ duration: 0.5, delay: 0.1 }}
            className="running-head hidden sm:inline"
          >
            Four in, one out
          </motion.span>
        </div>

        <motion.h2
          initial={{ opacity: 0, y: 12 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, margin: "-120px" }}
          transition={{ duration: 0.7, ease: EASE }}
          className="display-xl max-w-[16ch]"
        >
          Four modalities,
          <br />
          <span className="italic text-smoke">one verdict format.</span>
        </motion.h2>

        <motion.p
          initial={{ opacity: 0, y: 8 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, margin: "-120px" }}
          transition={{ duration: 0.6, delay: 0.1 }}
          className="body mt-8 md:mt-10 max-w-[52ch]"
        >
          Drop an image, a video, a voice note, or a paragraph. The output
          schema is the same: a suspicion score, a plain-language verdict,
          and every signal that contributed to it.
        </motion.p>

        <div className="mt-14 md:mt-20 grid grid-cols-1 md:grid-cols-2 gap-px bg-rule border-y border-rule">
          {ITEMS.map((it, i) => (
            <motion.article
              key={it.label}
              initial={{ opacity: 0, y: 16 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true, margin: "-80px" }}
              transition={{ duration: 0.6, delay: i * 0.07, ease: EASE }}
              className="group relative bg-paper px-8 md:px-10 py-10 md:py-12 flex flex-col gap-5 hover:bg-bone transition-colors"
            >
              <div className="flex items-center justify-between">
                <span className="font-mono text-xs text-ember tracking-[0.2em]">
                  {it.n}
                </span>
                <it.Icon className="size-5 text-ink" strokeWidth={1.4} />
              </div>
              <h3 className="display-md">{it.label}</h3>
              <p className="font-mono text-xs text-mute tracking-[0.1em]">
                {it.formats}
              </p>
              <p className="body-sm text-smoke max-w-[44ch]">{it.note}</p>
            </motion.article>
          ))}
        </div>
      </div>
    </section>
  );
}
