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
      "Two classifiers vote on whether the image was generated. If there are faces, we crop each one and run the same check. Camera physics and EXIF back things up when they can.",
  },
  {
    n: "02",
    Icon: Film,
    label: "Video",
    formats: "MP4 · WebM · MOV",
    note:
      "We sample frames along the timeline and run each one through the image pipeline. A separate pass watches for the frame-to-frame wobble face generators tend to leave behind.",
  },
  {
    n: "03",
    Icon: AudioLines,
    label: "Audio",
    formats: "WAV · MP3 · FLAC · OGG",
    note:
      "A wav2vec2 classifier handles the main call. Whisper features give a second read. Pitch, silence, and energy patterns flag what TTS and voice clones flatten.",
  },
  {
    n: "04",
    Icon: AlignLeft,
    label: "Text",
    formats: "any UTF-8 paragraph",
    note:
      "RoBERTa says whether it reads machine-written. GPT-2 measures how surprising the language is. Burstiness, rhythm, and LLM word tics round it out.",
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
            Plate 002 · Modalities
          </motion.span>
          <motion.span
            initial={{ opacity: 0 }}
            whileInView={{ opacity: 1 }}
            viewport={{ once: true, margin: "-120px" }}
            transition={{ duration: 0.5, delay: 0.1 }}
            className="running-head hidden sm:inline"
          >
            Same answer shape, four inputs
          </motion.span>
        </div>

        <motion.h2
          initial={{ opacity: 0, y: 12 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, margin: "-120px" }}
          transition={{ duration: 0.7, ease: EASE }}
          className="display-xl max-w-[16ch]"
        >
          Four kinds of input.
          <br />
          <span className="italic text-ink/70">One kind of answer.</span>
        </motion.h2>

        <motion.p
          initial={{ opacity: 0, y: 8 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, margin: "-120px" }}
          transition={{ duration: 0.6, delay: 0.1 }}
          className="body mt-8 md:mt-10 max-w-[52ch]"
        >
          Upload an image, a video, a voice clip, or paste some text. You
          always get back the same thing: a score, a plain verdict, and the
          list of signals behind it.
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
