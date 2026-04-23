"use client";

import { motion } from "framer-motion";
import { Image as ImageIcon, Film, AudioLines, AlignLeft } from "lucide-react";

const ITEMS = [
  { Icon: ImageIcon, label: "Image", note: "JPEG · PNG · WEBP" },
  { Icon: Film, label: "Video", note: "MP4 · WebM · MOV" },
  { Icon: AudioLines, label: "Audio", note: "WAV · MP3 · FLAC · OGG" },
  { Icon: AlignLeft, label: "Text", note: "any UTF-8 paragraph" },
];

export default function Capabilities() {
  return (
    <section className="border-t border-rule">
      <div className="mx-auto max-w-5xl px-6 py-24">
        <motion.p
          initial={{ opacity: 0, y: 6 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, margin: "-60px" }}
          transition={{ duration: 0.5 }}
          className="text-xs uppercase tracking-[0.22em] text-mute"
        >
          What it accepts
        </motion.p>
        <motion.h2
          initial={{ opacity: 0, y: 10 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, margin: "-60px" }}
          transition={{ duration: 0.5, delay: 0.05 }}
          className="font-display tracking-tight mt-3"
          style={{ fontSize: "clamp(1.6rem, 3vw, 2.2rem)", lineHeight: 1.1 }}
        >
          Four modalities, one verdict format.
        </motion.h2>

        <div className="mt-12 grid grid-cols-2 md:grid-cols-4 gap-px bg-rule border border-rule">
          {ITEMS.map((it, i) => (
            <motion.div
              key={it.label}
              initial={{ opacity: 0, y: 8 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true, margin: "-40px" }}
              transition={{ duration: 0.4, delay: i * 0.05 }}
              className="bg-paper px-6 py-10 flex flex-col items-start gap-4"
            >
              <it.Icon className="size-5 text-ember" strokeWidth={1.5} />
              <div>
                <div className="font-display text-lg tracking-tight">{it.label}</div>
                <div className="mt-1 text-xs text-mute font-mono">{it.note}</div>
              </div>
            </motion.div>
          ))}
        </div>
      </div>
    </section>
  );
}
