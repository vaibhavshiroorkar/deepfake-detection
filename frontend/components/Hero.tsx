"use client";

import { motion } from "framer-motion";

export default function Hero() {
  return (
    <section className="mx-auto max-w-5xl px-6 pt-20 pb-16 text-center">
      <motion.h1
        initial={{ opacity: 0, y: 8 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6, ease: [0.2, 0.8, 0.2, 1] }}
        className="font-display font-light tracking-tight text-ink"
        style={{ fontSize: "clamp(2.4rem, 5.2vw, 4rem)", lineHeight: 1.05 }}
      >
        Is it real?
        <br />
        <span className="italic text-smoke">Let's take a careful look.</span>
      </motion.h1>

      <motion.p
        initial={{ opacity: 0, y: 6 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.15, duration: 0.6 }}
        className="mx-auto mt-6 max-w-xl text-[1.05rem] leading-[1.6] text-smoke"
      >
        Upload an image, clip, or paragraph. Veritas examines it through
        several forensic channels and returns a plain-language reading —
        with the working shown.
      </motion.p>

      <motion.a
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.3, duration: 0.5 }}
        href="#console"
        className="inline-flex items-center gap-2 mt-10 bg-ink text-white px-6 py-3 text-sm tracking-wide hover:bg-ember transition-colors"
      >
        Start examining
        <span>→</span>
      </motion.a>
    </section>
  );
}
