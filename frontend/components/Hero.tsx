"use client";

import Link from "next/link";
import { motion } from "framer-motion";
import { ArrowRight, ArrowDown } from "lucide-react";

export default function Hero() {
  return (
    <section className="relative">
      <div className="mx-auto max-w-5xl px-6 pt-28 pb-32 md:pt-36 md:pb-44">
        <motion.p
          initial={{ opacity: 0, y: 6 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
          className="text-xs uppercase tracking-[0.22em] text-mute"
        >
          Veritas — forensic examination of synthetic media
        </motion.p>

        <motion.h1
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.7, ease: [0.2, 0.8, 0.2, 1] }}
          className="font-display font-light tracking-tight text-ink mt-6"
          style={{ fontSize: "clamp(2.6rem, 6vw, 4.8rem)", lineHeight: 1.02 }}
        >
          Is it real?
          <br />
          <span className="italic text-smoke">Let&rsquo;s look carefully.</span>
        </motion.h1>

        <motion.p
          initial={{ opacity: 0, y: 6 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2, duration: 0.6 }}
          className="mt-8 max-w-xl text-[1.05rem] leading-[1.7] text-smoke"
        >
          A small, opinionated detector for AI-generated images, video, audio
          and text — with the working shown.
        </motion.p>

        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.4, duration: 0.5 }}
          className="mt-12 flex items-center gap-6"
        >
          <Link
            href="/detect"
            className="group inline-flex items-center gap-2 bg-ink text-white px-6 py-3 text-sm tracking-wide hover:bg-ember transition-colors"
          >
            Open the workspace
            <ArrowRight className="size-4 transition-transform group-hover:translate-x-0.5" />
          </Link>
          <a
            href="#how"
            className="inline-flex items-center gap-2 text-sm text-smoke hover:text-ink transition-colors"
          >
            How it works
            <ArrowDown className="size-3.5" />
          </a>
        </motion.div>
      </div>

      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.8, duration: 0.6 }}
        className="absolute inset-x-0 bottom-6 flex justify-center text-mute"
      >
        <div className="flex flex-col items-center gap-2">
          <span className="text-[10px] uppercase tracking-[0.3em]">Scroll</span>
          <div className="h-8 w-px bg-rule" />
        </div>
      </motion.div>
    </section>
  );
}
