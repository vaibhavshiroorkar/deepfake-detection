"use client";

import Link from "next/link";
import { motion } from "framer-motion";
import { ArrowUpRight } from "lucide-react";

const EASE = [0.2, 0.8, 0.2, 1] as const;

export default function Hero() {
  return (
    <section className="section-screen">
      <div className="page-frame flex-1 flex flex-col justify-center py-24">
        <motion.div
          initial={{ opacity: 0, y: 6 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
          className="flex items-baseline justify-between mb-10 md:mb-16"
        >
          <span className="running-head">Plate 001 · Introduction</span>
          <span className="running-head hidden sm:inline">
            A careful look at synthetic media
          </span>
        </motion.div>

        <motion.h1
          initial={{ opacity: 0, y: 16 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.9, ease: EASE }}
          className="display-hero"
        >
          Is it real?
          <br />
          <span className="italic text-ink/70" style={{ fontWeight: 400 }}>
            Let&rsquo;s look.
          </span>
        </motion.h1>

        <motion.p
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.3, duration: 0.7, ease: EASE }}
          className="body-lead mt-10 md:mt-14 max-w-[44ch]"
        >
          Veritas runs images, video, clips, and text through a handful
          of detectors and shows you what each one saw. When the signals
          agree, you get a clear answer. When they don&rsquo;t, it says
          so honestly instead of guessing.
        </motion.p>

        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.55, duration: 0.6 }}
          className="mt-12 md:mt-16 flex flex-wrap items-center gap-x-8 gap-y-4"
        >
          <Link href="/detect" className="btn group">
            Open the workspace
            <ArrowUpRight className="size-4 transition-transform group-hover:-translate-y-0.5 group-hover:translate-x-0.5" />
          </Link>
          <Link
            href="/method"
            className="inline-flex items-center gap-2 text-[0.95rem] text-smoke hover:text-ink transition-colors border-b border-rule hover:border-ink pb-0.5"
          >
            How it works
            <ArrowUpRight className="size-3.5" />
          </Link>
        </motion.div>
      </div>

      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 1, duration: 0.6 }}
        className="page-frame pb-8 flex items-end justify-between"
      >
        <div className="running-head hidden sm:block">
          Image · Video · Audio · Text
        </div>
        <div className="flex items-center gap-3 running-head">
          <span>Scroll</span>
          <span className="h-10 w-px bg-ink inline-block" />
        </div>
      </motion.div>
    </section>
  );
}
