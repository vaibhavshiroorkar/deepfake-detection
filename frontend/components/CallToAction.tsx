"use client";

import Link from "next/link";
import { motion } from "framer-motion";
import { ArrowUpRight } from "lucide-react";

const EASE = [0.2, 0.8, 0.2, 1] as const;

export default function CallToAction() {
  return (
    <section className="section-screen border-t border-rule bg-ink text-paper overflow-hidden relative">
      <div className="page-frame flex-1 flex flex-col justify-center py-28 relative z-10">
        <motion.span
          initial={{ opacity: 0 }}
          whileInView={{ opacity: 1 }}
          viewport={{ once: true, margin: "-120px" }}
          transition={{ duration: 0.5 }}
          className="running-head text-mute mb-10"
        >
          Plate 005 · Invitation
        </motion.span>

        <motion.h2
          initial={{ opacity: 0, y: 16 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, margin: "-120px" }}
          transition={{ duration: 0.9, ease: EASE }}
          className="display-hero max-w-[22ch]"
        >
          Bring the
          <br />
          <span className="italic text-ember">questionable</span> thing.
        </motion.h2>

        <motion.p
          initial={{ opacity: 0, y: 10 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, margin: "-120px" }}
          transition={{ duration: 0.7, delay: 0.2 }}
          className="body-lead mt-12 max-w-[44ch] text-paper/80"
        >
          The workspace is one click away. No account required to try it.
          Uploads are processed in memory. Nothing is retained.
        </motion.p>

        <motion.div
          initial={{ opacity: 0, y: 10 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, margin: "-120px" }}
          transition={{ duration: 0.6, delay: 0.35 }}
          className="mt-16 flex flex-wrap items-center gap-x-10 gap-y-5"
        >
          <Link
            href="/detect"
            className="group inline-flex items-center gap-3 bg-ember text-paper px-8 py-4 text-base hover:bg-paper hover:text-ink transition-colors"
          >
            Open the workspace
            <ArrowUpRight className="size-5 transition-transform group-hover:-translate-y-0.5 group-hover:translate-x-0.5" />
          </Link>
          <Link
            href="/calibration"
            className="text-paper/70 hover:text-paper transition-colors border-b border-paper/30 hover:border-paper pb-0.5"
          >
            Or read how it&rsquo;s calibrated
          </Link>
        </motion.div>
      </div>

      {/* Decorative grid of tick marks — like a ruler on evidence film */}
      <div
        aria-hidden
        className="absolute inset-y-0 right-0 w-24 hidden md:flex flex-col justify-between py-24 opacity-30"
      >
        {Array.from({ length: 24 }).map((_, i) => (
          <span
            key={i}
            className="block h-px bg-paper"
            style={{ width: i % 5 === 0 ? "2.5rem" : "1.25rem", alignSelf: "flex-end" }}
          />
        ))}
      </div>
    </section>
  );
}
