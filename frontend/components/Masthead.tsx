"use client";

import { motion } from "framer-motion";

export default function Masthead() {
  const today = new Date().toLocaleDateString("en-US", {
    weekday: "long",
    year: "numeric",
    month: "long",
    day: "numeric",
  });

  return (
    <header className="relative border-b border-rule">
      <div className="mx-auto max-w-[1400px] px-6 pt-6 pb-4">
        <div className="flex items-end justify-between gap-6 text-xs font-mono uppercase tracking-[0.18em] text-smoke">
          <span className="hidden md:inline">Vol. I &nbsp;·&nbsp; No. 04</span>
          <span className="text-mute">{today}</span>
          <span className="hidden md:inline">Issued under peer review</span>
        </div>

        <div className="mt-3 flex items-baseline justify-between">
          <motion.h1
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.7, ease: [0.2, 0.8, 0.2, 1] }}
            className="font-display font-black tracking-tightest text-ink"
            style={{ fontSize: "clamp(2.4rem, 6vw, 5.25rem)", lineHeight: 0.9 }}
          >
            Veritas<span className="text-ember">.</span>
          </motion.h1>
          <nav className="hidden md:flex items-center gap-7 pb-3 text-sm text-smoke">
            <a className="hover:text-ink transition-colors" href="#console">The Desk</a>
            <a className="hover:text-ink transition-colors" href="#how">Method</a>
            <a className="hover:text-ink transition-colors" href="#caveats">Caveats</a>
            <a className="hover:text-ink transition-colors" href="#colophon">Colophon</a>
          </nav>
        </div>

        <div className="mt-3 flex items-center gap-4 text-xs font-mono uppercase tracking-[0.2em] text-mute">
          <span className="inline-block size-1.5 rounded-full bg-forest pulse-soft" />
          <span>A careful second opinion on synthetic media</span>
        </div>
      </div>
      <div className="hairline" />
    </header>
  );
}
