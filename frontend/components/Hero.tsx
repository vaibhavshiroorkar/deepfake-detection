"use client";

import { motion } from "framer-motion";

export default function Hero() {
  return (
    <section className="relative">
      <div className="mx-auto max-w-[1400px] px-6 py-12 md:py-20 grid md:grid-cols-12 gap-10 md:gap-8">
        <div className="md:col-span-7">
          <motion.p
            initial={{ opacity: 0, y: 6 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.1, duration: 0.6 }}
            className="font-mono text-[11px] uppercase tracking-[0.28em] text-ember mb-5"
          >
            The lead —  On trusting what we see
          </motion.p>

          <motion.h2
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.15, duration: 0.8, ease: [0.2, 0.8, 0.2, 1] }}
            className="font-display text-ink font-light tracking-tighter"
            style={{ fontSize: "clamp(2.2rem, 5.2vw, 4.4rem)", lineHeight: 1.02 }}
          >
            Seeing is no longer believing.
            <br />
            <span className="italic font-normal text-smoke">
              Careful looking still is.
            </span>
          </motion.h2>

          <motion.p
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.3, duration: 0.7 }}
            className="mt-7 max-w-[54ch] text-[1.05rem] leading-[1.65] text-smoke"
          >
            Upload an image, a clip, or a paragraph. Veritas reads the grain,
            listens for the seams, and measures the rhythm of the prose —
            then reports what it notices, in plain language, with its working
            shown. It will not decide for you. It will help you look harder.
          </motion.p>

          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.5, duration: 0.6 }}
            className="mt-10 flex items-center gap-6"
          >
            <a
              href="#console"
              className="group inline-flex items-center gap-3 bg-ink text-paper px-6 py-3.5 font-mono text-xs uppercase tracking-[0.2em] hover:bg-ember transition-colors"
            >
              Begin an examination
              <span className="inline-block transition-transform group-hover:translate-x-1">
                ↗
              </span>
            </a>
            <a
              href="#how"
              className="font-mono text-xs uppercase tracking-[0.2em] text-smoke underline decoration-rule underline-offset-8 hover:text-ink hover:decoration-ember"
            >
              How it reads
            </a>
          </motion.div>

          <div className="mt-14 grid grid-cols-3 gap-6 md:gap-10 max-w-xl">
            {[
              { k: "Signals examined", v: "14+" },
              { k: "Modalities", v: "Image · Video · Text" },
              { k: "Median latency", v: "1.4s" },
            ].map((s) => (
              <div key={s.k}>
                <div className="font-display text-2xl md:text-3xl tracking-tight text-ink">
                  {s.v}
                </div>
                <div className="mt-1 font-mono text-[10px] uppercase tracking-[0.2em] text-mute">
                  {s.k}
                </div>
              </div>
            ))}
          </div>
        </div>

        <div className="md:col-span-5 relative">
          <ScanningSpecimen />
        </div>
      </div>

      <Marquee />
    </section>
  );
}

function ScanningSpecimen() {
  return (
    <div className="relative aspect-[4/5] w-full overflow-hidden bg-paper shadow-paper border border-rule">
      {/* Caption plate */}
      <div className="absolute inset-x-0 top-0 flex items-center justify-between px-4 py-3 z-20 border-b border-rule bg-paper/90 backdrop-blur">
        <span className="font-mono text-[10px] uppercase tracking-[0.25em] text-smoke">
          Exhibit A — Live demonstration
        </span>
        <span className="inline-flex items-center gap-2 font-mono text-[10px] text-forest">
          <span className="size-1.5 rounded-full bg-forest pulse-soft" />
          scanning
        </span>
      </div>

      {/* Specimen image: generated SVG portrait silhouette with crosshatch */}
      <div className="absolute inset-0 pt-12 pb-24 px-6 flex items-center justify-center">
        <svg viewBox="0 0 400 500" className="w-full h-full">
          <defs>
            <pattern id="hatch" patternUnits="userSpaceOnUse" width="4" height="4" patternTransform="rotate(30)">
              <line x1="0" y1="0" x2="0" y2="4" stroke="#141413" strokeWidth="0.6" />
            </pattern>
            <radialGradient id="glow" cx="50%" cy="40%" r="60%">
              <stop offset="0%" stopColor="#D84727" stopOpacity="0.18" />
              <stop offset="100%" stopColor="#D84727" stopOpacity="0" />
            </radialGradient>
          </defs>
          <rect width="400" height="500" fill="url(#glow)" />
          {/* Shoulders + head silhouette */}
          <path
            d="M 80 500 L 80 410 Q 80 340 150 320 Q 170 310 175 290 Q 155 280 148 250 Q 135 245 132 225 Q 120 220 120 200 Q 115 160 145 130 Q 175 100 210 102 Q 260 106 275 150 Q 285 185 275 215 Q 273 235 263 245 Q 258 275 240 290 Q 245 312 262 320 Q 330 340 330 410 L 330 500 Z"
            fill="url(#hatch)"
            stroke="#141413"
            strokeWidth="1"
          />
          {/* Reticle */}
          <g stroke="#D84727" strokeWidth="1" fill="none">
            <circle cx="205" cy="195" r="64" strokeDasharray="3 4" />
            <line x1="205" y1="115" x2="205" y2="135" />
            <line x1="205" y1="255" x2="205" y2="275" />
            <line x1="125" y1="195" x2="145" y2="195" />
            <line x1="265" y1="195" x2="285" y2="195" />
          </g>
          <text x="275" y="190" fontFamily="JetBrains Mono" fontSize="8" fill="#141413">
            face · 97% conf.
          </text>
          <text x="275" y="202" fontFamily="JetBrains Mono" fontSize="8" fill="#8A8478">
            δ-edge 0.21
          </text>
        </svg>
      </div>

      {/* Moving scan line */}
      <div className="absolute inset-x-0 top-12 bottom-24 overflow-hidden pointer-events-none">
        <div className="scan-line absolute inset-x-0 h-24 bg-gradient-to-b from-transparent via-ember/25 to-transparent" />
        <div className="absolute inset-x-0 top-0 h-px bg-ember/50" />
      </div>

      {/* Readouts */}
      <div className="absolute inset-x-0 bottom-0 px-4 py-3 z-20 border-t border-rule bg-paper/90 backdrop-blur font-mono text-[10px]">
        <div className="flex items-center justify-between text-smoke">
          <span>ELA · residual 6.2</span>
          <span>NOISE · σ 1.84</span>
          <span>FFT · slope −0.034</span>
        </div>
        <div className="mt-2 flex items-center gap-2">
          <div className="flex-1 h-1.5 bg-rule relative overflow-hidden">
            <motion.div
              className="absolute inset-y-0 left-0 bg-forest"
              initial={{ width: 0 }}
              animate={{ width: "22%" }}
              transition={{ delay: 1.0, duration: 1.2, ease: [0.2, 0.8, 0.2, 1] }}
            />
          </div>
          <span className="text-forest uppercase tracking-[0.2em]">authentic</span>
        </div>
      </div>
    </div>
  );
}

function Marquee() {
  const items = [
    "Error-level analysis",
    "Spectral falloff",
    "Noise stationarity",
    "Face boundary",
    "Temporal flicker",
    "Burstiness",
    "Lexical rhythm",
    "Phrase repetition",
    "Punctuation entropy",
  ];
  return (
    <div className="border-y border-rule bg-paper/60 overflow-hidden">
      <div className="marquee-track flex whitespace-nowrap py-3 font-mono text-xs uppercase tracking-[0.28em] text-smoke">
        {[...items, ...items, ...items].map((t, i) => (
          <span key={i} className="mx-7 inline-flex items-center gap-3">
            <span className="text-ember">✦</span>
            {t}
          </span>
        ))}
      </div>
    </div>
  );
}
