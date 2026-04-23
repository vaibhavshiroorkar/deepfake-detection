"use client";

import { motion } from "framer-motion";

const PRINCIPLES = [
  {
    head: "Show the working",
    body:
      "Every verdict comes with the underlying signals, their individual scores, and a one-line explanation of what each one measured. Nothing is taken on faith.",
  },
  {
    head: "Prefer learned over guessed",
    body:
      "Where a properly trained classifier exists, we use it. Where it doesn't, we fall back to forensic heuristics — and label them as such, with low weight.",
  },
  {
    head: "Hedge when uncertain",
    body:
      "The verdict has four levels: likely authentic, inconclusive, likely synthetic, highly likely synthetic. The middle two are the honest answers most of the time.",
  },
];

export default function Principles() {
  return (
    <section className="border-t border-rule bg-bone">
      <div className="mx-auto max-w-5xl px-6 py-24">
        <motion.p
          initial={{ opacity: 0, y: 6 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, margin: "-60px" }}
          transition={{ duration: 0.5 }}
          className="text-xs uppercase tracking-[0.22em] text-mute"
        >
          What we believe
        </motion.p>

        <div className="mt-12 grid md:grid-cols-3 gap-12 md:gap-10">
          {PRINCIPLES.map((p, i) => (
            <motion.div
              key={p.head}
              initial={{ opacity: 0, y: 12 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true, margin: "-40px" }}
              transition={{ duration: 0.5, delay: i * 0.08 }}
            >
              <h3 className="font-display text-xl tracking-tight">{p.head}</h3>
              <div className="hairline w-10 mt-3 mb-4 bg-ember" style={{ background: "var(--ember)" }} />
              <p className="text-sm text-smoke leading-[1.75]">{p.body}</p>
            </motion.div>
          ))}
        </div>
      </div>
    </section>
  );
}
