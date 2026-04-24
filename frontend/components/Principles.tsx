"use client";

import { motion } from "framer-motion";

const PRINCIPLES = [
  {
    n: "i.",
    head: "Show the working",
    body:
      "Every verdict comes with the underlying signals, their individual scores, and a one-line explanation of what each one measured. Nothing is taken on faith.",
  },
  {
    n: "ii.",
    head: "Prefer learned over guessed",
    body:
      "Where a properly trained classifier exists, we use it. Where it doesn't, we fall back to forensic heuristics, and label them as such with low weight in the final verdict.",
  },
  {
    n: "iii.",
    head: "Hedge when uncertain",
    body:
      "The verdict has four levels: likely authentic, inconclusive, likely synthetic, highly likely synthetic. The middle two are the honest answers most of the time.",
  },
];

const EASE = [0.2, 0.8, 0.2, 1] as const;

export default function Principles() {
  return (
    <section className="section-screen border-t border-rule">
      <div className="page-frame flex-1 flex flex-col justify-center py-24">
        <div className="flex items-baseline justify-between mb-10">
          <motion.span
            initial={{ opacity: 0 }}
            whileInView={{ opacity: 1 }}
            viewport={{ once: true, margin: "-120px" }}
            transition={{ duration: 0.5 }}
            className="running-head"
          >
            Plate 004 · Principles
          </motion.span>
          <motion.span
            initial={{ opacity: 0 }}
            whileInView={{ opacity: 1 }}
            viewport={{ once: true, margin: "-120px" }}
            transition={{ duration: 0.5, delay: 0.1 }}
            className="running-head hidden sm:inline"
          >
            What we believe
          </motion.span>
        </div>

        <motion.h2
          initial={{ opacity: 0, y: 12 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, margin: "-120px" }}
          transition={{ duration: 0.7, ease: EASE }}
          className="display-xl max-w-[17ch]"
        >
          A detector with
          <br />
          <span className="italic text-ink/70">a point of view.</span>
        </motion.h2>

        <div className="mt-16 md:mt-24 grid md:grid-cols-3 gap-14 md:gap-12">
          {PRINCIPLES.map((p, i) => (
            <motion.article
              key={p.head}
              initial={{ opacity: 0, y: 18 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true, margin: "-80px" }}
              transition={{ duration: 0.7, delay: i * 0.1, ease: EASE }}
              className="relative"
            >
              <div
                className="font-display italic text-ember mb-5"
                style={{ fontSize: "2.4rem", lineHeight: 1 }}
              >
                {p.n}
              </div>
              <h3 className="display-md mb-5 max-w-[18ch]">{p.head}</h3>
              <p className="body-sm text-smoke max-w-[38ch]">{p.body}</p>
            </motion.article>
          ))}
        </div>
      </div>
    </section>
  );
}
