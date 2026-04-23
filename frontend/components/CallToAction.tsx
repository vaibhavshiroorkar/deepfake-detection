"use client";

import Link from "next/link";
import { motion } from "framer-motion";
import { ArrowRight } from "lucide-react";

export default function CallToAction() {
  return (
    <section className="border-t border-rule">
      <div className="mx-auto max-w-3xl px-6 py-28 text-center">
        <motion.h2
          initial={{ opacity: 0, y: 8 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, margin: "-80px" }}
          transition={{ duration: 0.6 }}
          className="font-display tracking-tight"
          style={{ fontSize: "clamp(2rem, 4vw, 3rem)", lineHeight: 1.1 }}
        >
          Ready when you are.
        </motion.h2>
        <motion.p
          initial={{ opacity: 0, y: 6 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, margin: "-80px" }}
          transition={{ duration: 0.5, delay: 0.05 }}
          className="mt-5 text-sm text-smoke leading-[1.7]"
        >
          The workspace is one click away. No account required to try it.
        </motion.p>
        <motion.div
          initial={{ opacity: 0 }}
          whileInView={{ opacity: 1 }}
          viewport={{ once: true, margin: "-80px" }}
          transition={{ duration: 0.5, delay: 0.15 }}
          className="mt-10"
        >
          <Link
            href="/detect"
            className="group inline-flex items-center gap-2 bg-ink text-white px-7 py-3.5 text-sm tracking-wide hover:bg-ember transition-colors"
          >
            Open the workspace
            <ArrowRight className="size-4 transition-transform group-hover:translate-x-0.5" />
          </Link>
        </motion.div>
      </div>
    </section>
  );
}
