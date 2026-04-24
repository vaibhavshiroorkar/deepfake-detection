"use client";

import { AnimatePresence, motion } from "framer-motion";
import { X } from "lucide-react";
import clsx from "clsx";

export type ToastKind = "error" | "warn" | "info";

type Props = {
  message: string | null;
  kind?: ToastKind;
  onDismiss: () => void;
};

export default function Toast({ message, kind = "error", onDismiss }: Props) {
  return (
    <AnimatePresence>
      {message && (
        <motion.div
          role="alert"
          initial={{ opacity: 0, y: -6 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: -6 }}
          transition={{ duration: 0.2 }}
          className={clsx(
            "mb-4 flex items-start gap-3 border-l-2 bg-paper px-4 py-3",
            kind === "error" && "border-alert bg-alert/5",
            kind === "warn" && "border-amber bg-amber/5",
            kind === "info" && "border-ink bg-bone/40",
          )}
        >
          <div className="flex-1 text-[0.9rem] leading-snug text-ink">
            {message}
          </div>
          <button
            onClick={onDismiss}
            className="text-mute hover:text-ink transition-colors"
            aria-label="Dismiss"
          >
            <X className="size-4" />
          </button>
        </motion.div>
      )}
    </AnimatePresence>
  );
}
