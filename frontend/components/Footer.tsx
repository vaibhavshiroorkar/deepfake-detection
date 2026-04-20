"use client";

export default function Footer() {
  return (
    <footer className="border-t border-rule">
      <div className="mx-auto max-w-5xl px-6 py-8 flex flex-wrap items-center justify-between gap-3 text-xs text-mute">
        <span>
          <span className="font-display text-sm text-ink">Veritas</span> — an informed second opinion, not a verdict.
        </span>
        <span>Uploads are processed in memory. Nothing retained.</span>
      </div>
    </footer>
  );
}
