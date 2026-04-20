"use client";

export default function Masthead() {
  return (
    <header className="border-b border-rule">
      <div className="mx-auto max-w-5xl px-6 h-16 flex items-center justify-between">
        <a href="#" className="font-display text-2xl tracking-tight">
          Veritas<span className="text-ember">.</span>
        </a>
        <nav className="hidden sm:flex items-center gap-8 text-sm text-smoke">
          <a href="#console" className="hover:text-ink transition-colors">Detect</a>
          <a href="#how" className="hover:text-ink transition-colors">How it works</a>
        </nav>
      </div>
    </header>
  );
}
