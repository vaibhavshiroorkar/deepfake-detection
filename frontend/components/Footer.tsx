import Link from "next/link";

export default function Footer() {
  return (
    <footer className="border-t border-rule bg-paper">
      <div className="page-frame py-16 md:py-20">
        <div className="flex flex-col md:flex-row md:items-end md:justify-between gap-10">
          <div>
            <Link
              href="/"
              className="font-display text-4xl md:text-5xl tracking-tight leading-none"
            >
              Veritas<span className="text-ember">.</span>
            </Link>
            <p className="body mt-5 max-w-[38ch]">
              An informed second opinion, not a verdict. Uploads are processed
              in memory. Nothing is retained.
            </p>
          </div>
          <nav className="grid grid-cols-2 gap-x-14 gap-y-3 md:text-right">
            <Link href="/detect" className="text-smoke hover:text-ink">Detect</Link>
            <Link href="/compare" className="text-smoke hover:text-ink">Compare</Link>
            <Link href="/calibration" className="text-smoke hover:text-ink">Calibration</Link>
            <Link href="/history" className="text-smoke hover:text-ink">History</Link>
            <Link href="/method" className="text-smoke hover:text-ink">Method</Link>
            <Link href="/keys" className="text-smoke hover:text-ink">API keys</Link>
          </nav>
        </div>
        <div className="mt-16 flex flex-wrap items-center justify-between gap-4 text-xs text-mute font-mono tracking-wide">
          <span>© 2026 · Evidence catalog</span>
          <span>Processed in memory · Nothing retained</span>
        </div>
      </div>
    </footer>
  );
}
