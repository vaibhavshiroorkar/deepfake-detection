import Link from "next/link";
import NavMenu from "./NavMenu";

export default function Masthead() {
  return (
    <header className="sticky top-0 z-30 backdrop-blur-md bg-bone/80 border-b border-rule">
      <div className="page-frame h-[4.5rem] flex items-center justify-between">
        <Link
          href="/"
          className="group flex items-baseline gap-3 font-display text-[1.5rem] tracking-tight leading-none"
        >
          <span className="relative">
            Veritas<span className="text-ember">.</span>
          </span>
          <span className="hidden sm:inline running-head translate-y-[-2px]">
            evidence / 2026
          </span>
        </Link>
        <NavMenu />
      </div>
    </header>
  );
}
