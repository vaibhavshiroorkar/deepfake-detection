"use client";

import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import { useEffect, useState } from "react";
import { ChevronDown, LogOut, Menu, X } from "lucide-react";
import clsx from "clsx";
import { createClient } from "@/lib/supabase/client";

const LINKS = [
  { href: "/detect", label: "Detect" },
  { href: "/compare", label: "Compare" },
  { href: "/method", label: "Method" },
  { href: "/calibration", label: "Calibration" },
];

export default function NavMenu() {
  const pathname = usePathname();
  const router = useRouter();
  const [email, setEmail] = useState<string | null>(null);
  const [userOpen, setUserOpen] = useState(false);
  const [mobileOpen, setMobileOpen] = useState(false);

  useEffect(() => {
    const supabase = createClient();
    supabase.auth.getSession().then(({ data }) => {
      setEmail(data.session?.user?.email ?? null);
    });
    const { data: sub } = supabase.auth.onAuthStateChange((_, session) => {
      setEmail(session?.user?.email ?? null);
    });
    return () => sub.subscription.unsubscribe();
  }, []);

  async function logout() {
    const supabase = createClient();
    await supabase.auth.signOut();
    setUserOpen(false);
    router.refresh();
    router.push("/");
  }

  return (
    <>
      <nav className="hidden md:flex items-center gap-8 text-[0.95rem]">
        {LINKS.map((l) => {
          const active = pathname === l.href;
          return (
            <Link
              key={l.href}
              href={l.href}
              prefetch
              className={clsx(
                "relative transition-colors",
                active ? "text-ink" : "text-smoke hover:text-ink",
              )}
            >
              {l.label}
              {active && (
                <span className="absolute -bottom-1.5 left-0 right-0 h-px bg-ember" />
              )}
            </Link>
          );
        })}
        {email ? (
          <div className="relative">
            <button
              onClick={() => setUserOpen((v) => !v)}
              className="flex items-center gap-1.5 text-ink hover:text-ember transition-colors"
            >
              <span className="max-w-[160px] truncate font-mono text-sm">
                {email}
              </span>
              <ChevronDown className="size-3.5" />
            </button>
            {userOpen && (
              <div
                onMouseLeave={() => setUserOpen(false)}
                className="absolute right-0 mt-3 w-48 border border-ink bg-paper z-20 shadow-[6px_6px_0_rgba(20,20,19,0.08)]"
              >
                <Link
                  href="/history"
                  className="block px-5 py-3 text-sm hover:bg-bone border-b border-rule"
                  onClick={() => setUserOpen(false)}
                >
                  History
                </Link>
                <Link
                  href="/keys"
                  className="block px-5 py-3 text-sm hover:bg-bone border-b border-rule"
                  onClick={() => setUserOpen(false)}
                >
                  API keys
                </Link>
                <button
                  onClick={logout}
                  className="w-full text-left px-5 py-3 text-sm hover:bg-bone flex items-center gap-2 text-alert"
                >
                  <LogOut className="size-3.5" /> Sign out
                </button>
              </div>
            )}
          </div>
        ) : (
          <Link href="/login" className="btn">
            Sign in
          </Link>
        )}
      </nav>

      {/* Mobile */}
      <button
        onClick={() => setMobileOpen((v) => !v)}
        className="md:hidden p-2 -mr-2 text-ink"
        aria-label="Menu"
      >
        {mobileOpen ? <X className="size-5" /> : <Menu className="size-5" />}
      </button>

      {mobileOpen && (
        <div className="md:hidden absolute top-full inset-x-0 border-t border-b border-rule bg-paper">
          <div className="page-frame py-6 flex flex-col gap-4">
            {LINKS.map((l) => (
              <Link
                key={l.href}
                href={l.href}
                prefetch
                onClick={() => setMobileOpen(false)}
                className="font-display text-xl"
              >
                {l.label}
              </Link>
            ))}
            <div className="hairline my-2" />
            {email ? (
              <>
                <Link href="/history" onClick={() => setMobileOpen(false)} className="text-sm">History</Link>
                <Link href="/keys" onClick={() => setMobileOpen(false)} className="text-sm">API keys</Link>
                <button onClick={logout} className="text-left text-sm text-alert">Sign out</button>
              </>
            ) : (
              <Link href="/login" onClick={() => setMobileOpen(false)} className="btn w-fit">
                Sign in
              </Link>
            )}
          </div>
        </div>
      )}
    </>
  );
}
