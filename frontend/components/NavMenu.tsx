"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { useState } from "react";
import { ChevronDown, LogOut } from "lucide-react";
import { createClient } from "@/lib/supabase/client";

const LINKS = [
  { href: "/#console", label: "Detect" },
  { href: "/compare", label: "Compare" },
  { href: "/calibration", label: "Calibration" },
  { href: "/#how", label: "How it works" },
];

export default function NavMenu({ email }: { email: string | null }) {
  const router = useRouter();
  const [open, setOpen] = useState(false);

  async function logout() {
    const supabase = createClient();
    await supabase.auth.signOut();
    setOpen(false);
    router.refresh();
    router.push("/");
  }

  return (
    <nav className="hidden sm:flex items-center gap-7 text-sm text-smoke">
      {LINKS.map((l) => (
        <Link key={l.href} href={l.href} className="hover:text-ink transition-colors">
          {l.label}
        </Link>
      ))}
      {email ? (
        <div className="relative">
          <button
            onClick={() => setOpen((v) => !v)}
            className="flex items-center gap-1.5 text-ink hover:text-ember transition-colors"
          >
            <span className="max-w-[140px] truncate">{email}</span>
            <ChevronDown className="size-3.5" />
          </button>
          {open && (
            <div
              onMouseLeave={() => setOpen(false)}
              className="absolute right-0 mt-2 w-44 border border-rule bg-paper shadow-sm z-20"
            >
              <Link
                href="/history"
                className="block px-4 py-2 text-sm hover:bg-bone"
                onClick={() => setOpen(false)}
              >
                History
              </Link>
              <Link
                href="/keys"
                className="block px-4 py-2 text-sm hover:bg-bone"
                onClick={() => setOpen(false)}
              >
                API keys
              </Link>
              <button
                onClick={logout}
                className="w-full text-left px-4 py-2 text-sm hover:bg-bone flex items-center gap-2 text-alert border-t border-rule"
              >
                <LogOut className="size-3.5" /> Sign out
              </button>
            </div>
          )}
        </div>
      ) : (
        <Link
          href="/login"
          className="px-3 py-1.5 bg-ink text-white text-sm hover:bg-ember transition-colors"
        >
          Sign in
        </Link>
      )}
    </nav>
  );
}
