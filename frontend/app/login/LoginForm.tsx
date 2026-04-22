"use client";

import { useState } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import Link from "next/link";
import clsx from "clsx";
import { createClient } from "@/lib/supabase/client";

type Mode = "signin" | "signup";

export default function LoginForm() {
  const router = useRouter();
  const params = useSearchParams();
  const next = params.get("next") || "/";

  const [mode, setMode] = useState<Mode>("signin");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [busy, setBusy] = useState(false);
  const [msg, setMsg] = useState<{ kind: "ok" | "err"; text: string } | null>(null);

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    setBusy(true);
    setMsg(null);
    const supabase = createClient();
    try {
      if (mode === "signin") {
        const { error } = await supabase.auth.signInWithPassword({ email, password });
        if (error) throw error;
        router.refresh();
        router.push(next);
      } else {
        const { data, error } = await supabase.auth.signUp({
          email,
          password,
          options: {
            emailRedirectTo:
              typeof window !== "undefined"
                ? `${window.location.origin}/auth/callback?next=${encodeURIComponent(next)}`
                : undefined,
          },
        });
        if (error) throw error;
        if (data.session) {
          router.refresh();
          router.push(next);
        } else {
          setMsg({ kind: "ok", text: "Check your email to confirm your account." });
        }
      }
    } catch (e) {
      setMsg({ kind: "err", text: e instanceof Error ? e.message : "Could not sign in." });
    } finally {
      setBusy(false);
    }
  }

  return (
    <main className="mx-auto max-w-md px-6 py-20">
      <div className="border border-rule bg-paper">
        <div className="px-6 py-5 border-b border-rule">
          <h1 className="font-display text-2xl">
            {mode === "signin" ? "Sign in" : "Create account"}
          </h1>
          <p className="mt-1 text-xs text-mute">
            {mode === "signin"
              ? "to view your scan history and manage API keys"
              : "free, no credit card"}
          </p>
        </div>

        <form onSubmit={submit} className="p-6 space-y-4">
          <label className="block">
            <span className="text-xs text-mute">Email</span>
            <input
              type="email"
              required
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              className="mt-1 w-full bg-bone/40 border border-rule px-3 py-2 text-sm focus:outline-none focus:border-ember"
            />
          </label>
          <label className="block">
            <span className="text-xs text-mute">Password</span>
            <input
              type="password"
              required
              minLength={6}
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              className="mt-1 w-full bg-bone/40 border border-rule px-3 py-2 text-sm focus:outline-none focus:border-ember"
            />
          </label>

          {msg && (
            <div
              className={clsx(
                "text-xs px-3 py-2 border",
                msg.kind === "ok"
                  ? "border-forest/40 bg-forest/5 text-forest"
                  : "border-alert/40 bg-alert/5 text-alert",
              )}
            >
              {msg.text}
            </div>
          )}

          <button
            type="submit"
            disabled={busy}
            className={clsx(
              "w-full px-4 py-2.5 text-sm transition-colors",
              busy ? "bg-mute text-white cursor-wait" : "bg-ink text-white hover:bg-ember",
            )}
          >
            {busy ? "..." : mode === "signin" ? "Sign in" : "Create account"}
          </button>

          <button
            type="button"
            onClick={() => {
              setMsg(null);
              setMode((m) => (m === "signin" ? "signup" : "signin"));
            }}
            className="w-full text-xs text-smoke hover:text-ember"
          >
            {mode === "signin"
              ? "No account? Create one →"
              : "Have an account? Sign in →"}
          </button>
        </form>
      </div>

      <p className="mt-4 text-center text-xs text-mute">
        <Link href="/" className="hover:text-ink">← back to detector</Link>
      </p>
    </main>
  );
}
