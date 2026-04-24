"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { Copy, Trash2, Plus } from "lucide-react";
import clsx from "clsx";
import { createClient } from "@/lib/supabase/client";

type KeyRow = {
  id: string;
  name: string;
  prefix: string;
  created_at: string;
  last_used: string | null;
  revoked_at: string | null;
};

async function sha256Hex(text: string) {
  const buf = new TextEncoder().encode(text);
  const digest = await crypto.subtle.digest("SHA-256", buf);
  return Array.from(new Uint8Array(digest))
    .map((b) => b.toString(16).padStart(2, "0"))
    .join("");
}

function generateKey(): { raw: string; prefix: string } {
  const bytes = new Uint8Array(24);
  crypto.getRandomValues(bytes);
  const secret = btoa(String.fromCharCode(...bytes))
    .replace(/\+/g, "-")
    .replace(/\//g, "_")
    .replace(/=+$/, "");
  const prefix = secret.slice(0, 6);
  return { raw: `vrt_${prefix}_${secret.slice(6)}`, prefix };
}

export default function KeysClient() {
  const router = useRouter();
  const [rows, setRows] = useState<KeyRow[]>([]);
  const [loading, setLoading] = useState(true);
  const [authed, setAuthed] = useState<boolean | null>(null);
  const [newName, setNewName] = useState("");
  const [creating, setCreating] = useState(false);
  const [justCreated, setJustCreated] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const supabase = createClient();

  async function load() {
    setLoading(true);
    const {
      data: { user },
    } = await supabase.auth.getUser();
    if (!user) {
      setAuthed(false);
      router.push("/login?next=/keys");
      return;
    }
    setAuthed(true);
    const { data, error } = await supabase
      .from("api_keys")
      .select("id,name,prefix,created_at,last_used,revoked_at")
      .order("created_at", { ascending: false });
    if (error) setError(error.message);
    else setRows((data ?? []) as KeyRow[]);
    setLoading(false);
  }

  useEffect(() => {
    load();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  async function create() {
    setCreating(true);
    setError(null);
    try {
      const { raw, prefix } = generateKey();
      const key_hash = await sha256Hex(raw);
      const {
        data: { user },
      } = await supabase.auth.getUser();
      if (!user) throw new Error("Not signed in");
      const { error } = await supabase.from("api_keys").insert({
        user_id: user.id,
        name: newName || "Untitled key",
        prefix,
        key_hash,
      });
      if (error) throw error;
      setJustCreated(raw);
      setNewName("");
      await load();
    } catch (e) {
      setError(e instanceof Error ? e.message : "Could not create key.");
    } finally {
      setCreating(false);
    }
  }

  async function revoke(id: string) {
    if (!confirm("Revoke this key? Any scripts using it will stop working.")) return;
    const { error } = await supabase
      .from("api_keys")
      .update({ revoked_at: new Date().toISOString() })
      .eq("id", id);
    if (error) setError(error.message);
    else load();
  }

  async function destroy(id: string) {
    if (!confirm("Delete this key permanently?")) return;
    const { error } = await supabase.from("api_keys").delete().eq("id", id);
    if (error) setError(error.message);
    else load();
  }

  if (authed === false) return null;

  return (
    <main className="mx-auto max-w-3xl px-6 py-12">
      <h1 className="font-display text-3xl">API keys</h1>
      <p className="mt-2 text-sm text-smoke max-w-prose">
        Use a key to POST scans from a script. Send it as the{" "}
        <code className="text-ink">X-API-Key</code> header against the same
        endpoints the website uses.
      </p>

      <div className="mt-6 border border-rule bg-paper">
        <div className="px-5 py-4 border-b border-rule flex items-center gap-3">
          <input
            value={newName}
            onChange={(e) => setNewName(e.target.value)}
            placeholder="Key name (e.g. 'pipeline-prod')"
            className="flex-1 bg-bone/40 border border-rule px-3 py-2 text-sm focus:outline-none focus:border-ember"
          />
          <button
            onClick={create}
            disabled={creating}
            className={clsx(
              "inline-flex items-center gap-2 px-4 py-2 text-sm",
              creating ? "bg-mute text-white" : "bg-ink text-white hover:bg-ember",
            )}
          >
            <Plus className="size-3.5" />
            Create
          </button>
        </div>

        {justCreated && (
          <div className="px-5 py-4 border-b border-rule bg-amber/5">
            <div className="text-xs text-mute mb-2">
              Copy this now. It won&apos;t be shown again.
            </div>
            <div className="flex items-center gap-2">
              <code className="flex-1 break-all bg-paper border border-rule px-3 py-2 text-xs">
                {justCreated}
              </code>
              <button
                onClick={() => navigator.clipboard.writeText(justCreated)}
                className="px-3 py-2 text-xs border border-rule hover:bg-bone"
              >
                <Copy className="size-3.5" />
              </button>
              <button
                onClick={() => setJustCreated(null)}
                className="px-3 py-2 text-xs text-mute hover:text-ink"
              >
                Done
              </button>
            </div>
          </div>
        )}

        {error && (
          <div className="px-5 py-3 border-b border-rule text-xs text-alert bg-alert/5">
            {error}
          </div>
        )}

        {loading ? (
          <div className="px-5 py-8 text-center text-sm text-mute">Loading…</div>
        ) : rows.length === 0 ? (
          <div className="px-5 py-8 text-center text-sm text-mute">No keys yet.</div>
        ) : (
          <div className="divide-y divide-rule">
            {rows.map((k) => {
              const revoked = !!k.revoked_at;
              return (
                <div key={k.id} className="px-5 py-3 flex items-center gap-4">
                  <div className="flex-1 min-w-0">
                    <div className="text-sm text-ink truncate">{k.name}</div>
                    <div className="text-xs text-mute">
                      <code className="text-smoke">vrt_{k.prefix}_…</code> ·
                      created {new Date(k.created_at).toLocaleDateString()} ·
                      {k.last_used
                        ? ` last used ${new Date(k.last_used).toLocaleString()}`
                        : " never used"}
                      {revoked && <span className="ml-1 text-alert">· revoked</span>}
                    </div>
                  </div>
                  <div className="flex items-center gap-1">
                    {!revoked && (
                      <button
                        onClick={() => revoke(k.id)}
                        className="px-2 py-1 text-xs text-amber hover:bg-amber/10"
                      >
                        Revoke
                      </button>
                    )}
                    <button
                      onClick={() => destroy(k.id)}
                      className="px-2 py-1 text-xs text-alert hover:bg-alert/10"
                      title="Delete"
                    >
                      <Trash2 className="size-3.5" />
                    </button>
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </div>

      <div className="mt-8 border border-rule bg-paper p-5">
        <div className="text-xs text-mute mb-2">Example</div>
        <pre className="text-xs text-ink overflow-x-auto leading-relaxed">{`curl -X POST https://your-backend.onrender.com/api/detect/image \\
  -H "X-API-Key: vrt_xxxxxx_..." \\
  -F "file=@suspect.jpg"`}</pre>
      </div>
    </main>
  );
}
