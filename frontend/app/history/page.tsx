import Link from "next/link";
import { redirect } from "next/navigation";
import Masthead from "@/components/Masthead";
import Footer from "@/components/Footer";
import { createClient } from "@/lib/supabase/server";
import { verdictTone } from "@/lib/api";

type Row = {
  id: string;
  kind: "image" | "video" | "audio" | "text";
  filename: string | null;
  suspicion: number;
  verdict: string;
  confidence: number;
  created_at: string;
};

export const dynamic = "force-dynamic";

export default async function HistoryPage() {
  const supabase = createClient();
  const {
    data: { user },
  } = await supabase.auth.getUser();
  if (!user) redirect("/login?next=/history");

  const { data, error } = await supabase
    .from("scans")
    .select("id,kind,filename,suspicion,verdict,confidence,created_at")
    .order("created_at", { ascending: false })
    .limit(100);

  const rows = (data ?? []) as Row[];

  return (
    <main>
      <Masthead />
      <section className="mx-auto max-w-4xl px-6 py-12">
        <div className="flex items-baseline justify-between mb-6">
          <h1 className="font-display text-3xl">History</h1>
          <span className="text-xs text-mute">{rows.length} scan{rows.length === 1 ? "" : "s"}</span>
        </div>

        {error && (
          <div className="mb-4 border border-alert/40 bg-alert/5 px-4 py-3 text-sm text-alert">
            {error.message}
          </div>
        )}

        {rows.length === 0 ? (
          <div className="border border-rule bg-paper px-6 py-12 text-center">
            <p className="text-smoke">No scans yet.</p>
            <Link href="/" className="mt-3 inline-block text-sm text-ember hover:underline">
              Run your first one →
            </Link>
          </div>
        ) : (
          <div className="border border-rule bg-paper divide-y divide-rule">
            {rows.map((r) => {
              const tone = verdictTone(r.suspicion);
              const when = new Date(r.created_at);
              return (
                <div
                  key={r.id}
                  className="grid grid-cols-[auto_1fr_auto] items-center gap-4 px-5 py-3"
                >
                  <span
                    className={`size-2 rounded-full ${
                      tone.color === "forest" ? "bg-forest"
                        : tone.color === "amber" ? "bg-amber"
                        : tone.color === "ember" ? "bg-ember" : "bg-alert"
                    }`}
                  />
                  <div className="min-w-0">
                    <div className="text-sm text-ink truncate">
                      {r.filename || (r.kind === "text" ? "Text submission" : "Untitled")}
                    </div>
                    <div className="text-xs text-mute capitalize">
                      {r.kind} · {r.verdict}
                    </div>
                  </div>
                  <div className="text-right">
                    <div className="text-sm tabular-nums text-ink">
                      {Math.round(r.suspicion * 100)}
                    </div>
                    <div className="text-[10px] text-mute" title={when.toISOString()}>
                      {when.toLocaleString()}
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </section>
      <Footer />
    </main>
  );
}
