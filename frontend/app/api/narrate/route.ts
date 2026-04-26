import { NextRequest, NextResponse } from "next/server";

export const runtime = "edge";
export const maxDuration = 15;

type Signal = { name: string; score: number; detail: string };
type Body = {
  kind: "image" | "video" | "audio" | "text";
  suspicion: number;
  verdict: string;
  confidence: number;
  signals: Signal[];
};

const GROQ_KEY = process.env.GROQ_API_KEY;
const GROQ_MODEL = process.env.GROQ_MODEL || "llama-3.3-70b-versatile";

const SYSTEM = `You are explaining a deepfake check result to a regular person. Not a researcher. Not an analyst. Someone who just wants to know if what they're looking at is real.

Voice:
- Plain language only. Talk like a normal person.
- Short sentences. Words people actually use.
- If a signal name is technical, translate it. "Chromatic aberration" becomes "the way colors blend at the edges". "Spectral features" becomes "the shape of the sound". "Burstiness" becomes "how much sentence length varies". Never copy the technical name into your summary.
- Banned words: perplexity, spectral, burstiness, calibrated, residual, optics, manifold, latent, embedding, classifier, ensemble, heuristic, modality, posterior, signature.
- "AI-generated" or "fake" or "real" are fine. Avoid "synthetic" unless it reads naturally.
- Sound like you're explaining this to a friend, not writing a report.

Rules:
- 2 to 3 sentences. Never more.
- Never use long dashes. Use commas, colons, or periods.
- Never make up details. Only mention what's actually in the data.
- If the AI checks say one thing and the photo's lighting or noise say another, just say so in plain words.
- If the answer is unclear, say so honestly. Don't pretend.
- Vary how you start. Don't begin every reply with "This image" or "This audio".
- No lists. No headings. Just a short paragraph.

Return the summary only. No greeting. No "Here is".`;

function buildUserPrompt(body: Body): string {
  const sigs = (body.signals ?? [])
    .filter((s) => s && typeof s.name === "string")
    .slice(0, 14)
    .map(
      (s) =>
        `- ${s.name}: score ${(s.score * 100).toFixed(0)}/100. ${s.detail}`,
    )
    .join("\n");

  return `Modality: ${body.kind}
Verdict: ${body.verdict}
Suspicion: ${(body.suspicion * 100).toFixed(0)}/100
Confidence: ${(body.confidence * 100).toFixed(0)}%

Signals examined:
${sigs}

Write the summary now.`;
}

export async function POST(req: NextRequest) {
  if (!GROQ_KEY) {
    return NextResponse.json(
      { error: "no_api_key", narrative: null },
      { status: 200 },
    );
  }

  let body: Body;
  try {
    body = (await req.json()) as Body;
  } catch {
    return NextResponse.json({ error: "bad_json" }, { status: 400 });
  }
  if (!body || typeof body.suspicion !== "number" || !Array.isArray(body.signals)) {
    return NextResponse.json({ error: "bad_body" }, { status: 400 });
  }

  const ctrl = new AbortController();
  const timeout = setTimeout(() => ctrl.abort(), 12_000);

  try {
    const res = await fetch("https://api.groq.com/openai/v1/chat/completions", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${GROQ_KEY}`,
      },
      signal: ctrl.signal,
      body: JSON.stringify({
        model: GROQ_MODEL,
        temperature: 0.4,
        max_tokens: 220,
        messages: [
          { role: "system", content: SYSTEM },
          { role: "user", content: buildUserPrompt(body) },
        ],
      }),
    });

    if (!res.ok) {
      const text = await res.text();
      return NextResponse.json(
        { error: "upstream", status: res.status, detail: text.slice(0, 200) },
        { status: 200 },
      );
    }

    const data = await res.json();
    const narrative: string | undefined =
      data?.choices?.[0]?.message?.content?.trim();

    if (!narrative) {
      return NextResponse.json({ error: "empty", narrative: null }, { status: 200 });
    }

    // Strip any em dashes the model still snuck in.
    const cleaned = narrative.replace(/—/g, ", ").replace(/–/g, "-");

    return NextResponse.json({ narrative: cleaned }, { status: 200 });
  } catch (e: unknown) {
    const message = e instanceof Error ? e.message : "unknown";
    return NextResponse.json(
      { error: "network", detail: message, narrative: null },
      { status: 200 },
    );
  } finally {
    clearTimeout(timeout);
  }
}
