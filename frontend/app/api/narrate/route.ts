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

const SYSTEM = `You are an evidence analyst writing a short, plain-English summary of a deepfake detection result.

Voice:
- Careful, analytic, calm. Not hype, not breathless, not corporate.
- Talk like a person who knows the topic, not a marketing brochure.
- Specific over vague. Name the actual signals and their scores.
- Acknowledge uncertainty honestly. "Inconclusive" is a real answer.

Rules:
- 2 to 4 sentences. Never more.
- Never use em dashes (long dashes). Use commas, colons, or periods.
- Never invent signals or numbers. Only reference what is in the data.
- If learned classifiers disagree with forensic heuristics, name that explicitly.
- If the verdict is "inconclusive" do not pretend you know.
- Do not start with "This image" or "This text" repeatedly. Vary openings.
- Do not say "synthetic" if you can say "AI-generated"; mix the terms naturally.
- No bullet points. No headings. Just prose.

Return only the summary, no preamble.`;

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
