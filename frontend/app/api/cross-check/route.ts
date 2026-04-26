import { NextRequest, NextResponse } from "next/server";

export const runtime = "edge";
export const maxDuration = 18;

type Body = {
  kind: "image" | "text";
  text?: string;
  image_data_url?: string | null;
};

const GROQ_KEY = process.env.GROQ_API_KEY;
const GROQ_MODEL = process.env.GROQ_MODEL || "llama-3.3-70b-versatile";
const GROQ_VISION_MODEL =
  process.env.GROQ_VISION_MODEL || "meta-llama/llama-4-scout-17b-16e-instruct";

const SYSTEM_TEXT = `You are an extra cross-check on whether a piece of writing was produced by a large language model.

You are given the writing. You return a single JSON object with two fields:
- "score": a number between 0 and 1, where 0 means clearly written by a human and 1 means clearly written by an LLM.
- "reason": one short sentence (under 25 words) explaining the strongest cue.

Do not include backticks, code fences, or anything outside the JSON object.

Look at:
- Sentence rhythm and length variation. LLMs tend to be uniform; humans bursty.
- Word choice. LLMs reach for "moreover", "furthermore", "delve", "tapestry", "navigating", "complexities", "leverage", "crucial role".
- Punctuation. LLMs over-use semicolons, parentheses, and clean comma lists.
- Specificity. Human writing usually has a specific moment, name, or detail. LLM writing stays general.
- Hedging. LLMs hedge symmetrically ("on one hand... on the other"). Humans don't.

If the text is too short to judge confidently, return 0.5 with a reason saying so.`;

const SYSTEM_VISION = `You are an extra cross-check on whether an image was AI-generated.

You are given the image. You return a single JSON object with two fields:
- "score": a number between 0 and 1, where 0 means clearly a real photograph and 1 means clearly AI-generated.
- "reason": one short sentence (under 25 words) naming the strongest visual cue.

Do not include backticks, code fences, or anything outside the JSON object.

Look at:
- Skin: too smooth, perfect pores, uncanny lighting on faces.
- Hair: melted strands, hair fading into background.
- Hands and fingers: wrong count, bent the wrong way, fused.
- Eyes: mismatched reflections, asymmetric pupils.
- Background: text is gibberish, repeating patterns, objects bend impossibly.
- Lighting: shadows from different sources, glow without a source.
- Edges: hair-edge halos, blur around subject that doesn't match focus.
- Symmetry: too perfect or wrong-asymmetric.

If you genuinely can't tell, return 0.5 with a reason saying so. Do not pretend.`;

function parseJsonResponse(raw: string): { score: number; reason: string } | null {
  // Strip code fences if the model added them anyway.
  const cleaned = raw
    .trim()
    .replace(/^```(?:json)?\s*/i, "")
    .replace(/```\s*$/i, "")
    .trim();
  try {
    const obj = JSON.parse(cleaned);
    if (typeof obj.score !== "number" || typeof obj.reason !== "string") {
      return null;
    }
    const score = Math.max(0, Math.min(1, obj.score));
    return { score, reason: obj.reason.replace(/—/g, ", ").slice(0, 200) };
  } catch {
    // Sometimes the model wraps the JSON in prose. Try to find a JSON
    // blob inside.
    const match = cleaned.match(/\{[\s\S]*\}/);
    if (!match) return null;
    try {
      const obj = JSON.parse(match[0]);
      if (typeof obj.score !== "number" || typeof obj.reason !== "string") {
        return null;
      }
      const score = Math.max(0, Math.min(1, obj.score));
      return { score, reason: obj.reason.replace(/—/g, ", ").slice(0, 200) };
    } catch {
      return null;
    }
  }
}

export async function POST(req: NextRequest) {
  if (!GROQ_KEY) {
    return NextResponse.json({ error: "no_api_key" }, { status: 200 });
  }

  let body: Body;
  try {
    body = (await req.json()) as Body;
  } catch {
    return NextResponse.json({ error: "bad_json" }, { status: 400 });
  }

  const isText = body.kind === "text" && typeof body.text === "string";
  const isImage =
    body.kind === "image" &&
    typeof body.image_data_url === "string" &&
    body.image_data_url.startsWith("data:image/");

  if (!isText && !isImage) {
    return NextResponse.json({ error: "bad_body" }, { status: 400 });
  }

  const ctrl = new AbortController();
  const timeout = setTimeout(() => ctrl.abort(), 14_000);

  try {
    const messages = isText
      ? [
          { role: "system", content: SYSTEM_TEXT },
          { role: "user", content: body.text!.slice(0, 8000) },
        ]
      : [
          { role: "system", content: SYSTEM_VISION },
          {
            role: "user",
            content: [
              { type: "text", text: "Rate this image. Return JSON only." },
              {
                type: "image_url",
                image_url: { url: body.image_data_url! },
              },
            ],
          },
        ];

    const res = await fetch("https://api.groq.com/openai/v1/chat/completions", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${GROQ_KEY}`,
      },
      signal: ctrl.signal,
      body: JSON.stringify({
        model: isText ? GROQ_MODEL : GROQ_VISION_MODEL,
        temperature: 0.2,
        max_tokens: 200,
        messages,
        response_format: { type: "json_object" },
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
    const raw: string | undefined = data?.choices?.[0]?.message?.content;
    if (!raw) {
      return NextResponse.json({ error: "empty" }, { status: 200 });
    }

    const parsed = parseJsonResponse(raw);
    if (!parsed) {
      return NextResponse.json(
        { error: "bad_format", raw: raw.slice(0, 200) },
        { status: 200 },
      );
    }

    return NextResponse.json(
      { score: parsed.score, reason: parsed.reason },
      { status: 200 },
    );
  } catch (e: unknown) {
    const message = e instanceof Error ? e.message : "unknown";
    return NextResponse.json({ error: "network", detail: message }, { status: 200 });
  } finally {
    clearTimeout(timeout);
  }
}
