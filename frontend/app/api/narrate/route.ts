import { NextRequest, NextResponse } from "next/server";

export const runtime = "edge";
export const maxDuration = 18;

type Signal = { name: string; score: number; detail: string };
type Body = {
  kind: "image" | "video" | "audio" | "text";
  suspicion: number;
  verdict: string;
  confidence: number;
  signals: Signal[];
  // Single image (image scans) or array of keyframes (video scans).
  // Either may be present, never both.
  image_data_url?: string | null;
  image_data_urls?: string[] | null;
};

const GROQ_KEY = process.env.GROQ_API_KEY;
const GROQ_MODEL = process.env.GROQ_MODEL || "llama-3.3-70b-versatile";
const GROQ_VISION_MODEL =
  process.env.GROQ_VISION_MODEL || "meta-llama/llama-4-scout-17b-16e-instruct";

const SYSTEM_TEXT = `You are explaining a deepfake check result to a regular person. Not a researcher. Not an analyst. Someone who just wants to know if what they're looking at is real.

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

const SYSTEM_VISION = `You are looking at an image (or a few frames from a video) and explaining a deepfake check result to a regular person. You can see the picture(s). Your job is to point at what's actually there.

Voice:
- Plain language. No jargon. Talk like a friend.
- Short sentences. Words anyone uses.
- Banned words: perplexity, spectral, burstiness, calibrated, residual, optics, manifold, latent, embedding, classifier, ensemble, heuristic, modality, posterior, signature, synthetic.
- "AI-generated" or "fake" or "real" are fine.

What to do:
- First, describe what you actually see. If it's a video and the frames show a car, say "the clip shows a car" not "I see a person". Don't make up content that isn't in the frames.
- Mention one or two specific things that look real or fake (skin, hair, eyes, hands, lighting, background, edges, shadows, text, motion blur, reflections, the way objects sit on a surface).
- For videos: if multiple frames are shown, they're keyframes from the same clip. Treat them as one piece of evidence. If something is consistent across frames, that's a clue. If something jumps around, that's a different clue.
- For videos: when the data includes "Audio:" signals or an "Audio track overall" signal, the audio was also analysed. Mention what the audio says (e.g., "the audio sounds clean" or "the voice has the flat quality of a clone") if it's relevant.
- Tie what you see to the verdict. If the picture and the signals disagree, say so plainly.

Rules:
- 2 to 4 sentences. Never more.
- Never use long dashes. Use commas, colons, or periods.
- Don't invent details that aren't in the frames or the data. If you don't see a person, don't talk about a person.
- If you can't tell, say so.
- Vary how you start. Don't begin with "This image" every time.
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

  // Collect the images we have. image_data_urls (array) takes precedence
  // for video keyframes; image_data_url (single) is used for images.
  const imageList: string[] = [];
  if (Array.isArray(body.image_data_urls)) {
    for (const u of body.image_data_urls) {
      if (typeof u === "string" && u.startsWith("data:image/")) imageList.push(u);
    }
  }
  if (
    imageList.length === 0 &&
    typeof body.image_data_url === "string" &&
    body.image_data_url.startsWith("data:image/")
  ) {
    imageList.push(body.image_data_url);
  }
  const useVision = imageList.length > 0 && (body.kind === "image" || body.kind === "video");

  const ctrl = new AbortController();
  const timeout = setTimeout(() => ctrl.abort(), 15_000);

  try {
    const messages = useVision
      ? [
          { role: "system", content: SYSTEM_VISION },
          {
            role: "user",
            content: [
              { type: "text", text: buildUserPrompt(body) },
              ...imageList.map((u) => ({
                type: "image_url" as const,
                image_url: { url: u },
              })),
            ],
          },
        ]
      : [
          { role: "system", content: SYSTEM_TEXT },
          { role: "user", content: buildUserPrompt(body) },
        ];

    const res = await fetch("https://api.groq.com/openai/v1/chat/completions", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${GROQ_KEY}`,
      },
      signal: ctrl.signal,
      body: JSON.stringify({
        model: useVision ? GROQ_VISION_MODEL : GROQ_MODEL,
        temperature: 0.4,
        max_tokens: 260,
        messages,
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

    const cleaned = narrative.replace(/—/g, ", ").replace(/–/g, "-");

    return NextResponse.json(
      { narrative: cleaned, vision: useVision },
      { status: 200 },
    );
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
