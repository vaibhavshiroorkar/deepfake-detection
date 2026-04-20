import { NextRequest, NextResponse } from "next/server";

const BACKEND = process.env.BACKEND_URL || "http://localhost:8000";

export const runtime = "nodejs";
export const maxDuration = 60;

export async function POST(req: NextRequest) {
  const kind = req.nextUrl.searchParams.get("kind");
  if (!kind || !["image", "video", "audio", "text"].includes(kind)) {
    return NextResponse.json({ error: "kind must be image, video, audio, or text" }, { status: 400 });
  }

  try {
    if (kind === "text") {
      const body = await req.json();
      const res = await fetch(`${BACKEND}/api/detect/text`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      const data = await res.json();
      return NextResponse.json(data, { status: res.status });
    }

    // image, video, or audio — forward multipart
    const form = await req.formData();
    const upstream = new FormData();
    const file = form.get("file");
    if (!(file instanceof File)) {
      return NextResponse.json({ error: "file missing" }, { status: 400 });
    }
    upstream.append("file", file, file.name);
    const res = await fetch(`${BACKEND}/api/detect/${kind}`, {
      method: "POST",
      body: upstream,
    });
    const data = await res.json();
    return NextResponse.json(data, { status: res.status });
  } catch (e: unknown) {
    const message = e instanceof Error ? e.message : "Upstream error";
    return NextResponse.json(
      { error: `Detection service unreachable: ${message}` },
      { status: 502 },
    );
  }
}
