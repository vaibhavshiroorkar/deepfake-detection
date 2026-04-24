import { createClient } from "@/lib/supabase/client";

export type Signal = {
  name: string;
  score: number;
  detail: string;
};

export type BaseResult = {
  kind: "image" | "video" | "audio" | "text";
  suspicion: number;
  verdict: string;
  confidence: number;
  signals: Signal[];
};

export type C2PAManifest = {
  present: boolean;
  claim_generator?: string;
  signed_by?: string;
  actions?: string[];
  trusted?: boolean;
};

export type ImageResult = BaseResult & {
  kind: "image";
  filename: string;
  dimensions: { width: number; height: number };
  heatmaps?: { ela?: string; noise?: string } | null;
  c2pa?: C2PAManifest;
  id?: string;
};

export type VideoResult = BaseResult & {
  kind: "video";
  filename: string;
  duration_seconds: number;
  dimensions: { width: number; height: number };
  timeline: { timestamp: number; suspicion: number; verdict: string }[];
  id?: string;
};

export type AudioResult = BaseResult & {
  kind: "audio";
  filename: string;
  duration_seconds: number;
  sample_rate: number;
  id?: string;
};

export type TextResult = BaseResult & {
  kind: "text";
  length: { characters: number; words: number; sentences: number };
  id?: string;
};

export type DetectionResult = ImageResult | VideoResult | AudioResult | TextResult;

// Public backend URL — when set, the browser calls the detection service
// directly, bypassing Vercel's serverless function (which would otherwise
// time out during HF Spaces cold-starts).
const BACKEND = (process.env.NEXT_PUBLIC_BACKEND_URL || "").replace(/\/+$/, "");

function detectUrl(kind: "image" | "video" | "audio" | "text"): string {
  return BACKEND ? `${BACKEND}/api/detect/${kind}` : `/api/detect?kind=${kind}`;
}

async function authHeaders(): Promise<Record<string, string>> {
  if (!BACKEND) return {};
  try {
    const supabase = createClient();
    const {
      data: { session },
    } = await supabase.auth.getSession();
    if (session?.access_token) {
      return { Authorization: `Bearer ${session.access_token}` };
    }
  } catch {
    // No session — backend allows anonymous scans.
  }
  return {};
}

async function parseOrThrow(res: Response): Promise<unknown> {
  const raw = await res.text();
  let parsed: unknown = null;
  try {
    parsed = raw ? JSON.parse(raw) : null;
  } catch {
    // Non-JSON response — usually an HTML error page (gateway timeout,
    // size-limit rejection at the proxy, CORS block, etc). Surface
    // something readable based on status.
    if (res.status === 413) {
      throw new Error(
        "File is too large for the detection service. Try a shorter clip or a smaller file.",
      );
    }
    if (res.status === 502 || res.status === 503 || res.status === 504) {
      throw new Error(
        "The backend may be cold-starting — try again in 30–60 seconds.",
      );
    }
    if (!res.ok) {
      throw new Error(
        `Detection service returned ${res.status} ${res.statusText}.`,
      );
    }
    throw new Error("Detection service returned a non-JSON response.");
  }
  if (!res.ok) {
    const detail =
      (parsed as { detail?: unknown; error?: unknown })?.detail ??
      (parsed as { error?: unknown })?.error ??
      raw;
    if (res.status === 413) {
      throw new Error(
        typeof detail === "string"
          ? detail
          : "File is too large for the detection service.",
      );
    }
    throw new Error(
      typeof detail === "string" ? detail : `Request failed (${res.status})`,
    );
  }
  if (
    parsed &&
    typeof parsed === "object" &&
    "error" in parsed &&
    typeof (parsed as { error: unknown }).error === "string"
  ) {
    throw new Error((parsed as { error: string }).error);
  }
  return parsed;
}

async function upload(kind: "image" | "video" | "audio", file: File) {
  const fd = new FormData();
  fd.append("file", file);
  const headers = await authHeaders();
  const res = await fetch(detectUrl(kind), {
    method: "POST",
    body: fd,
    headers,
  });
  return parseOrThrow(res);
}

export async function detectImage(file: File): Promise<ImageResult> {
  return upload("image", file) as Promise<ImageResult>;
}

export async function detectVideo(file: File): Promise<VideoResult> {
  return upload("video", file) as Promise<VideoResult>;
}

export async function detectAudio(file: File): Promise<AudioResult> {
  return upload("audio", file) as Promise<AudioResult>;
}

export async function detectText(text: string): Promise<TextResult> {
  const headers = { "Content-Type": "application/json", ...(await authHeaders()) };
  const res = await fetch(detectUrl("text"), {
    method: "POST",
    headers,
    body: JSON.stringify({ text }),
  });
  return parseOrThrow(res) as Promise<TextResult>;
}

/** Fire-and-forget request to wake up the backend. Call on pages that are
 *  about to issue a detection so the HF Space container is warm by the
 *  time the user submits. */
export function warmBackend(): void {
  if (!BACKEND) return;
  try {
    fetch(`${BACKEND}/health`, { cache: "no-store" }).catch(() => {});
  } catch {
    // ignore
  }
}

export function verdictTone(suspicion: number) {
  if (suspicion < 0.3) return { tone: "authentic", label: "Authentic", color: "forest" } as const;
  if (suspicion < 0.55) return { tone: "inconclusive", label: "Inconclusive", color: "amber" } as const;
  if (suspicion < 0.75) return { tone: "suspicious", label: "Suspicious", color: "ember" } as const;
  return { tone: "manipulated", label: "Manipulated", color: "alert" } as const;
}
