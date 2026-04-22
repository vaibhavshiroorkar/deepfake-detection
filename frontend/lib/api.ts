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

async function readError(res: Response): Promise<string> {
  const raw = await res.text();
  try {
    const parsed = JSON.parse(raw);
    if (typeof parsed === "string") return parsed;
    if (parsed?.detail) {
      if (typeof parsed.detail === "string") return parsed.detail;
      return JSON.stringify(parsed.detail);
    }
    if (typeof parsed?.error === "string") return parsed.error;
    return raw || `Request failed (${res.status})`;
  } catch {
    return raw || `Request failed (${res.status})`;
  }
}

async function upload(kind: "image" | "video" | "audio", file: File) {
  const fd = new FormData();
  fd.append("file", file);
  const res = await fetch(`/api/detect?kind=${kind}`, { method: "POST", body: fd });
  if (!res.ok) throw new Error(await readError(res));
  const data = await res.json();
  if (data?.error) throw new Error(data.error);
  return data;
}

export async function detectImage(file: File): Promise<ImageResult> {
  return upload("image", file);
}

export async function detectVideo(file: File): Promise<VideoResult> {
  return upload("video", file);
}

export async function detectAudio(file: File): Promise<AudioResult> {
  return upload("audio", file);
}

export async function detectText(text: string): Promise<TextResult> {
  const res = await fetch("/api/detect?kind=text", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text }),
  });
  if (!res.ok) throw new Error(await readError(res));
  return res.json();
}

export function verdictTone(suspicion: number) {
  if (suspicion < 0.3) return { tone: "authentic", label: "Authentic", color: "forest" } as const;
  if (suspicion < 0.55) return { tone: "inconclusive", label: "Inconclusive", color: "amber" } as const;
  if (suspicion < 0.75) return { tone: "suspicious", label: "Suspicious", color: "ember" } as const;
  return { tone: "manipulated", label: "Manipulated", color: "alert" } as const;
}
