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

export type ImageResult = BaseResult & {
  kind: "image";
  filename: string;
  dimensions: { width: number; height: number };
};

export type VideoResult = BaseResult & {
  kind: "video";
  filename: string;
  duration_seconds: number;
  dimensions: { width: number; height: number };
  timeline: { timestamp: number; suspicion: number; verdict: string }[];
};

export type AudioResult = BaseResult & {
  kind: "audio";
  filename: string;
  duration_seconds: number;
  sample_rate: number;
};

export type TextResult = BaseResult & {
  kind: "text";
  length: { characters: number; words: number; sentences: number };
};

export type DetectionResult = ImageResult | VideoResult | AudioResult | TextResult;

async function upload(kind: "image" | "video" | "audio", file: File) {
  const fd = new FormData();
  fd.append("file", file);
  const res = await fetch(`/api/detect?kind=${kind}`, { method: "POST", body: fd });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
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
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export function verdictTone(suspicion: number) {
  if (suspicion < 0.3) return { tone: "authentic", label: "Authentic", color: "forest" } as const;
  if (suspicion < 0.55) return { tone: "inconclusive", label: "Inconclusive", color: "amber" } as const;
  if (suspicion < 0.75) return { tone: "suspicious", label: "Suspicious", color: "ember" } as const;
  return { tone: "manipulated", label: "Manipulated", color: "alert" } as const;
}
