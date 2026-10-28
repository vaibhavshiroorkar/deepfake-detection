// Holds the ONNX sessions and runs the pipeline off the main thread, so the
// page stays responsive. The page decodes video (only it has a <video>
// element) and posts frames here as RGB buffers.
import * as ort from "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/ort.webgpu.min.mjs";
import * as P from "./pipeline.js";

const ORT_DIST = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/";
const MODELS = "https://huggingface.co/vaibhavshiroorkar/deepfake-generalization-demo/resolve/main/browser/";
const FILES = ["pnet", "rnet", "onet", "visual-best", "dinov3"];

ort.env.wasm.wasmPaths = ORT_DIST;

let sessions = null;
let gate = null;
let backend = "wasm";

const post = (type, data = {}) => self.postMessage({ type, ...data });

// Cache Storage keeps the 110 MB of weights across visits. It can be missing
// or refuse writes (private windows, full disks), so every use is optional.
async function fetchCached(url, onBytes) {
  let cache = null;
  try {
    cache = await caches.open("ddf-models-v1");
    const hit = await cache.match(url);
    if (hit) {
      const buffer = await hit.arrayBuffer();
      onBytes(buffer.byteLength);
      return buffer;
    }
  } catch {
    cache = null;
  }
  const response = await fetch(url);
  if (!response.ok) throw new Error(`Could not download ${url.split("/").pop()} (${response.status})`);
  const reader = response.body.getReader();
  const parts = [];
  let total = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    parts.push(value);
    total += value.length;
    onBytes(value.length);
  }
  const buffer = new Uint8Array(total);
  let offset = 0;
  for (const part of parts) {
    buffer.set(part, offset);
    offset += part.length;
  }
  if (cache) {
    try {
      await cache.put(url, new Response(buffer.slice()));
    } catch {
      // Not cached; the next visit downloads again.
    }
  }
  return buffer.buffer;
}

async function create(buffer, providers) {
  return ort.InferenceSession.create(buffer, { executionProviders: providers, graphOptimizationLevel: "all" });
}

async function load(requested) {
  let loaded = 0;
  const onBytes = (n) => {
    loaded += n;
    post("download", { loaded });
  };
  const buffers = {};
  for (const name of FILES) buffers[name] = await fetchCached(`${MODELS}${name}.onnx`, onBytes);
  gate = JSON.parse(new TextDecoder().decode(await fetchCached(`${MODELS}gate.json`, onBytes)));

  const gpu = requested !== "wasm" && navigator.gpu && (await navigator.gpu.requestAdapter().catch(() => null));
  backend = gpu ? "webgpu" : "wasm";
  const heavy = gpu ? ["webgpu", "wasm"] : ["wasm"];
  sessions = {};
  // MTCNN's nets are tiny and run on many odd shapes; wasm handles them best.
  for (const name of ["pnet", "rnet", "onet"]) sessions[name] = await create(buffers[name], ["wasm"]);
  for (const name of ["visual-best", "dinov3"]) {
    try {
      sessions[name] = await create(buffers[name], heavy);
    } catch {
      backend = "wasm";
      sessions[name] = await create(buffers[name], ["wasm"]);
    }
  }
  post("ready", { backend, gateSize: gate.size, windows: gate.windows, windowLength: gate.window_length });
}

const asFrame = ({ width, height, buffer }) => ({ width, height, data: new Uint8Array(buffer) });

async function analyse({ frames, windows }) {
  const nets = { pnet: sessions.pnet, rnet: sessions.rnet, onet: sessions.onet };
  const decoded = frames.map(asFrame);
  const detections = [];
  for (let i = 0; i < decoded.length; i++) {
    detections.push(await P.detectFaces(ort, nets, decoded[i]));
    post("progress", { stage: "faces", done: i + 1, total: decoded.length });
  }
  const track = P.selectPrimaryTrack(detections);
  const view = P.faceView(decoded, track);
  let score = null;
  if (view) {
    post("progress", { stage: "score" });
    score = await P.scoreFaces(ort, sessions["visual-best"], view.tensor);
  }
  post("faces", {
    stable: track.stable,
    coverage: track.coverage,
    crops: view ? view.crops.map((c) => c.buffer) : [],
    score,
  });

  // The page already shrank these to gate.size with INTER_AREA.
  const resized = windows.map((list) => list.map((buffer) => new Uint8Array(buffer)));
  let done = 0;
  const features = await P.embedWindows(ort, sessions.dinov3, resized, gate.size, () => {
    done += 1;
    post("progress", { stage: "familiarity", done, total: resized.length });
  });
  const check = P.familiarity(gate, features);
  post("result", { score, stable: track.stable, coverage: track.coverage, familiarity: { ...check, reference: gate.reference } });
}

self.onmessage = async ({ data }) => {
  try {
    if (data.type === "load") await load(data.backend);
    if (data.type === "analyse") await analyse(data);
  } catch (error) {
    post("error", { message: error && error.message ? error.message : String(error) });
  }
};
