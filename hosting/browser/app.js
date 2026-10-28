// Page side of the in-browser demo: decode the chosen video, hand frames to
// the worker, and show what came back. Nothing leaves the browser.
import { resizeArea, uniformTimestamps, windowStarts, VISUAL_FRAMES, VISUAL_SIZE } from "./pipeline.js";

const THRESHOLD = 0.5;
const MODEL_MB = 110;
const $ = (id) => document.getElementById(id);

const worker = new Worker(new URL("./worker.js", import.meta.url), { type: "module" });
let config = null;
let busy = false;

function setStatus(text) {
  $("status").textContent = text;
}

function setStage(stage, state, detail = "") {
  const item = document.querySelector(`[data-stage="${stage}"]`);
  if (!item) return;
  item.dataset.state = state;
  item.querySelector(".detail").textContent = detail;
}

worker.onmessage = ({ data }) => {
  if (data.type === "download") {
    const mb = Math.min(MODEL_MB, Math.round(data.loaded / 1e6));
    setStatus(`Downloading models: ${mb} of about ${MODEL_MB} MB. This happens once.`);
  } else if (data.type === "ready") {
    config = data;
    setStatus(`Models ready. Running on ${data.backend === "webgpu" ? "your GPU (WebGPU)" : "your CPU (WebAssembly)"}.`);
    $("drop").classList.remove("disabled");
    $("file").disabled = false;
  } else if (data.type === "progress") {
    if (data.stage === "faces") setStage("faces", "active", `${data.done} of ${data.total} frames`);
    if (data.stage === "score") { setStage("faces", "done"); setStage("score", "active"); }
    if (data.stage === "familiarity") setStage("familiarity", "active", `${data.done} of ${data.total} windows`);
  } else if (data.type === "faces") {
    setStage("faces", "done");
    setStage("score", "done", data.score ? `p = ${data.score.probability.toFixed(3)}` : "no steady face");
    setStage("familiarity", "active");
    showFaces(data.crops);
  } else if (data.type === "result") {
    setStage("familiarity", "done", `distance ${data.familiarity.distance.toFixed(2)}`);
    showResult(data);
    busy = false;
  } else if (data.type === "error") {
    setStatus(`Something went wrong: ${data.message}`);
    busy = false;
  }
};
// `?backend=wasm` forces the CPU path, for comparing results across backends.
worker.postMessage({ type: "load", backend: new URLSearchParams(location.search).get("backend") });

// ---------------------------------------------------------------- decoding

const once = (target, event) => new Promise((resolve, reject) => {
  target.addEventListener(event, resolve, { once: true });
  target.addEventListener("error", () => reject(new Error(
    "This browser cannot play that video's format. Phone clips (H.264 MP4) work; older MPEG-4 Part 2 files need converting first.",
  )), { once: true });
});

async function openVideo(file) {
  const video = document.createElement("video");
  video.muted = true;
  video.playsInline = true;
  video.preload = "auto";
  video.src = URL.createObjectURL(file);
  await once(video, "loadedmetadata");
  // Recorded WebM often reports an infinite duration until the end is read.
  if (!Number.isFinite(video.duration)) {
    video.currentTime = 1e9;
    await once(video, "seeked");
  }
  return video;
}

// The browser does not expose a frame rate, so read it from the gaps between
// presented frames while the clip plays silently for a moment.
async function frameRate(video) {
  if (!("requestVideoFrameCallback" in video)) return 30;
  const times = [];
  await new Promise((resolve) => {
    const step = (_, meta) => {
      times.push(meta.mediaTime);
      if (times.length < 12) video.requestVideoFrameCallback(step);
      else resolve();
    };
    video.requestVideoFrameCallback(step);
    video.play().catch(resolve);
    setTimeout(resolve, 3000);
  });
  video.pause();
  const gaps = times.slice(1).map((t, i) => t - times[i]).filter((g) => g > 0).sort((a, b) => a - b);
  if (gaps.length === 0) return 30;
  return 1 / gaps[Math.floor(gaps.length / 2)];
}

async function frameAt(video, time, canvas, context) {
  video.currentTime = Math.min(Math.max(0, time), Math.max(0, video.duration - 1e-3));
  await once(video, "seeked");
  context.drawImage(video, 0, 0, canvas.width, canvas.height);
  const rgba = context.getImageData(0, 0, canvas.width, canvas.height).data;
  const rgb = new Uint8Array((rgba.length / 4) * 3);
  for (let i = 0, j = 0; i < rgba.length; i += 4, j += 3) {
    rgb[j] = rgba[i];
    rgb[j + 1] = rgba[i + 1];
    rgb[j + 2] = rgba[i + 2];
  }
  return { width: canvas.width, height: canvas.height, data: rgb };
}

async function decode(file) {
  const video = await openVideo(file);
  const canvas = document.createElement("canvas");
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  const context = canvas.getContext("2d", { willReadFrequently: true });
  const duration = video.duration;
  const fps = await frameRate(video);

  // OpenCV's millisecond seek returns the frame nearest the timestamp. The
  // browser shows the frame whose interval contains it, one earlier about
  // half the time, so seek to the middle of OpenCV's frame instead.
  const lastFrame = Math.max(0, Math.round(duration * fps) - 1);
  const frames = [];
  for (const time of uniformTimestamps(duration, VISUAL_FRAMES)) {
    const index = Math.min(lastFrame, Math.round(time * fps));
    frames.push(await frameAt(video, (index + 0.5) / fps, canvas, context));
    setStage("read", "active", `${frames.length} of ${VISUAL_FRAMES + config.windows * config.windowLength} frames`);
  }

  // Three runs of consecutive frames for the familiarity check, shrunk here so
  // the worker gets small buffers instead of full-size ones.
  const total = Math.max(1, Math.round(duration * fps));
  const windows = [];
  let read = frames.length;
  for (const start of windowStarts(total, config.windows, config.windowLength)) {
    const picked = [];
    for (let index = start; index < Math.min(total, start + config.windowLength); index++) {
      const frame = await frameAt(video, (index + 0.5) / fps, canvas, context);
      picked.push(resizeArea(frame, config.gateSize, config.gateSize).buffer);
      read += 1;
      setStage("read", "active", `${read} of ${VISUAL_FRAMES + config.windows * config.windowLength} frames`);
    }
    while (picked.length < config.windowLength) picked.push(picked[picked.length - 1].slice(0));
    windows.push(picked);
  }
  URL.revokeObjectURL(video.src);
  return { frames, windows };
}

// ------------------------------------------------------------------ results

function showFaces(crops) {
  const strip = $("faces");
  strip.replaceChildren();
  for (const buffer of crops) {
    const rgb = new Uint8Array(buffer);
    const canvas = document.createElement("canvas");
    canvas.width = VISUAL_SIZE;
    canvas.height = VISUAL_SIZE;
    const context = canvas.getContext("2d");
    const image = context.createImageData(VISUAL_SIZE, VISUAL_SIZE);
    for (let i = 0, j = 0; i < rgb.length; i += 3, j += 4) {
      image.data[j] = rgb[i];
      image.data[j + 1] = rgb[i + 1];
      image.data[j + 2] = rgb[i + 2];
      image.data[j + 3] = 255;
    }
    context.putImageData(image, 0, 0);
    strip.append(canvas);
  }
  $("faces-wrap").hidden = crops.length === 0;
}

function showResult({ score, stable, coverage, familiarity }) {
  const verdict = $("verdict");
  const familiar = familiarity.familiar;
  let label;
  let kind;
  let text;
  if (!stable || !score) {
    kind = "held";
    label = "No steady face";
    text = `The face detector did not find one face across most of the clip (found in ${Math.round(coverage * 100)} percent of sampled frames). The model needs a single, steady face.`;
  } else if (!familiar) {
    kind = "held";
    label = "Withheld";
    text = `This clip sits ${familiarity.distance.toFixed(1)} from the genuine footage the model was trained on, past the ${familiarity.threshold.toFixed(1)} that ${familiarity.reference} clips stay within. On footage this unfamiliar the detector calls genuine video fake about nine times in ten, so it gives no verdict.`;
  } else {
    const fake = score.probability >= THRESHOLD;
    kind = fake ? "fake" : "real";
    label = fake ? "Looks fake" : "Looks real";
    text = `The clip looks like the training footage (distance ${familiarity.distance.toFixed(1)}, limit ${familiarity.threshold.toFixed(1)}), so the model answers. That still only means it resembles FakeAVCeleb, not that the answer is right.`;
  }
  verdict.dataset.kind = kind;
  $("verdict-label").textContent = label;
  $("verdict-text").textContent = text;
  $("raw").textContent = score
    ? `Model score: ${score.probability.toFixed(3)} (fake at ${THRESHOLD} or above). Familiarity distance: ${familiarity.distance.toFixed(2)}, limit ${familiarity.threshold.toFixed(2)}.`
    : `Familiarity distance: ${familiarity.distance.toFixed(2)}, limit ${familiarity.threshold.toFixed(2)}.`;
  $("result").hidden = false;
  setStatus("Done. Pick another clip to run again.");
}

// -------------------------------------------------------------------- input

async function run(file) {
  if (busy || !config) return;
  if (!file.type.startsWith("video/")) {
    setStatus("That is not a video file. Try an MP4, WebM or MOV clip.");
    return;
  }
  busy = true;
  $("result").hidden = true;
  $("faces-wrap").hidden = true;
  $("steps").hidden = false;
  for (const stage of ["read", "faces", "score", "familiarity"]) setStage(stage, "waiting");
  setStage("read", "active");
  setStatus(`Reading ${file.name}`);
  try {
    const { frames, windows } = await decode(file);
    setStage("read", "done");
    setStage("faces", "active");
    const transfer = [...frames.map((f) => f.data.buffer), ...windows.flat()];
    worker.postMessage({
      type: "analyse",
      frames: frames.map((f) => ({ width: f.width, height: f.height, buffer: f.data.buffer })),
      windows,
    }, transfer);
  } catch (error) {
    setStatus(error.message);
    $("steps").hidden = true;
    busy = false;
  }
}

$("file").addEventListener("change", (event) => {
  if (event.target.files[0]) run(event.target.files[0]);
  event.target.value = "";
});
const drop = $("drop");
drop.addEventListener("dragover", (event) => { event.preventDefault(); drop.classList.add("over"); });
drop.addEventListener("dragleave", () => drop.classList.remove("over"));
drop.addEventListener("drop", (event) => {
  event.preventDefault();
  drop.classList.remove("over");
  if (event.dataTransfer.files[0]) run(event.dataTransfer.files[0]);
});
