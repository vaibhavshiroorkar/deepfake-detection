// The served visual-best path, ported from Python so it runs in a browser.
//
// Each function names the Python it mirrors. Frames are {width, height, data}
// with data an RGB Uint8Array, row-major. Model calls go through an `ort`
// object (onnxruntime-web in the page, onnxruntime-node in the parity test),
// so this file has no browser or Node dependency of its own.

export const VISUAL_FRAMES = 16;
export const VISUAL_SIZE = 224;
export const CROP_MARGIN = 0.2;
export const DETECTOR_CONFIDENCE = 0.8;
const IMAGENET_MEAN = [0.485, 0.456, 0.406];
const IMAGENET_STD = [0.229, 0.224, 0.225];

// Python's round(): halves go to the even neighbour.
export function roundHalfEven(value) {
  const floor = Math.floor(value);
  const diff = value - floor;
  if (diff > 0.5) return floor + 1;
  if (diff < 0.5) return floor;
  return floor % 2 === 0 ? floor : floor + 1;
}

// views/timeline.py: uniform_timestamps
export function uniformTimestamps(duration, count, start = 0) {
  const step = (duration - start) / count;
  return Array.from({ length: count }, (_, i) => start + (i + 0.5) * step);
}

// inference/frozen_features.py: window_starts
export function windowStarts(total, count, length) {
  if (total <= length) return [0];
  if (count === 1) return [Math.max(0, Math.floor((total - length) / 2))];
  const span = total - length;
  return Array.from({ length: count }, (_, i) => roundHalfEven((span * i) / (count - 1)));
}

// ---------------------------------------------------------------- resizing

// torch.nn.functional.interpolate(mode="area"), which MTCNN uses. That is
// adaptive average pooling, so each output cell averages a whole-pixel window.
function resizeAreaTorch(chw, channels, height, width, outH, outW) {
  const out = new Float32Array(channels * outH * outW);
  for (let oy = 0; oy < outH; oy++) {
    const y0 = Math.floor((oy * height) / outH);
    const y1 = Math.ceil(((oy + 1) * height) / outH);
    for (let ox = 0; ox < outW; ox++) {
      const x0 = Math.floor((ox * width) / outW);
      const x1 = Math.ceil(((ox + 1) * width) / outW);
      const count = (y1 - y0) * (x1 - x0);
      for (let c = 0; c < channels; c++) {
        const plane = c * height * width;
        let sum = 0;
        for (let y = y0; y < y1; y++) {
          const row = plane + y * width;
          for (let x = x0; x < x1; x++) sum += chw[row + x];
        }
        out[c * outH * outW + oy * outW + ox] = sum / count;
      }
    }
  }
  return out;
}

// OpenCV's linear resize taps: float32 offsets and 11-bit integer weights.
function linearTaps(size, out) {
  const scale = 1 / (out / size);
  const f = Math.fround;
  return Array.from({ length: out }, (_, o) => {
    const position = f((o + 0.5) * scale - 0.5);
    let start = Math.floor(position);
    let fraction = f(position - start);
    if (start < 0) { start = 0; fraction = 0; }
    if (start >= size - 1) { start = size - 1; fraction = 0; }
    return [
      start,
      Math.min(start + 1, size - 1),
      roundHalfEven(f(f(1 - fraction) * 2048)),
      roundHalfEven(f(fraction * 2048)),
    ];
  });
}

// cv2.resize(INTER_LINEAR) on uint8 RGB, copying OpenCV's fixed-point
// arithmetic and its vectorised rounding. Checked against cv2 on real crops:
// only the first and last rows of an upscaled crop can differ, by one.
export function resizeLinear(frame, left, top, right, bottom, outW, outH) {
  const srcW = right - left;
  const srcH = bottom - top;
  const xs = linearTaps(srcW, outW);
  const ys = linearTaps(srcH, outH);
  const horizontal = (y) => {
    const row = new Int32Array(outW * 3);
    const base = ((top + y) * frame.width + left) * 3;
    xs.forEach(([x0, x1, a0, a1], ox) => {
      for (let c = 0; c < 3; c++) {
        row[ox * 3 + c] = frame.data[base + x0 * 3 + c] * a0 + frame.data[base + x1 * 3 + c] * a1;
      }
    });
    return row;
  };
  const rows = new Map();
  const rowAt = (y) => {
    if (!rows.has(y)) rows.set(y, horizontal(y));
    return rows.get(y);
  };
  const out = new Uint8Array(outW * outH * 3);
  ys.forEach(([y0, y1, b0, b1], oy) => {
    const r0 = rowAt(y0);
    const r1 = rowAt(y1);
    for (let i = 0; i < outW * 3; i++) {
      const value = (Math.floor((b0 * (r0[i] >> 4)) / 65536) + Math.floor((b1 * (r1[i] >> 4)) / 65536) + 2) >> 2;
      out[oy * outW * 3 + i] = value < 0 ? 0 : value > 255 ? 255 : value;
    }
  });
  return out;
}

// cv2.resize(INTER_AREA) when shrinking: each output pixel is the
// area-weighted mean of the source pixels it covers, fractions included.
export function resizeArea(frame, outW, outH) {
  if (frame.width < outW || frame.height < outH) {
    return resizeLinear(frame, 0, 0, frame.width, frame.height, outW, outH);
  }
  const spans = (size, out) => {
    const scale = size / out;
    return Array.from({ length: out }, (_, o) => {
      const start = o * scale;
      const end = start + scale;
      const parts = [];
      for (let i = Math.floor(start); i < Math.min(size, Math.ceil(end)); i++) {
        const w = Math.min(end, i + 1) - Math.max(start, i);
        if (w > 1e-9) parts.push([i, w / scale]);
      }
      return parts;
    });
  };
  const xSpans = spans(frame.width, outW);
  const ySpans = spans(frame.height, outH);
  const out = new Uint8Array(outW * outH * 3);
  for (let oy = 0; oy < outH; oy++) {
    for (let ox = 0; ox < outW; ox++) {
      let r = 0, g = 0, b = 0;
      for (const [y, wy] of ySpans[oy]) {
        const row = y * frame.width;
        for (const [x, wx] of xSpans[ox]) {
          const i = (row + x) * 3;
          const w = wy * wx;
          r += frame.data[i] * w;
          g += frame.data[i + 1] * w;
          b += frame.data[i + 2] * w;
        }
      }
      const o = (oy * outW + ox) * 3;
      out[o] = Math.round(r);
      out[o + 1] = Math.round(g);
      out[o + 2] = Math.round(b);
    }
  }
  return out;
}

// -------------------------------------------------------------------- MTCNN
// facenet_pytorch/models/utils/detect_face.py with MTCNN's defaults:
// min_face_size 20, thresholds 0.6/0.7/0.7, factor 0.709.

const MIN_FACE = 20;
const THRESHOLDS = [0.6, 0.7, 0.7];
const FACTOR = 0.709;

function toCHW(frame) {
  const plane = frame.width * frame.height;
  const out = new Float32Array(3 * plane);
  for (let i = 0; i < plane; i++) {
    out[i] = frame.data[i * 3];
    out[plane + i] = frame.data[i * 3 + 1];
    out[2 * plane + i] = frame.data[i * 3 + 2];
  }
  return out;
}

function normalise(values) {
  for (let i = 0; i < values.length; i++) values[i] = (values[i] - 127.5) * 0.0078125;
  return values;
}

// torchvision.ops.nms: IoU without the +1, highest score first. The IoU is a
// float32 in torch, and PNet's grid-aligned boxes often land on exactly the
// threshold, so the float64 value here would keep boxes torch drops.
function nmsTorch(boxes, threshold) {
  const order = boxes.map((_, i) => i).sort((a, b) => boxes[b][4] - boxes[a][4]);
  const keep = [];
  const removed = new Uint8Array(boxes.length);
  for (const i of order) {
    if (removed[i]) continue;
    keep.push(i);
    const [ax1, ay1, ax2, ay2] = boxes[i];
    const areaA = (ax2 - ax1) * (ay2 - ay1);
    for (const j of order) {
      if (removed[j] || j === i) continue;
      const [bx1, by1, bx2, by2] = boxes[j];
      const w = Math.max(0, Math.min(ax2, bx2) - Math.max(ax1, bx1));
      const h = Math.max(0, Math.min(ay2, by2) - Math.max(ay1, by1));
      const inter = w * h;
      const union = areaA + (bx2 - bx1) * (by2 - by1) - inter;
      if (union > 0 && Math.fround(inter / union) > threshold) removed[j] = 1;
    }
  }
  return keep;
}

// detect_face.py: nms_numpy with method "Min" and the +1 areas.
function nmsMin(boxes, threshold) {
  let order = boxes.map((_, i) => i).sort((a, b) => boxes[a][4] - boxes[b][4]);
  const area = boxes.map(([x1, y1, x2, y2]) => (x2 - x1 + 1) * (y2 - y1 + 1));
  const keep = [];
  while (order.length > 0) {
    const i = order[order.length - 1];
    keep.push(i);
    const rest = order.slice(0, -1);
    order = rest.filter((j) => {
      const w = Math.max(0, Math.min(boxes[i][2], boxes[j][2]) - Math.max(boxes[i][0], boxes[j][0]) + 1);
      const h = Math.max(0, Math.min(boxes[i][3], boxes[j][3]) - Math.max(boxes[i][1], boxes[j][1]) + 1);
      return (w * h) / Math.min(area[i], area[j]) <= threshold;
    });
  }
  return keep;
}

function rerec(box) {
  const h = box[3] - box[1];
  const w = box[2] - box[0];
  const l = Math.max(w, h);
  box[0] = box[0] + w * 0.5 - l * 0.5;
  box[1] = box[1] + h * 0.5 - l * 0.5;
  box[2] = box[0] + l;
  box[3] = box[1] + l;
  return box;
}

function bbreg(box, reg) {
  const w = box[2] - box[0] + 1;
  const h = box[3] - box[1] + 1;
  return [box[0] + reg[0] * w, box[1] + reg[1] * h, box[2] + reg[2] * w, box[3] + reg[3] * h, box[4]];
}

// detect_face.py: pad, then the crop and resample it feeds.
function cropResampled(chw, width, height, box, size) {
  let x = Math.trunc(box[0]), y = Math.trunc(box[1]);
  let ex = Math.trunc(box[2]), ey = Math.trunc(box[3]);
  if (x < 1) x = 1;
  if (y < 1) y = 1;
  if (ex > width) ex = width;
  if (ey > height) ey = height;
  if (!(ey > y - 1 && ex > x - 1)) return null;
  const cw = ex - (x - 1), ch = ey - (y - 1);
  const crop = new Float32Array(3 * cw * ch);
  for (let c = 0; c < 3; c++) {
    for (let r = 0; r < ch; r++) {
      const src = c * width * height + (y - 1 + r) * width + (x - 1);
      crop.set(chw.subarray(src, src + cw), c * cw * ch + r * cw);
    }
  }
  return normalise(resizeAreaTorch(crop, 3, ch, cw, size, size));
}

async function runCrops(ort, session, chw, width, height, boxes, size) {
  const kept = [];
  const crops = [];
  boxes.forEach((box, index) => {
    const crop = cropResampled(chw, width, height, box, size);
    if (crop) { kept.push(index); crops.push(crop); }
  });
  if (crops.length === 0) return { kept, outputs: null };
  const batch = new Float32Array(crops.length * 3 * size * size);
  crops.forEach((crop, i) => batch.set(crop, i * crop.length));
  const input = new ort.Tensor("float32", batch, [crops.length, 3, size, size]);
  return { kept, outputs: await session.run({ input }) };
}

// MTCNN.detect plus views/face_detector.py: boxes as {box, confidence},
// filtered at the detector confidence and sorted by confidence.
export async function detectFaces(ort, nets, frame) {
  const { width, height } = frame;
  const chw = toCHW(frame);
  const m = 12 / MIN_FACE;
  let minl = Math.min(height, width) * m;
  const scales = [];
  let scale = m;
  while (minl >= 12) {
    scales.push(scale);
    scale *= FACTOR;
    minl *= FACTOR;
  }

  let boxes = [];
  for (const s of scales) {
    const hs = Math.trunc(height * s + 1);
    const ws = Math.trunc(width * s + 1);
    const data = normalise(resizeAreaTorch(chw, 3, height, width, hs, ws));
    const input = new ort.Tensor("float32", data, [1, 3, hs, ws]);
    const { reg, prob } = await nets.pnet.run({ input });
    const [, , oh, ow] = prob.dims;
    const plane = oh * ow;
    // generateBoundingBox divides in float32. At scales like 0.6 the exact
    // quotient is often a whole number that float64 lands just below.
    const s32 = Math.fround(s);
    const edge = (v) => Math.floor(Math.fround(v / s32));
    const found = [];
    for (let yy = 0; yy < oh; yy++) {
      for (let xx = 0; xx < ow; xx++) {
        const k = yy * ow + xx;
        const score = prob.data[plane + k];
        if (score < THRESHOLDS[0]) continue;
        found.push([
          edge(2 * xx + 1),
          edge(2 * yy + 1),
          edge(2 * xx + 12),
          edge(2 * yy + 12),
          score,
          reg.data[k], reg.data[plane + k], reg.data[2 * plane + k], reg.data[3 * plane + k],
        ]);
      }
    }
    boxes.push(...nmsTorch(found, 0.5).map((i) => found[i]));
  }
  boxes = nmsTorch(boxes, 0.7).map((i) => boxes[i]);
  boxes = boxes.map((b) => {
    const rw = b[2] - b[0];
    const rh = b[3] - b[1];
    return rerec([b[0] + b[5] * rw, b[1] + b[6] * rh, b[2] + b[7] * rw, b[3] + b[8] * rh, b[4]]);
  });
  if (boxes.length === 0) return [];

  let stage = await runCrops(ort, nets.rnet, chw, width, height, boxes, 24);
  if (!stage.outputs) return [];
  let next = [];
  stage.kept.forEach((index, row) => {
    const score = stage.outputs.prob.data[row * 2 + 1];
    if (score > THRESHOLDS[1]) {
      const reg = stage.outputs.reg.data.slice(row * 4, row * 4 + 4);
      next.push({ box: [...boxes[index].slice(0, 4), score], reg });
    }
  });
  next = nmsTorch(next.map((n) => n.box), 0.7).map((i) => next[i]);
  boxes = next.map((n) => rerec(bbreg(n.box, n.reg)));
  if (boxes.length === 0) return [];

  stage = await runCrops(ort, nets.onet, chw, width, height, boxes, 48);
  if (!stage.outputs) return [];
  const final = [];
  stage.kept.forEach((index, row) => {
    const score = stage.outputs.prob.data[row * 2 + 1];
    if (score > THRESHOLDS[2]) {
      const reg = stage.outputs.reg.data.slice(row * 4, row * 4 + 4);
      final.push(bbreg([...boxes[index].slice(0, 4), score], reg));
    }
  });
  const kept = nmsMin(final, 0.7).map((i) => final[i]);
  return kept
    .map((b) => ({ box: { left: b[0], top: b[1], right: b[2], bottom: b[3] }, confidence: b[4] }))
    .filter((d) => d.confidence >= DETECTOR_CONFIDENCE)
    .sort((a, b) => b.confidence - a.confidence);
}

// ----------------------------------------------------------------- tracking

function boxArea(b) {
  return Math.max(0, b.right - b.left) * Math.max(0, b.bottom - b.top);
}

function iou(a, b) {
  const inter = boxArea({
    left: Math.max(a.left, b.left),
    top: Math.max(a.top, b.top),
    right: Math.min(a.right, b.right),
    bottom: Math.min(a.bottom, b.bottom),
  });
  const union = boxArea(a) + boxArea(b) - inter;
  return union ? inter / union : 0;
}

// views/tracking.py: select_primary_track with greedy IoU association.
export function selectPrimaryTrack(frames, minIou = 0.3, minCoverage = 0.8, minDominance = 1.25) {
  const tracks = [];
  frames.forEach((detections, frameIndex) => {
    const unused = new Set(detections.map((_, i) => i));
    const ordered = [...tracks].sort((a, b) => b.length - a.length);
    for (const track of ordered) {
      if (unused.size === 0) continue;
      let best = null;
      for (const index of unused) {
        const overlap = iou(track[track.length - 1][1].box, detections[index].box);
        // max() over (overlap, index) tuples: ties go to the larger index.
        if (!best || overlap > best[0] || (overlap === best[0] && index > best[1])) best = [overlap, index];
      }
      if (best[0] >= minIou) {
        track.push([frameIndex, detections[best[1]]]);
        unused.delete(best[1]);
      }
    }
    for (const index of [...unused].sort((a, b) => a - b)) tracks.push([[frameIndex, detections[index]]]);
  });
  if (tracks.length === 0) return { frameIndices: [], detections: [], coverage: 0, stable: false };
  const meanConfidence = (t) => t.reduce((s, [, d]) => s + d.confidence, 0) / t.length;
  const ranked = [...tracks].sort((a, b) => b.length - a.length || meanConfidence(b) - meanConfidence(a));
  const primary = ranked[0];
  const coverage = primary.length / frames.length;
  const second = ranked.length > 1 ? ranked[1].length : 0;
  const dominance = second ? primary.length / second : Infinity;
  return {
    frameIndices: primary.map(([i]) => i),
    detections: primary.map(([, d]) => d),
    coverage,
    stable: coverage >= minCoverage && dominance >= minDominance,
  };
}

// --------------------------------------------------------------- face view

// views/preprocessor.py: _expanded_square
export function expandedSquare(box, frame, margin = CROP_MARGIN) {
  const side = Math.max(box.right - box.left, box.bottom - box.top) * (1 + 2 * margin);
  const cx = (box.left + box.right) / 2;
  const cy = (box.top + box.bottom) / 2;
  return [
    Math.max(0, roundHalfEven(cx - side / 2)),
    Math.max(0, roundHalfEven(cy - side / 2)),
    Math.min(frame.width, roundHalfEven(cx + side / 2)),
    Math.min(frame.height, roundHalfEven(cy + side / 2)),
  ];
}

// views/preprocessor.py: _face_view with _filled_detections. Returns the
// normalised [16, 3, 224, 224] tensor data and the uint8 crops for display.
export function faceView(frames, track) {
  if (!track.stable) return null;
  const size = VISUAL_SIZE;
  const plane = size * size;
  const tensor = new Float32Array(frames.length * 3 * plane);
  const crops = [];
  frames.forEach((frame, index) => {
    let nearest = track.frameIndices[0];
    let position = 0;
    track.frameIndices.forEach((candidate, i) => {
      if (Math.abs(candidate - index) < Math.abs(nearest - index)) { nearest = candidate; position = i; }
    });
    const [left, top, right, bottom] = expandedSquare(track.detections[position].box, frame);
    const crop = resizeLinear(frame, left, top, right, bottom, size, size);
    crops.push(crop);
    const base = index * 3 * plane;
    for (let i = 0; i < plane; i++) {
      for (let c = 0; c < 3; c++) {
        tensor[base + c * plane + i] = (crop[i * 3 + c] / 255 - IMAGENET_MEAN[c]) / IMAGENET_STD[c];
      }
    }
  });
  return { tensor, crops };
}

export async function scoreFaces(ort, session, tensor) {
  const input = new ort.Tensor("float32", tensor, [1, VISUAL_FRAMES, 3, VISUAL_SIZE, VISUAL_SIZE]);
  const { logit } = await session.run({ input });
  const value = logit.data[0];
  return { logit: value, probability: 1 / (1 + Math.exp(-value)) };
}

// --------------------------------------------------------- familiarity gate

// inference/frozen_features.py: embed_windows. `windows` holds one array of
// 224x224 RGB frames per window, already resized with INTER_AREA.
export async function embedWindows(ort, session, windows, size, onBatch) {
  const plane = size * size;
  const perWindow = [];
  for (const frames of windows) {
    const batch = new Float32Array(frames.length * 3 * plane);
    frames.forEach((pixels, f) => {
      const base = f * 3 * plane;
      for (let i = 0; i < plane; i++) {
        for (let c = 0; c < 3; c++) batch[base + c * plane + i] = pixels[i * 3 + c] / 255;
      }
    });
    const input = new ort.Tensor("float32", batch, [frames.length, 3, size, size]);
    const { features } = await session.run({ input });
    const dim = features.dims[1];
    const mean = new Float64Array(dim);
    for (let f = 0; f < frames.length; f++) {
      for (let d = 0; d < dim; d++) mean[d] += features.data[f * dim + d];
    }
    perWindow.push(mean.map((v) => v / frames.length));
    if (onBatch) onBatch();
  }
  const dim = perWindow[0].length;
  const vector = new Float64Array(2 * dim);
  for (let d = 0; d < dim; d++) {
    const values = perWindow.map((w) => w[d]);
    const mean = values.reduce((s, v) => s + v, 0) / values.length;
    const spread = Math.sqrt(values.reduce((s, v) => s + (v - mean) ** 2, 0) / values.length);
    vector[d] = mean;
    vector[dim + d] = spread;
  }
  return vector;
}

// inference/authenticity.py: AuthenticityDensity.distance and check
export function familiarity(gate, features) {
  const reduced = gate.components.map((row, r) => {
    let sum = 0;
    for (let i = 0; i < row.length; i++) sum += (features[i] - gate.mean[i]) * row[i];
    return sum - gate.centre[r];
  });
  let total = 0;
  for (let i = 0; i < reduced.length; i++) {
    let row = 0;
    for (let j = 0; j < reduced.length; j++) row += gate.inverse[i][j] * reduced[j];
    total += reduced[i] * row;
  }
  const distance = Math.sqrt(total);
  return { distance, threshold: gate.threshold, familiar: distance <= gate.threshold };
}
