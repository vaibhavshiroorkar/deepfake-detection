"use strict";

const $ = (selector, root = document) => root.querySelector(selector);
const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];

const state = {
  config: null,
  research: null,
  file: null,
  sample: null,
  model: null,
  threshold: 0.5,
  // Untouched, the server picks the cut-off for the media kind: images and
  // sound files have their own fitted values.
  thresholdTouched: false,
  analysis: null,
};

const MODEL_NAMES = {
  "efficientnet-b0-seed17": "EfficientNet-B0, seed 17",
  "efficientnet-b0-seed29": "EfficientNet-B0, seed 29",
  "efficientnet-b0-seed43": "EfficientNet-B0, seed 43",
  "convnext-tiny-seed17": "ConvNeXt-Tiny, seed 17",
  "convnext-tiny-seed29": "ConvNeXt-Tiny, seed 29",
  "convnext-tiny-seed43": "ConvNeXt-Tiny, seed 43",
  "visual-efficientnet": "Visual stream, EfficientNet",
  "visual-dinov3": "Visual stream, DINOv3",
  "final-audio-seed17": "Voice stream",
  "stream-lipsync": "Lip-sync stream",
  "stream-emotion": "Emotion stream",
};

function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>"']/g, (c) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  })[c]);
}

const pct = (x) => (x == null ? "n/a" : `${(x * 100).toFixed(x < 0.1 && x > 0 ? 1 : 0)}%`);
const fmt = (x, digits = 4) => (x == null ? "" : Number(x).toFixed(digits));
function fmtProb(p) {
  if (p == null) return "n/a";
  if (p < 0.001) return "<0.001";
  if (p > 0.999) return ">0.999";
  return p < 0.1 || p > 0.9 ? p.toFixed(3) : p.toFixed(2);
}

/* Theme: remembered per browser, never required. */
function storedTheme() {
  try { return localStorage.getItem("ddf-theme"); } catch { return null; }
}
function applyTheme(theme) {
  if (theme) document.documentElement.dataset.theme = theme;
}
applyTheme(storedTheme());
$("#theme").addEventListener("click", () => {
  const dark = document.documentElement.dataset.theme
    ? document.documentElement.dataset.theme === "dark"
    : matchMedia("(prefers-color-scheme: dark)").matches;
  const next = dark ? "light" : "dark";
  applyTheme(next);
  try { localStorage.setItem("ddf-theme", next); } catch { /* private window */ }
  if (state.research) renderCharts();
});

/* Tabs */
function showTab(name) {
  $$(".tab").forEach((tab) => tab.setAttribute("aria-selected", String(tab.dataset.tab === name)));
  $$("main > section").forEach((section) => { section.hidden = section.id !== `tab-${name}`; });
  if (name !== "check") loadResearch();
  try { history.replaceState(null, "", `#${name}`); } catch { /* file URL */ }
}
$$(".tab").forEach((tab) => tab.addEventListener("click", () => showTab(tab.dataset.tab)));

/* Config */
async function loadConfig() {
  const response = await fetch("/api/config");
  state.config = await response.json();
  const { settings } = state.config;
  state.model = settings.default_model;
  state.threshold = settings.default_threshold;
  if (!settings.show_research_tabs) $$(".tab.research").forEach((tab) => tab.remove());
  if (!settings.allow_unfamiliar_override) $("#withhold").disabled = true;
  $("#withhold").checked = settings.withhold_unfamiliar;
  if (state.config.admin_enabled) {
    $("#footer").insertAdjacentHTML("beforeend", ' <a href="/admin">Admin settings</a>');
  }
  renderModels();
  renderSamples();
}

function renderModels() {
  const box = $("#models");
  box.innerHTML = state.config.models.map((model) => {
    const scores = Object.entries(model.metrics).map(([dataset, m]) =>
      `<span>${escapeHtml(dataset)} <b>${fmt(m.roc_auc, 3)}</b></span>`).join("");
    return `<label class="model">
      <input type="radio" name="model" value="${escapeHtml(model.id)}" ${model.id === state.model ? "checked" : ""}>
      <span class="model-name">${escapeHtml(model.label)}${model.id === "ensemble" ? " (recommended)" : ""}</span>
      <span class="model-summary">${escapeHtml(model.summary)}</span>
      ${scores ? `<span class="model-scores"><span class="quiet">ROC-AUC</span>${scores}</span>` : ""}
    </label>`;
  }).join("");
  $$("input[name=model]", box).forEach((input) => input.addEventListener("change", () => {
    state.model = input.value;
    updateHint();
  }));
  updateHint();
}

function updateHint() {
  const model = state.config.models.find((m) => m.id === state.model);
  const media = model ? model.media : ["video"];
  $("#drop-hint").textContent = media.length > 1
    ? "Video, image or audio file"
    : "MP4, MOV, MKV, WebM or AVI";
}

function renderSamples() {
  const samples = state.config.samples;
  if (!samples.length) return;
  $("#samples-sheet").hidden = false;
  $("#samples").innerHTML = samples.map((sample) => `
    <button class="sample" type="button" data-id="${escapeHtml(sample.id)}" aria-pressed="false">
      <span class="sample-head"><b>${escapeHtml(sample.title)}</b><span class="truth ${sample.truth}">${sample.truth === "fake" ? "Manipulated" : "Genuine"}</span></span>
      <span>${escapeHtml(sample.kind.replace(/\.$/, ""))}. ${escapeHtml(sample.dataset)}.</span>
    </button>`).join("");
  $$(".sample").forEach((button) => button.addEventListener("click", () => chooseSample(button.dataset.id)));
}

/* Input */
const drop = $("#drop");
const fileInput = $("#file");
["dragenter", "dragover"].forEach((type) => drop.addEventListener(type, (event) => {
  event.preventDefault(); drop.classList.add("over");
}));
["dragleave", "drop"].forEach((type) => drop.addEventListener(type, (event) => {
  event.preventDefault(); drop.classList.remove("over");
}));
drop.addEventListener("drop", (event) => {
  const file = event.dataTransfer.files[0];
  if (file) chooseFile(file);
});
fileInput.addEventListener("change", () => { if (fileInput.files[0]) chooseFile(fileInput.files[0]); });
$("#clear").addEventListener("click", clearInput);

let previewUrl = null;
function setPreview(src, name) {
  if (previewUrl) URL.revokeObjectURL(previewUrl);
  previewUrl = src.startsWith("blob:") ? src : null;
  const video = $("#preview-video");
  video.src = src;
  $("#preview-name").textContent = name;
  $("#preview").classList.add("on");
  $("#go").disabled = false;
}

function chooseFile(file) {
  state.file = file;
  state.sample = null;
  $$(".sample").forEach((b) => b.setAttribute("aria-pressed", "false"));
  setPreview(URL.createObjectURL(file), `${file.name} (${(file.size / 1048576).toFixed(1)} MB)`);
}

function chooseSample(id) {
  const sample = state.config.samples.find((s) => s.id === id);
  state.sample = id;
  state.file = null;
  fileInput.value = "";
  $$(".sample").forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.id === id)));
  setPreview(`/api/samples/${encodeURIComponent(id)}/video`, `${sample.title}, ${sample.dataset}`);
}

function clearInput() {
  state.file = null;
  state.sample = null;
  fileInput.value = "";
  $$(".sample").forEach((b) => b.setAttribute("aria-pressed", "false"));
  $("#preview-video").removeAttribute("src");
  $("#preview").classList.remove("on");
  $("#go").disabled = true;
}

/* Analysis */
$("#go").addEventListener("click", analyze);

async function analyze() {
  if (!state.file && !state.sample) return;
  const form = new FormData();
  form.append("model", state.model);
  if (state.thresholdTouched) form.append("threshold", String(state.threshold));
  form.append("withhold_unfamiliar", String($("#withhold").checked));
  let url = "/api/analyze";
  if (state.file) form.append("file", state.file);
  else { form.append("sample", state.sample); url = "/api/analyze-sample"; }

  $("#go").disabled = true;
  $("#result").innerHTML = `<div class="busy"><div class="spinner"></div><span>Finding the face and scoring 16 frames. The first check loads the models and takes longer.</span></div>`;
  try {
    const response = await fetch(url, { method: "POST", body: form });
    const body = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(body.detail || `The server answered ${response.status}`);
    state.analysis = body;
    state.analysis.sample = state.sample
      ? state.config.samples.find((s) => s.id === state.sample) : null;
    renderResult();
  } catch (error) {
    $("#result").innerHTML = `<div class="error"><strong>The check did not finish.</strong> ${escapeHtml(error.message)}</div>`;
  } finally {
    $("#go").disabled = false;
  }
}

function verdictFor(analysis) {
  if (analysis.probability == null) return "indeterminate";
  if (analysis.withheld) return "withheld";
  return analysis.probability >= state.threshold ? "fake" : "real";
}

const VERDICT_WORDS = {
  fake: "Likely manipulated",
  real: "No manipulation found",
  withheld: "Verdict held back",
  indeterminate: "Cannot judge",
};

function ratesAt(threshold) {
  const points = state.config.operating_points[state.analysis.model];
  if (!points) return null;
  let index = 0;
  points.thresholds.forEach((t, i) => {
    if (Math.abs(Math.log(t) - Math.log(threshold)) < Math.abs(Math.log(points.thresholds[index]) - Math.log(threshold))) index = i;
  });
  return Object.entries(points.datasets).map(([dataset, d]) => ({
    dataset,
    threshold: points.thresholds[index],
    detection: d.detection[index],
    falseAlarm: d.false_alarm[index],
    fakes: d.fakes,
    reals: d.reals,
  }));
}

function verdictSentence(analysis, verdict) {
  const rates = ratesAt(state.threshold) || [];
  const dfdc = rates.find((r) => r.dataset === "DFDC");
  if (verdict === "indeterminate") {
    if (analysis.blockers.includes("missing_visual")) {
      return "No face was found steadily enough across the frames to score. Use a clip where one person faces the camera.";
    }
    return "This file could not be scored.";
  }
  if (verdict === "withheld") {
    return "This footage is unlike the genuine videos the models were trained on. The score is shown below, but on footage like this the models are often wrong, so no verdict is given.";
  }
  if (verdict === "fake") {
    return `The probability, ${fmtProb(analysis.probability)}, is above your threshold of ${fmtProb(state.threshold)}.`;
  }
  const miss = dfdc && dfdc.detection != null ? ` On DFDC footage this setting still misses ${pct(1 - dfdc.detection)} of fakes, so a low score is weak evidence outside the training data.` : "";
  return `The probability, ${fmtProb(analysis.probability)}, is below your threshold of ${fmtProb(state.threshold)}.${miss}`;
}

const L = 6.9; // logit(0.999); the scale runs 0.001 to 0.999
const logit = (p) => Math.log(p / (1 - p));
const toX = (p) => {
  const clamped = Math.min(Math.max(p, 0.0011), 0.9989);
  return ((logit(clamped) + L) / (2 * L)) * 100;
};
const fromX = (x) => 1 / (1 + Math.exp(-((x / 100) * 2 * L - L)));

const WARN = `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" aria-hidden="true"><path d="M12 3l9.5 17h-19z"/><path d="M12 10v4M12 17v.5"/></svg>`;
const INFO = `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" aria-hidden="true"><circle cx="12" cy="12" r="9"/><path d="M12 11v6M12 7.5v.5"/></svg>`;

function renderResult() {
  const a = state.analysis;
  if (!state.thresholdTouched || state.config.settings.lock_threshold) state.threshold = a.threshold;
  const verdict = verdictFor(a);
  const modelLabel = state.config.models.find((m) => m.id === a.model)?.label || a.model;
  const notices = [];
  if (a.familiarity && !a.familiarity.familiar) {
    notices.push(`<div class="notice">${WARN}<span><strong>Unfamiliar footage.</strong> Distance ${a.familiarity.distance.toFixed(1)} from the genuine training footage, past the ${a.familiarity.threshold.toFixed(1)} that familiar clips stay within. On footage like this the face models have called up to 91% of genuine videos fake (Celeb-DF).${a.withheld ? "" : " You chose to see the verdict anyway."}</span></div>`);
  }
  if (a.sample && a.sample.id === "favc-voiceclone" && a.model !== "multimodal") {
    notices.push(`<div class="notice info">${INFO}<span>Only the voice is fake in this clip. The face models judge the video, which is genuine, so they are expected to call it real. The audio-visual fusion is the model that can hear it.</span></div>`);
  }
  if (a.sample) {
    const truth = a.sample.truth === "fake" ? "manipulated" : "genuine";
    notices.push(`<div class="notice info">${INFO}<span>Known answer for this test clip: <strong>${truth}</strong>.</span></div>`);
  }

  const members = Object.entries(a.members || {});
  const membersHtml = members.length > 1 ? `
    <details class="more" open>
      <summary>What each model said</summary>
      <div class="members">${members.map(([name, p]) => `
        <div class="member">
          <span>${escapeHtml(MODEL_NAMES[name] || name)}</span>
          <span class="member-track"><span class="member-fill ${p >= state.threshold ? "over" : ""}" style="width:${toX(p)}%"></span></span>
          <span class="member-value">${fmtProb(p)}</span>
        </div>`).join("")}
      </div>
      <p class="quiet" style="margin-top:8px;font-size:.88rem">Bars use the same log-odds scale as above. ${a.model === "ensemble" ? "The ensemble averages the six logits." : ""}</p>
    </details>` : "";

  const facesHtml = a.faces && a.faces.length ? `
    <details class="more" open>
      <summary>The face crops the models saw</summary>
      <div class="faces">${a.faces.map((src, i) => `<img src="${src}" alt="Face crop ${i + 1}">`).join("")}</div>
    </details>` : "";

  const probability = a.probability;
  const scale = probability == null ? "" : `
    <div class="scale">
      <div class="label-row"><span class="label">Probability of manipulation</span><small>Log-odds scale, so the low end is readable</small></div>
      <div class="scale-track" id="scale-track">
        <div class="scale-bar"></div>
        <div class="scale-zone" id="scale-zone"></div>
        ${members.length > 1 ? members.map(([, p]) => `<div class="scale-member" style="left:${toX(p)}%"></div>`).join("") : ""}
        <div class="scale-clip" style="left:${toX(probability)}%"></div>
        <div class="scale-clip-label" style="left:${Math.min(Math.max(toX(probability), 6), 94)}%">${fmtProb(probability)}</div>
        <div class="scale-handle" id="handle" role="slider" tabindex="0" aria-label="Decision threshold"
          aria-valuemin="0.001" aria-valuemax="0.999"></div>
      </div>
      <div class="scale-axis">${[0.001, 0.01, 0.1, 0.5, 0.9, 0.99, 0.999].map((t) => `<span style="left:${toX(t)}%">${t}</span>`).join("")}</div>
      <div class="scale-ends"><span>Genuine</span><span>Manipulated</span></div>
      <div class="threshold-row">
        <span>Threshold <span class="threshold-value" id="threshold-value"></span></span>
        <div class="presets" id="presets">
          <button class="preset" type="button" data-t="0.5">Balanced 0.5</button>
          <button class="preset" type="button" data-t="0.1">Sensitive 0.1</button>
          <button class="preset" type="button" data-t="0.02">Very sensitive 0.02</button>
        </div>
      </div>
      <div id="rates"></div>
    </div>`;

  $("#result").innerHTML = `
    <div class="verdict">
      <span class="verdict-word ${verdict}" id="verdict-word">${VERDICT_WORDS[verdict]}</span>
    </div>
    <div class="verdict-meta" style="margin-top:8px"><b>${escapeHtml(modelLabel)}</b>, checked in ${a.seconds.toFixed(1)} s</div>
    <p class="verdict-line" id="verdict-line">${escapeHtml(verdictSentence(a, verdict))}</p>
    ${notices.join("")}
    ${scale}
    ${membersHtml}
    ${facesHtml}`;

  if (probability != null) wireScale();
}

function wireScale() {
  const locked = state.config.settings.lock_threshold;
  const track = $("#scale-track");
  const handle = $("#handle");
  const set = (t) => {
    state.thresholdTouched = true;
    state.threshold = Math.min(Math.max(t, 0.001), 0.999);
    refreshThreshold();
  };
  if (locked) {
    handle.style.cursor = "not-allowed";
    $("#presets").innerHTML = '<small>Set by the administrator</small>';
  } else {
    const move = (event) => {
      const box = track.getBoundingClientRect();
      const x = ((event.clientX - box.left) / box.width) * 100;
      set(fromX(Math.min(Math.max(x, 0), 100)));
    };
    track.addEventListener("pointerdown", (event) => {
      track.setPointerCapture(event.pointerId);
      move(event);
      const up = () => { track.removeEventListener("pointermove", move); track.removeEventListener("pointerup", up); };
      track.addEventListener("pointermove", move);
      track.addEventListener("pointerup", up);
    });
    handle.addEventListener("keydown", (event) => {
      const step = event.shiftKey ? 10 : 2;
      const x = toX(state.threshold);
      if (event.key === "ArrowRight" || event.key === "ArrowUp") { set(fromX(Math.min(x + step, 100))); event.preventDefault(); }
      if (event.key === "ArrowLeft" || event.key === "ArrowDown") { set(fromX(Math.max(x - step, 0))); event.preventDefault(); }
      if (event.key === "Home") { set(0.5); event.preventDefault(); }
    });
    $$(".preset", $("#presets")).forEach((b) => b.addEventListener("click", () => set(Number(b.dataset.t))));
  }
  refreshThreshold();
}

function refreshThreshold() {
  const a = state.analysis;
  const x = toX(state.threshold);
  const handle = $("#handle");
  handle.style.left = `${x}%`;
  handle.setAttribute("aria-valuenow", state.threshold.toFixed(3));
  handle.setAttribute("aria-valuetext", fmtProb(state.threshold));
  $("#scale-zone").style.left = `${x}%`;
  $("#threshold-value").textContent = fmtProb(state.threshold);
  $$(".preset", $("#presets")).forEach((b) => b.setAttribute("aria-pressed", String(Math.abs(Number(b.dataset.t) - state.threshold) < 1e-6)));

  const verdict = verdictFor(a);
  const word = $("#verdict-word");
  word.className = `verdict-word ${verdict}`;
  word.textContent = VERDICT_WORDS[verdict];
  $("#verdict-line").textContent = verdictSentence(a, verdict);
  $$(".member-fill").forEach((bar, i) => {
    const p = Object.values(a.members)[i];
    bar.classList.toggle("over", p >= state.threshold);
  });

  const rates = ratesAt(state.threshold);
  $("#rates").innerHTML = rates ? `
    <table class="rates">
      <caption class="quiet" style="text-align:left;padding:4px 0 6px;font-size:.88rem">What a threshold near ${fmtProb(rates[0].threshold)} did on test footage with known answers</caption>
      <thead><tr><th>Footage</th><th class="num">Fakes caught</th><th class="num">Genuine flagged</th></tr></thead>
      <tbody>${rates.map((r) => `<tr>
        <td>${escapeHtml(r.dataset)}${r.dataset === "FakeAVCeleb test" ? " (like training)" : " (unseen)"}</td>
        <td class="num">${r.detection == null ? "n/a" : pct(r.detection)} <span class="rate-bar" style="width:${(r.detection || 0) * 60}px;background:var(--fake)"></span></td>
        <td class="num">${r.falseAlarm == null ? "n/a" : pct(r.falseAlarm)} <span class="rate-bar" style="width:${(r.falseAlarm || 0) * 60}px;background:var(--held)"></span></td>
      </tr>`).join("")}</tbody>
    </table>` : `<p class="quiet" style="margin-top:12px">No per-threshold record is saved for this model. Its fitted cut-off for ${escapeHtml(a.media_kind || "video")} is the default.</p>`;
}

/* Research tabs */
async function loadResearch() {
  if (state.research) return;
  state.research = {};
  const response = await fetch("/api/research");
  state.research = await response.json();
  renderMath();
  renderCharts();
  renderTable();
  renderPrior();
  renderFailures();
}

function renderMath() {
  $("#math").innerHTML = state.research.math.map((block, b) => `
    <div class="sheet math-block">
      <h3>${escapeHtml(block.title)}</h3>
      <div class="src">${escapeHtml(block.source)}</div>
      ${block.items.map((item, i) => `<div class="eq"><div class="eq-tex" id="eq-${b}-${i}"></div><p>${escapeHtml(item.note)}</p></div>`).join("")}
    </div>`).join("");
  state.research.math.forEach((block, b) => block.items.forEach((item, i) => {
    const node = $(`#eq-${b}-${i}`);
    try { katex.render(item.tex, node, { displayMode: true, throwOnError: false }); }
    catch { node.textContent = item.tex; }
  }));
}

function cssVar(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

function rocChart(host, curves) {
  const W = 420, H = 340, m = { l: 46, r: 12, t: 10, b: 40 };
  const iw = W - m.l - m.r, ih = H - m.t - m.b;
  const x = (v) => m.l + v * iw, y = (v) => m.t + (1 - v) * ih;
  const colors = ["--series-1", "--series-2", "--series-3"].map(cssVar);
  const grid = [0, 0.25, 0.5, 0.75, 1].map((v) => `
    <line x1="${x(v)}" x2="${x(v)}" y1="${m.t}" y2="${m.t + ih}" stroke="${cssVar("--line")}" />
    <line x1="${m.l}" x2="${m.l + iw}" y1="${y(v)}" y2="${y(v)}" stroke="${cssVar("--line")}" />
    <text x="${x(v)}" y="${H - 22}" text-anchor="middle" font-size="12" fill="${cssVar("--ink-3")}">${v}</text>
    <text x="${m.l - 8}" y="${y(v) + 4}" text-anchor="end" font-size="12" fill="${cssVar("--ink-3")}">${v}</text>`).join("");
  const lines = curves.map((c, i) => `<polyline fill="none" stroke="${colors[i]}" stroke-width="2" stroke-linejoin="round"
      points="${c.fpr.map((f, k) => `${x(f).toFixed(1)},${y(c.tpr[k]).toFixed(1)}`).join(" ")}" />`).join("");
  host.innerHTML = `
    <div class="chart-legend">${curves.map((c, i) => `<span><i style="background:${colors[i]}"></i>${escapeHtml(c.label)} <b>${c.auc.toFixed(4)}</b></span>`).join("")}</div>
    <svg viewBox="0 0 ${W} ${H}" role="img" aria-label="ROC curves">
      ${grid}
      <line x1="${x(0)}" y1="${y(0)}" x2="${x(1)}" y2="${y(1)}" stroke="${cssVar("--chance")}" stroke-dasharray="4 4" />
      ${lines}
      <line id="cross" x1="0" x2="0" y1="${m.t}" y2="${m.t + ih}" stroke="${cssVar("--ink-3")}" visibility="hidden" />
      <text x="${m.l + iw / 2}" y="${H - 4}" text-anchor="middle" font-size="12" fill="${cssVar("--ink-2")}">Genuine clips flagged (false-positive rate)</text>
      <text transform="translate(12 ${m.t + ih / 2}) rotate(-90)" text-anchor="middle" font-size="12" fill="${cssVar("--ink-2")}">Fakes caught (true-positive rate)</text>
      <rect x="${m.l}" y="${m.t}" width="${iw}" height="${ih}" fill="transparent" class="hit" />
    </svg>
    <div class="tip" hidden></div>`;
  const svg = $("svg", host), tip = $(".tip", host), cross = $("#cross", host);
  $(".hit", host).addEventListener("pointermove", (event) => {
    const box = svg.getBoundingClientRect();
    const scale = W / box.width;
    const px = (event.clientX - box.left) * scale;
    const f = Math.min(Math.max((px - m.l) / iw, 0), 1);
    const k = Math.round(f * (curves[0].fpr.length - 1));
    cross.setAttribute("x1", x(curves[0].fpr[k])); cross.setAttribute("x2", x(curves[0].fpr[k]));
    cross.setAttribute("visibility", "visible");
    tip.hidden = false;
    tip.innerHTML = `False-positive rate ${pct(curves[0].fpr[k])}<br>${curves.map((c, i) => `<span style="color:${colors[i]}">&#9632;</span> ${escapeHtml(c.label)}: ${pct(c.tpr[k])} caught`).join("<br>")}`;
    const left = (event.clientX - box.left) + 14;
    tip.style.left = `${Math.min(left, box.width - tip.offsetWidth)}px`;
    tip.style.top = `${event.clientY - box.top + 30}px`;
  });
  $(".hit", host).addEventListener("pointerleave", () => { tip.hidden = true; cross.setAttribute("visibility", "hidden"); });
}

function seedChart(host, seeds) {
  const rows = Object.entries(seeds).filter(([, v]) => v.length);
  const W = 460, rowH = 40, m = { l: 190, r: 16, t: 8, b: 34 };
  const H = m.t + rows.length * rowH + m.b, iw = W - m.l - m.r;
  const lo = 0.6, hi = 0.8;
  const x = (v) => m.l + ((Math.min(Math.max(v, lo), hi) - lo) / (hi - lo)) * iw;
  const ticks = [0.6, 0.65, 0.7, 0.75, 0.8];
  host.innerHTML = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="DFDC ROC-AUC per training seed">
    ${ticks.map((t) => `<line x1="${x(t)}" x2="${x(t)}" y1="${m.t}" y2="${H - m.b}" stroke="${cssVar("--line")}" />
      <text x="${x(t)}" y="${H - m.b + 18}" text-anchor="middle" font-size="12" fill="${cssVar("--ink-3")}">${t}</text>`).join("")}
    ${rows.map(([label, values], r) => {
      const cy = m.t + r * rowH + rowH / 2;
      const ensemble = values[0].seed === "all";
      const span = values.length > 1 ? `<line x1="${x(Math.min(...values.map((v) => v.roc_auc)))}" x2="${x(Math.max(...values.map((v) => v.roc_auc)))}" y1="${cy}" y2="${cy}" stroke="${cssVar("--line-strong")}" stroke-width="2" />` : "";
      return `<text x="${m.l - 10}" y="${cy + 4}" text-anchor="end" font-size="13" fill="${cssVar("--ink")}" font-weight="${ensemble ? 600 : 400}">${escapeHtml(label)}</text>
        ${span}
        ${values.map((v) => `<circle cx="${x(v.roc_auc)}" cy="${cy}" r="${ensemble ? 7 : 5.5}" fill="${ensemble ? cssVar("--cobalt") : cssVar("--ink-2")}" stroke="${cssVar("--sheet")}" stroke-width="2"><title>${escapeHtml(label)}, seed ${v.seed}: ${v.roc_auc.toFixed(4)}</title></circle>
          ${ensemble || v.seed === 17 ? `<text x="${x(v.roc_auc)}" y="${cy - 11}" text-anchor="middle" font-size="11" fill="${cssVar("--ink-2")}">${ensemble ? v.roc_auc.toFixed(3) : "seed 17"}</text>` : ""}`).join("")}`;
    }).join("")}
    <text x="${m.l + iw / 2}" y="${H - 2}" text-anchor="middle" font-size="12" fill="${cssVar("--ink-2")}">DFDC ROC-AUC</text>
  </svg>`;
}

function renderCharts() {
  const r = state.research;
  if (!r.roc) return;
  if (r.roc["FakeAVCeleb test"]) rocChart($("#roc-favc"), r.roc["FakeAVCeleb test"]);
  if (r.roc.DFDC) rocChart($("#roc-dfdc"), r.roc.DFDC);
  seedChart($("#seeds"), r.seeds);
  const e = r.ensemble;
  if (e) {
    const p17 = e.paired.versus_efficientnet_seed17, pm = e.paired.versus_efficientnet_seed_mean;
    const sign = (v) => (v >= 0 ? "+" : "") + v.toFixed(4);
    $("#ensemble-note").innerHTML = `
      <h3>Why the app defaults to the ensemble</h3>
      <p style="margin-top:10px">Six models trained on the same data disagree on DFDC by up to 0.13 of ROC-AUC. Averaging their logits gives <b>${e.roc_auc.toFixed(4)}</b> [${e.ci[0].toFixed(3)}, ${e.ci[1].toFixed(3)}] on DFDC, above every member.</p>
      <ul class="plain">
        <li>Against the average single seed: <b>${sign(pm.estimate)}</b> [${sign(pm.lower)}, ${sign(pm.upper)}], clear of zero.</li>
        <li>Against the luckiest seed: <b>${sign(p17.estimate)}</b> [${sign(p17.lower)}, ${sign(p17.upper)}], not clear of zero.</li>
      </ul>
      <p style="margin-top:10px">The gain does not carry to every dataset.</p>
      <ul class="plain">
        ${e.celebdf ? `<li>Celeb-DF: <b>${e.celebdf.roc_auc.toFixed(4)}</b>, level with the seed-17 model (${sign(e.celebdf.paired.versus_efficientnet_seed17.estimate)}, interval spans zero).</li>` : ""}
        ${e.mnw ? `<li>MNW, at threshold 0.5: catches <b>${e.mnw.ensemble} of ${e.mnw.fakes}</b> fakes, against ${e.mnw.best_single} for the seed-17 model. The ConvNeXt members score these fakes low.</li>` : ""}
        <li>One member, ConvNeXt seed 43, diverged in training and is effectively never fine-tuned (F29). Leaving it out lowers DFDC to 0.7666 and Celeb-DF to 0.6553, so it stays.</li>
      </ul>
      <p class="quiet" style="margin-top:10px">No weight was fitted. "All six" was chosen after comparing three groupings on DFDC, so treat the gain over the luckiest seed as unproven. Its probabilities are badly calibrated off-domain (ECE 0.87), which is why the check page shows measured rates rather than trusting the number.</p>`;
  }
}

let datasetFilter = "All";
function renderTable() {
  const rows = state.research.table;
  const datasets = ["All", ...new Set(rows.map((r) => r.dataset))];
  $("#dataset-filters").innerHTML = datasets.map((d) => `<button class="preset" type="button" data-d="${escapeHtml(d)}" aria-pressed="${d === datasetFilter}">${escapeHtml(d)}</button>`).join("");
  $$("#dataset-filters .preset").forEach((b) => b.addEventListener("click", () => { datasetFilter = b.dataset.d; renderTable(); }));
  const visible = rows.filter((r) => datasetFilter === "All" || r.dataset === datasetFilter);
  const forest = (r) => {
    const lo = 0.4, hi = 1.0, w = 180, x = (v) => ((Math.min(Math.max(v, lo), hi) - lo) / (hi - lo)) * (w - 8) + 4;
    const ci = r.ci && r.ci[0] != null ? `<line x1="${x(r.ci[0])}" x2="${x(r.ci[1])}" y1="11" y2="11" stroke="${cssVar("--ink-3")}" stroke-width="2" />` : "";
    return `<svg class="forest" viewBox="0 0 ${w} 22" aria-hidden="true">
      <line x1="${x(0.5)}" x2="${x(0.5)}" y1="2" y2="20" stroke="${cssVar("--chance")}" stroke-dasharray="2 2" />
      ${ci}<circle cx="${x(r.roc_auc)}" cy="11" r="4.5" fill="${r.group === "Served in this app" ? cssVar("--cobalt") : cssVar("--ink")}" /></svg>`;
  };
  let group = null;
  const body = visible.map((r) => {
    const head = r.group !== group ? `<tr class="group"><td colspan="8">${escapeHtml(r.group)}</td></tr>` : "";
    group = r.group;
    return `${head}<tr class="${r.group === "Served in this app" ? "served" : ""}" title="Source: ${escapeHtml(r.source)}">
      <td>${escapeHtml(r.model)}</td><td>${escapeHtml(r.dataset)}</td>
      <td class="num"><b>${fmt(r.roc_auc)}</b></td>
      <td class="num quiet">${r.ci && r.ci[0] != null ? `${fmt(r.ci[0], 3)} to ${fmt(r.ci[1], 3)}` : ""}</td>
      <td>${forest(r)}</td>
      <td class="num">${fmt(r.eer, 3)}</td><td class="num">${fmt(r.balanced_accuracy, 3)}</td><td class="num">${fmt(r.ece, 3)}</td>
    </tr>`;
  }).join("");
  $("#metric-table").innerHTML = `<thead><tr><th>Model</th><th>Tested on</th><th class="num">ROC-AUC</th><th class="num">95% CI</th><th>0.4 to 1.0</th><th class="num">EER</th><th class="num">Bal. acc. at 0.5</th><th class="num">ECE</th></tr></thead><tbody>${body}</tbody>`;
}

function renderPrior() {
  const r = state.research;
  const statusClass = (s) => (s === "met" ? "status-met" : s === "not met" ? "status-miss" : "status-partial");
  $("#targets").innerHTML = `<thead><tr><th>Target from the literature</th><th>This project</th><th>Status</th></tr></thead><tbody>${r.targets.map((t) => `
    <tr><td>${escapeHtml(t.target)}</td><td>${escapeHtml(t.ours)}</td><td class="status ${statusClass(t.status)}">${escapeHtml(t.status)}</td></tr>`).join("")}</tbody>`;
  $("#prior").innerHTML = r.prior_work.map((p) => `
    <article class="card"><h3>${escapeHtml(p.title)}</h3><p>${escapeHtml(p.comparison)}</p><p class="diff">${escapeHtml(p.difference)}</p><p class="src">${escapeHtml(p.source)}</p></article>`).join("");
  $("#contributions").innerHTML = r.contributions.map((c) => `<li>${escapeHtml(c)}</li>`).join("");
}

function renderFailures() {
  const r = state.research;
  $("#limits").innerHTML = r.limits.map((l) => `<li>${escapeHtml(l)}</li>`).join("");
  $("#failures").innerHTML = r.failures.map((g) => `
    <div class="sheet failure-group"><h2>${escapeHtml(g.group)}</h2>
      ${g.items.map((f) => `<div class="failure"><b>${escapeHtml(f.title)}</b><span class="src">${escapeHtml(f.source)}</span><p>${escapeHtml(f.detail)}</p></div>`).join("")}
    </div>`).join("");
}

loadConfig().then(() => {
  const start = location.hash.slice(1);
  if (start && $(`.tab[data-tab="${start}"]`)) showTab(start);
}).catch((error) => {
  $("#result").innerHTML = `<div class="error"><strong>The app could not reach its server.</strong> ${escapeHtml(error.message)}</div>`;
});
