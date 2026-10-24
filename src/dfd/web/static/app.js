"use strict";
// All text from the server is inserted with textContent, never innerHTML (file names are user controlled).
const STAGES = ["validate", "decode", "detect", "score", "calibrate", "fuse"];
const $ = (id) => document.getElementById(id);
const fileInput = $("file"), goBtn = $("go");
let chosen = null;

function setFile(f) {
  chosen = f || null;
  goBtn.disabled = !chosen;
  $("file-info").textContent = chosen ? chosen.name + " (" + (chosen.size / 1048576).toFixed(1) + " MB)" : "";
  $("drop-text").textContent = chosen ? "Selected: " + chosen.name : "Choose a file or drop it here";
}
fileInput.addEventListener("change", () => setFile(fileInput.files[0]));
const drop = $("drop");
["dragenter", "dragover"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add("over"); }));
["dragleave", "drop"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove("over"); }));
drop.addEventListener("drop", (e) => { if (e.dataTransfer.files[0]) { setFile(e.dataTransfer.files[0]); } });

function renderStages(done, active) {
  const ol = $("stage-list");
  ol.replaceChildren();
  STAGES.forEach((s) => {
    const li = document.createElement("li");
    li.textContent = s;
    if (done.includes(s)) li.className = "done";
    else if (s === active) li.className = "active";
    ol.appendChild(li);
  });
}

function row(tbody, a, b) {
  const tr = document.createElement("tr");
  const th = document.createElement("th"), td = document.createElement("td");
  th.textContent = a; td.textContent = b;
  tr.append(th, td); tbody.appendChild(tr);
}

function showResult(resp) {
  $("result").hidden = false;
  // D-029: a visible MOCK banner on every result whose inference_source is not "real".
  $("result-mock-banner").hidden = resp.inference_source === "real";
  const v = $("verdict"), failed = resp.status.state === "failed";
  const fused = resp.result && resp.result.fused;
  const decision = failed ? "undetermined" : fused.decision;
  v.className = "card verdict " + decision;
  $("verdict-label").textContent = failed ? "No result" : ({ fake: "Likely fake", real: "Likely real", undetermined: "Undetermined" })[decision];
  $("verdict-score").textContent = (!failed && fused.score !== null) ? "Probability of fake: " + (fused.score * 100).toFixed(1) + " %" : "";
  $("verdict-note").textContent = failed ? "The request failed; see the errors below." : (fused.reason || "");

  const errs = $("errors"); errs.replaceChildren();
  resp.errors.forEach((e) => { const li = document.createElement("li"); li.className = "err"; li.textContent = e.code + ": " + e.message; errs.appendChild(li); });
  $("errors-card").hidden = resp.errors.length === 0;
  const warns = $("warnings"); warns.replaceChildren();
  resp.warnings.forEach((w) => { const li = document.createElement("li"); li.className = "warn"; li.textContent = w.code + ": " + w.message; warns.appendChild(li); });
  $("warnings-card").hidden = resp.warnings.length === 0;

  const body = $("modalities").querySelector("tbody"); body.replaceChildren();
  const per = resp.result ? resp.result.per_modality : {};
  Object.entries(per).forEach(([name, m]) => {
    const tr = document.createElement("tr");
    [name, m.available ? (m.score * 100).toFixed(1) + " %" : "not available", m.n_units === null ? "" : String(m.n_units), m.reason || ""]
      .forEach((t) => { const td = document.createElement("td"); td.textContent = t; tr.appendChild(td); });
    body.appendChild(tr);
  });
  $("modalities-card").hidden = Object.keys(per).length === 0;
  $("meaning").textContent = resp.score_meaning;

  const d = $("details").querySelector("tbody"); d.replaceChildren();
  row(d, "inference_source", resp.inference_source);
  row(d, "file", resp.input.filename); row(d, "type", String(resp.input.media_type));
  row(d, "size (bytes)", String(resp.input.size_bytes)); row(d, "sha256", resp.input.sha256);
  row(d, "duration (s)", resp.input.duration_s === null ? "" : String(resp.input.duration_s));
  row(d, "model", resp.model ? JSON.stringify(resp.model) : "none (mock or failed)");
  row(d, "state", resp.status.state); row(d, "time (ms)", String(resp.status.processing_time_ms));
  const st = $("stage-times").querySelector("tbody"); st.replaceChildren();
  resp.status.stages.forEach((s) => row(st, s.name, s.ms + " ms"));
  $("raw").textContent = JSON.stringify(resp, null, 2);
  $("result").scrollIntoView({ behavior: "smooth" });
}

function fail(message) {
  showResult({ inference_source: document.body.dataset.source, status: { state: "failed", processing_time_ms: 0, stages: [] },
    input: { filename: chosen ? chosen.name : "", media_type: null, size_bytes: 0, sha256: "", duration_s: null },
    model: null, result: null, score_meaning: "", warnings: [], errors: [{ code: "UNREADABLE_FILE", message: message }] });
}

function poll(jobId) {
  fetch("/api/jobs/" + jobId).then((r) => r.json()).then((j) => {
    renderStages(j.stages_done, j.stage);
    if (j.state === "done") { goBtn.disabled = false; renderStages(STAGES, null); showResult(j.response); }
    else setTimeout(() => poll(jobId), 300);
  }).catch(() => { goBtn.disabled = false; fail("lost contact with the local server"); });
}

goBtn.addEventListener("click", () => {
  if (!chosen) return;
  goBtn.disabled = true; $("result").hidden = true; $("progress").hidden = false;
  $("upload-fill").style.width = "0%"; renderStages([], null);
  const fd = new FormData(); fd.append("file", chosen);
  const xhr = new XMLHttpRequest();
  xhr.open("POST", "/api/jobs");
  xhr.upload.onprogress = (e) => { if (e.lengthComputable) $("upload-fill").style.width = (100 * e.loaded / e.total).toFixed(0) + "%"; };
  xhr.onload = () => {
    let body = null; try { body = JSON.parse(xhr.responseText); } catch (e) { /* not JSON */ }
    if (xhr.status === 202 && body && body.job_id) { $("upload-fill").style.width = "100%"; poll(body.job_id); }
    else if (body && body.schema_version) { goBtn.disabled = false; showResult(body); }
    else { goBtn.disabled = false; fail("the server answered with status " + xhr.status); }
  };
  xhr.onerror = () => { goBtn.disabled = false; fail("could not reach the local server"); };
  xhr.send(fd);
});
