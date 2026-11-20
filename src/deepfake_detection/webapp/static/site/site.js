/* Deepfake Check: the mobile menu, the detector card, and the lip-sync illustration. */
(function () {
  "use strict";

  /* mobile menu */
  const menu = document.querySelector(".menu");
  const links = document.getElementById("nav-links");
  if (menu && links) {
    menu.addEventListener("click", () => {
      const open = links.classList.toggle("open");
      menu.setAttribute("aria-expanded", String(open));
    });
  }

  /* detector ------------------------------------------------------------------------------- */
  const OUTCOMES = {
    signs: { cls: "signs", text: "Signs of manipulation" },
    no_signs: { cls: "none", text: "No signs found" },
    cant_tell: { cls: "unsure", text: "Can't tell" },
    cant_check: { cls: "cant", text: "Can't check" },
  };
  const CHECKS = [["face", "Face"], ["voice", "Voice"], ["sync", "Lip sync"]];
  const MAX_MB = 300;
  // The static frontend names the PC's API in a meta tag; served by the PC itself, the API is this origin.
  const API = (document.querySelector('meta[name="ddf-api"]') || {}).content || "";

  const el = (tag, cls, text) => {
    const node = document.createElement(tag);
    if (cls) node.className = cls;
    if (text !== undefined) node.textContent = text;
    return node;
  };

  const fmt = (x) => x.toFixed(2);
  const clock = (s) => Math.floor(s / 60) + ":" + String(Math.floor(s % 60)).padStart(2, "0");

  /* A small "i" button whose text shows on hover or tap, so details stay out of the way until asked for. */
  function tip(text, label) {
    const button = el("button", "tip");
    button.type = "button";
    button.setAttribute("aria-label", label || "More information");
    button.append(el("span", "tipmark", "i"), el("span", "bubble", text));
    return button;
  }

  function aboutBody(about, found) {
    const body = el("div", "cbody small");
    if (!about) return body;
    body.append(el("p", "", about.what));
    body.append(el("p", "", (found ? "If found: " : "High score: ") + about.high));
    body.append(el("p", "", "Misses: " + about.misses));
    if (about.caught && about.caught.length) {
      body.append(el("p", "muted", "Fakes it caught on our test sets: "
        + about.caught.map(([set, rate]) => set + " " + rate).join(", ") + "."));
    }
    return body;
  }

  const GENERATORS = { "kling2.5": "Kling 2.5", pixverseV5: "PixVerse V5", "seedance1.0": "Seedance 1.0",
    sora2: "Sora 2", "veo3.1": "Veo 3.1", "wan2.5": "Wan 2.5", dreamina: "Dreamina (Seedance)" };

  function generatorRow(g, about) {
    g = g || { suspected: false };
    const wrap = el("div", "genwrap");
    const row = el("details", "crow generator" + (g.suspected ? " raised" : ""));
    const line = el("summary", "cline");
    const named = g.name ? (GENERATORS[g.name] || g.name) : null;
    const text = !g.suspected ? "None suspected" : g.source === "label" ? (named || "AI-made (label)") : "Looks AI-made";
    line.append(el("span", "cname", "Generator audit"), el("span", "cfile", text),
      el("span", "cval mono", g.source === "label" && g.suspected ? "from label" : ""));
    row.append(line, aboutBody(about, true));
    wrap.append(row);
    // One box per generator it learned, highest first; shown only when the video looks AI-made.
    const shares = g.suspected && g.source === "model" ? Object.entries(g.shares || {}) : [];
    const shown = shares.filter(([, v]) => Math.round(v * 100) > 0).sort((a, b) => b[1] - a[1]);
    if (shown.length) {
      const chips = el("div", "genchips");
      for (const [name, v] of shown) {
        const chip = el("span", "genchip" + (name === g.name && g.sure ? " top" : ""));
        chip.append(el("b", "", GENERATORS[name] || name), el("span", "mono", " " + Math.round(v * 100) + "%"));
        chips.append(chip);
      }
      wrap.append(chips);
    }
    return wrap;
  }

  /* The main meter: every check's evidence added up into one score on the page's 0..1 scale. */
  function overallMeter(score, lines) {
    const box = el("div", "overall" + (score >= lines.flag ? " raised" : score >= lines.lower ? " unsure" : ""));
    const head = el("div", "overall-head");
    head.append(el("span", "overall-name", "Overall"), tip("All the checks' evidence added up: several mildly unusual "
      + "checks count for more than one. Below " + fmt(lines.lower) + " looks real, " + fmt(lines.lower) + " to "
      + fmt(lines.flag) + " can't tell, from " + fmt(lines.flag) + " (the mark) flagged; about 5 in 100 real "
      + "videos reach the mark.", "How the overall score works"), el("span", "overall-val mono", fmt(score)));
    const track = el("div", "overall-track");
    const fill = el("i");
    fill.style.width = Math.min(Math.max(score, 0), 1) * 100 + "%";
    const tick = el("b", "flagtick");
    tick.style.left = lines.flag * 100 + "%";
    track.append(fill, tick);
    box.append(head, track);
    return box;
  }

  function checkRows(data) {
    const checks = data.checks || {}, lines = data.lines, about = data.about || {};
    const box = el("div", "checks");
    const flag = lines ? lines.flag : null;
    const head = el("p", "checkhead");
    head.append(el("span", "", "Checks"));
    if (flag != null) {
      head.append(tip("Scores run from 0 (looks real) through 0.5 (neutral) to 1 (looks fake). The mark on each bar is the "
        + "flag line, " + fmt(flag) + ": about 5 in 100 real videos reach it. Tap a row to see what that check does.",
        "How to read the scores"));
    }
    box.append(head);
    for (const [key, label] of CHECKS) {
      const c = checks[key];
      const score = c && c.score != null ? c.score : null;
      const row = el("details", "crow " + key + (c && c.raised ? " raised" : ""));
      const line = el("summary", "cline");
      line.append(el("span", "cname", label));
      const track = el("span", "ctrack");
      const fill = el("i");
      fill.style.width = score != null ? score * 100 + "%" : "0%";
      track.append(fill);
      if (flag != null) {
        const tick = el("b", "flagtick");
        tick.style.left = flag * 100 + "%";
        track.append(tick);
      }
      line.append(track, el("span", "cval mono", score != null ? fmt(score) : "not checked"));
      row.append(line);
      const body = aboutBody(about[key]);
      if (c && c.reason) body.prepend(el("p", "", "Not checked: " + c.reason));
      if (key === "sync") {
        const trace = traceView(data.trace);
        if (trace) body.append(trace);
      }
      row.append(body);
      box.append(row);
    }
    const labels = data.file_check || [];
    const file = el("details", "crow file" + (labels.length ? " raised" : ""));
    const line = el("summary", "cline");
    line.append(el("span", "cname", "File audit"),
      el("span", "cfile", labels.length ? "Suspicious" : "Clean"),
      el("span", "cval mono", ""));
    file.append(line);
    const body = aboutBody(about.file, true);
    for (const text of labels) body.prepend(el("p", "strong", "The file carries " + text + "."));
    if (!labels.length) body.prepend(el("p", "", "Clean means no AI label was found. Most uploads lose their labels when re-saved, so a clean file can still be fake."));
    file.append(body);
    box.append(file);
    box.append(generatorRow(data.generator, about.generator));
    return box;
  }

  /* Preview of the visitor's own file, with a timeline coloured part by part: hold or drag on it to see the
     scores of the 3-second part under the cursor. The file never leaves the browser again. */
  function partAt(timeline, time) {
    let found = null;
    for (const part of timeline) if (part.start <= time && time < part.end) found = part;
    if (found) return found;
    return timeline.reduce((best, p) =>
      Math.abs((p.start + p.end) / 2 - time) < Math.abs((best.start + best.end) / 2 - time) ? p : best);
  }

  function level(reading, lines) {
    if (reading == null) return "none";
    if (reading >= lines.flag) return "flag";
    return reading >= lines.lower ? "unsure" : "ok";
  }

  /* One check's score as a ring: the arc fills to the score, coloured by which side of the lines it falls. */
  function ring(label, value, lines) {
    const ns = "http://www.w3.org/2000/svg";
    const box = el("div", "ring " + level(value, lines));
    const svg = document.createElementNS(ns, "svg");
    svg.setAttribute("viewBox", "0 0 44 44");
    svg.setAttribute("aria-hidden", "true");
    const track = document.createElementNS(ns, "circle");
    const arc = document.createElementNS(ns, "circle");
    for (const c of [track, arc]) {
      c.setAttribute("cx", "22"); c.setAttribute("cy", "22"); c.setAttribute("r", "18");
      svg.append(c);
    }
    track.setAttribute("class", "track");
    arc.setAttribute("class", "arc");
    const length = 2 * Math.PI * 18;
    arc.setAttribute("stroke-dasharray", (Math.min(Math.max(value, 0), 1) * length) + " " + length);
    const text = el("span", "ring-val mono", fmt(value));
    const figure = el("span", "ring-fig");
    figure.append(svg, text);
    box.append(figure, el("span", "ring-label small", label));
    return box;
  }

  function preview(file, data) {
    const timeline = data.timeline || [];
    const lines = data.lines;
    const box = el("figure", "preview");
    const kind = (file.type || "").split("/")[0];
    const url = URL.createObjectURL(file);
    let media = null;
    if (kind === "image") {
      const img = el("img");
      img.src = url;
      img.alt = "The picture that was checked";
      box.append(img);
    } else {
      media = el(kind === "audio" || data.where && data.where.startsWith("A voice") ? "audio" : "video");
      media.src = url;
      media.controls = true;
      media.playsInline = true;
      media.preload = "auto"; // the first frame shows at once, not a black box
      box.append(media);
    }
    if (!timeline.length || !lines || kind === "image") return box;
    const end = Math.max(...timeline.map((p) => p.end));
    const bar = el("div", "timeline");
    bar.setAttribute("role", "slider");
    bar.setAttribute("aria-label", "Scores along the video; hold or drag to see a part's scores");
    bar.tabIndex = 0;
    for (const part of timeline) {
      const seg = el("span", "seg " + level(part.reading, lines));
      seg.style.left = part.start / end * 100 + "%";
      seg.style.width = (part.end - part.start) / end * 100 + "%";
      bar.append(seg);
    }
    const cursor = el("i", "cursor");
    bar.append(cursor);
    // The part under the cursor: its time, then one round meter per check that ran on it.
    const readout = el("div", "readout");
    const when = el("p", "when mono");
    const rings = el("div", "rings");
    readout.append(when, rings);
    const show = (time) => {
      time = Math.min(Math.max(time, 0), end);
      cursor.style.left = time / end * 100 + "%";
      const part = partAt(timeline, time);
      when.replaceChildren(el("b", "", clock(time)), el("span", "", clock(part.start) + " to " + clock(part.end)));
      // Named from CHECKS, the list the rows use, so every check the server sends has a name.
      const scored = CHECKS.filter(([k]) => part.checks[k] != null);
      rings.replaceChildren(...(scored.length
        ? scored.map(([k, label]) => ring(label, part.checks[k], lines))
        : [el("span", "small", "No check could run on this part.")]));
      bar.setAttribute("aria-valuetext", clock(time) + ": " + (scored.map(([k, label]) => label + " "
        + fmt(part.checks[k])).join(", ") || "no check ran"));
    };
    const fromPointer = (event) => {
      const rect = bar.getBoundingClientRect();
      const time = (event.clientX - rect.left) / rect.width * end;
      show(time);
      if (media && Number.isFinite(media.duration)) media.currentTime = Math.min(time, media.duration);
    };
    // While held, the readout follows the finger; the video's own time updates lag behind a seek and would
    // snap it back to where the video was.
    let held = false;
    bar.addEventListener("pointerdown", (event) => {
      held = true;
      fromPointer(event);
      try { bar.setPointerCapture(event.pointerId); } catch (error) { /* keep scrubbing without capture */ }
    });
    bar.addEventListener("pointermove", (event) => { if (held) fromPointer(event); });
    const release = () => { held = false; };
    bar.addEventListener("pointerup", release);
    bar.addEventListener("pointercancel", release);
    bar.addEventListener("keydown", (event) => {
      const now = media ? media.currentTime : 0;
      const step = event.key === "ArrowRight" ? 1 : event.key === "ArrowLeft" ? -1 : 0;
      if (!step) return;
      event.preventDefault();
      if (media) media.currentTime = Math.min(Math.max(now + step, 0), end);
      show(now + step);
    });
    if (media) media.addEventListener("timeupdate", () => { if (!held) show(media.currentTime); });
    const readrow = el("div", "readrow");
    readrow.append(readout, tip("The bar is the video in 3-second parts. Green: real side, below " + fmt(lines.lower)
      + ". Yellow: can't tell. Red: fake side, from " + fmt(lines.flag) + ". Grey: not scored. Hold or drag on it: each ring shows a check's score for that part, against the flag line " + fmt(lines.flag) + ".",
      "How to read the timeline"));
    box.append(bar, readrow);
    show(0);
    return box;
  }

  function render(card, data, file) {
    card.replaceChildren();
    card.className = "result";
    if (data.status === "rejected") {
      card.classList.add("cant");
      card.append(el("p", "outcome cant", "Can't check"), el("p", "", data.reason));
    } else if (data.status === "busy" || data.status === "error" || data.status === "unknown") {
      card.classList.add("cant");
      card.append(el("p", "outcome cant", data.status === "busy" ? "Busy" : "Not checked"), el("p", "", data.reason));
    } else {
      if (file && data.outcome !== "cant_check") card.append(preview(file, data));
      const o = OUTCOMES[data.outcome] || OUTCOMES.cant_check;
      card.classList.add(o.cls);
      const title = el("p", "outcome " + o.cls, o.text);
      if (data.how_sure) title.append(tip(data.how_sure, "How sure is this answer?"));
      card.append(title);
      if (data.description) {
        const what = el("p", "describe");
        what.append(data.description, tip("Written by an AI model on our computer from one still and the heatmap. "
          + "It describes what is visible; the verdict comes from the checks below.", "Who wrote this"));
        card.append(what);
      }
      if (data.summary) card.append(el("p", "", data.summary));
      if (data.overall != null && data.lines) card.append(overallMeter(data.overall, data.lines));
      const heat = explainView(data.explain, card, !!(data.checks && data.checks.face && data.checks.face.raised));
      if (heat) card.append(heat);
      card.append(checkRows(data));
    }
    card.hidden = false;
  }

  /* Grad-CAM (webapp/explain.py): one highlighted sentence on where the face model's evidence sits, and
     the three face crops it leaned on most behind a button. A click on a crop moves the preview there. */
  function explainView(explain, card, faceRaised) {
    if (!explain || !explain.frames || !explain.frames.length) return null;
    const box = el("figure", "heatbox");
    const focus = explain.focus;
    const lead = el("p", "heatlead");
    if (focus) {
      const place = focus.region + (focus.side === "centre" ? "" : ", " + focus.side);
      lead.append("Our face model looked hardest at the ", el("strong", "", place), ", at " + clock(focus.time) + ".");
    } else {
      lead.textContent = explain.summary || "";
    }
    const toggle = el("button", "heattoggle", "Show heatmaps");
    toggle.type = "button";
    toggle.setAttribute("aria-expanded", "false");
    const panel = el("div", "heatpanel");
    panel.hidden = true;
    toggle.addEventListener("click", () => {
      panel.hidden = !panel.hidden;
      toggle.textContent = panel.hidden ? "Show heatmaps" : "Hide heatmaps";
      toggle.setAttribute("aria-expanded", String(!panel.hidden));
    });
    const row = el("div", "heatrow");
    for (const frame of explain.frames) {
      const button = el("button", "heat");
      button.type = "button";
      const img = el("img");
      img.src = frame.image;
      img.alt = "Face at " + clock(frame.time) + ", heat on the " + frame.region;
      button.append(img, el("span", "small mono", clock(frame.time) + " " + frame.region
        + (frame.side && frame.side !== "centre" ? ", " + frame.side.replace(" of the picture", "") : "")));
      button.addEventListener("click", () => {
        const media = card.querySelector(".preview video");
        if (media && Number.isFinite(media.duration)) {
          media.currentTime = Math.min(frame.time, media.duration);
          media.scrollIntoView({ block: "nearest", behavior: "smooth" });
        }
      });
      row.append(button);
    }
    panel.append(lead, row, el("p", "small heatnote", "Warmer colours pushed the face score toward fake. Tap a face to jump the video there."));
    const tools = el("div", "heattools");
    tools.append(toggle, tip("A heatmap shows where the face model looked. It is not proof that this area was edited.",
      "What a heatmap shows"));
    box.append(tools, panel);
    return box;
  }

  /* The trace: speech loudness over the first 2 s, and the lip-sync misfit of each 2 s window. */
  function traceView(trace) {
    if (!trace || !trace.loudness_db || !trace.loudness_db.length) return null;
    const ns = "http://www.w3.org/2000/svg", W = 300, H = 60;
    const values = trace.loudness_db, lo = Math.min(...values), hi = Math.max(...values, lo + 1);
    const svg = document.createElementNS(ns, "svg");
    svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
    svg.setAttribute("class", "trace");
    svg.setAttribute("preserveAspectRatio", "none");
    svg.setAttribute("role", "img");
    svg.setAttribute("aria-label", "Speech loudness over the first 2 seconds of the cleaned clip");
    const line = document.createElementNS(ns, "polyline");
    line.setAttribute("points", values.map((v, i) =>
      `${(i / (values.length - 1) * W).toFixed(1)},${(H - 4 - (v - lo) / (hi - lo) * (H - 8)).toFixed(1)}`).join(" "));
    line.setAttribute("fill", "none");
    line.setAttribute("stroke", "var(--sound)");
    line.setAttribute("stroke-width", "2");
    line.setAttribute("vector-effect", "non-scaling-stroke");
    svg.append(line);
    const box = el("figure", "tracebox");
    box.append(svg);
    const windows = (trace.windows || []).map((w) =>
      `${w.start.toFixed(0)} to ${(w.start + 2).toFixed(0)} s: misfit ${w.misfit.toFixed(2)}`);
    box.append(el("figcaption", "small", "Speech loudness, first 2 s. Lip-sync misfit per window (higher is worse): "
      + (windows.length ? windows.join("; ") : "not measured") + "."));
    return box;
  }

  const STAGES = { received: "Received. Waiting for the detector", "file check": "Reading the file for AI labels", "face check": "Checking the face", describing: "Describing what it shows",
    cleaning: "Cleaning the clip",
    "finding the face": "Finding the face", scoring: "Scoring", done: "Done" };
  const pause = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

  async function follow(status, job) {
    for (let tries = 0; tries < 400; tries++) {
      await pause(1500);
      const response = await fetch(API + "/api/check/" + job);
      const data = await response.json();
      if (data.status === "waiting" || data.status === "running") {
        const ahead = data.ahead ? ` (${data.ahead} ahead of you)` : "";
        const [stage, part] = String(data.stage || "").split(":");
        const of = part ? " (part " + part.replace("/", " of ") + ")" : "";
        status.textContent = (STAGES[stage] || "Working") + of + ahead + "...";
        continue;
      }
      return data;
    }
    return { status: "error", reason: "The check took too long. Please try again later." };
  }

  /* The chosen file replaces the upload box: a preview while it is checked, then the answer below it. */
  function showPicked(root, file) {
    const picked = root.querySelector("[data-picked]");
    const media = root.querySelector("[data-picked-media]");
    if (media.dataset.url) URL.revokeObjectURL(media.dataset.url);
    media.replaceChildren();
    const kind = (file.type || "").split("/")[0];
    const url = URL.createObjectURL(file);
    media.dataset.url = url;
    const node = el(kind === "image" ? "img" : kind === "audio" ? "audio" : "video");
    node.src = url;
    if (kind === "image") node.alt = "The picture you chose";
    else { node.controls = true; node.playsInline = true; node.preload = "auto"; }
    media.append(node);
    media.hidden = false;
    root.querySelector("[data-picked-name]").textContent = file.name || "Your recording";
    picked.hidden = false;
    root.querySelectorAll("[data-drop], .live-buttons, [data-live-panel]").forEach((n) => { n.hidden = true; });
  }

  function reset(root) {
    root.dataset.token = String(Number(root.dataset.token || 0) + 1);
    const media = root.querySelector("[data-picked-media]");
    if (media.dataset.url) URL.revokeObjectURL(media.dataset.url);
    delete media.dataset.url;
    media.replaceChildren();
    root.querySelector("[data-picked]").hidden = true;
    root.querySelector("[data-status]").textContent = "";
    const card = root.querySelector("[data-result]");
    card.replaceChildren();
    card.hidden = true;
    root.querySelector("[data-file]").value = "";
    root.querySelectorAll("[data-drop], .live-buttons").forEach((n) => { n.hidden = false; });
  }

  /* A dataset sample shows its preview and is checked on the server's original; `sample` is its catalogue entry. */
  async function submit(root, file, mode, sample) {
    const status = root.querySelector("[data-status]");
    const card = root.querySelector("[data-result]");
    const token = String(Number(root.dataset.token || 0) + 1);
    root.dataset.token = token;
    const current = () => root.dataset.token === token;
    showPicked(root, file);
    card.hidden = true;
    if (!sample && file.size > MAX_MB * 1024 * 1024) {
      status.textContent = "";
      render(card, { status: "rejected", reason: "The file is over " + MAX_MB + " MB." });
      return;
    }
    status.textContent = sample ? "Sending the sample..." : "Uploading " + file.name + "...";
    const body = new FormData();
    body.append("file", file);
    if (mode) body.append("mode", mode);
    try {
      const response = sample
        ? await fetch(API + "/api/check-sample", {
          method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ id: sample.id }) })
        : await fetch(API + "/api/check", { method: "POST", body });
      let data = await response.json().catch(() => ({ status: "rejected", reason: "The server sent an unreadable answer." }));
      if (data.status === "queued") {
        status.textContent = STAGES.received + "...";
        data = await follow(status, data.job);
      }
      if (!current()) return;
      status.textContent = "";
      render(card, data, file);
      if (sample && data.status === "done") card.querySelector(".outcome").after(truth(sample));
      // The answer carries its own preview with the timeline; keep only the name and the buttons above it.
      if (card.querySelector(".preview")) root.querySelector("[data-picked-media]").hidden = true;
    } catch (error) {
      if (!current()) return;
      status.textContent = "";
      render(card, { status: "rejected", reason: API
        ? "The detector is offline right now: it runs on our own computer, which may be switched off. Try again later."
        : "The upload failed. Check your connection and try again." });
    }
  }

  document.querySelectorAll("[data-detector]").forEach((root) => {
    const drop = root.querySelector("[data-drop]");
    const input = root.querySelector("[data-file]");
    input.addEventListener("change", () => input.files[0] && submit(root, input.files[0]));
    root.querySelector("[data-remove]").addEventListener("click", () => reset(root));
    root.querySelector("[data-new]").addEventListener("click", () => { reset(root); input.click(); });
    ["dragenter", "dragover"].forEach((name) => drop.addEventListener(name, (event) => {
      event.preventDefault();
      drop.classList.add("over");
    }));
    ["dragleave", "drop"].forEach((name) => drop.addEventListener(name, () => drop.classList.remove("over")));
    drop.addEventListener("drop", (event) => {
      event.preventDefault();
      const file = event.dataTransfer.files[0];
      if (file) submit(root, file);
    });
    liveInput(root);
    samplePicker(root);
  });

  /* slide pages (Home, Working, Research): the rail marks the section in view ----------------------- */
  const marks = [...document.querySelectorAll(".rail a")];
  if (marks.length) {
    const watch = new IntersectionObserver((entries) => {
      for (const entry of entries) {
        if (!entry.isIntersecting) continue;
        for (const mark of marks) mark.setAttribute("aria-current", String(mark.hash === "#" + entry.target.id));
      }
    }, { threshold: 0.55 });
    document.querySelectorAll(".sl").forEach((s) => watch.observe(s));
  }

  /* dataset samples: pick a held-out test clip, see its preview, check its original ---------------- */
  function truth(sample) {
    const fake = sample.label === "fake";
    const box = el("p", "truth " + (fake ? "fake" : "real"));
    box.append(el("strong", "", "The true answer: " + (fake ? "Fake" : "Real")),
      " (" + sample.kind + ", from the " + sample.dataset + " test set)");
    return box;
  }

  function samplePicker(root) {
    const dialog = root.querySelector("[data-samples]");
    const open = root.querySelector("[data-samples-open]");
    if (!dialog || !open) return;
    const list = dialog.querySelector("[data-samples-list]");
    const count = dialog.querySelector("[data-samples-count]");
    const search = dialog.querySelector("[data-samples-search]");
    const filters = [...dialog.querySelectorAll("[data-samples-filter]")];
    let samples = null;

    const draw = () => {
      const words = search.value.trim().toLowerCase().split(/\s+/).filter(Boolean);
      const shown = samples.filter((s) => filters.every((f) => !f.value || s[f.dataset.samplesFilter] === f.value)
        && words.every((w) => [s.title, s.dataset, s.kind, s.label].join(" ").toLowerCase().includes(w)));
      list.replaceChildren(...shown.map((s) => {
        const item = el("li");
        const button = el("button", "sample");
        button.type = "button";
        button.append(el("strong", "", s.title), el("span", "small", s.dataset),
          el("span", "chip " + s.label, s.label === "fake" ? "Fake" : "Real"));
        button.addEventListener("click", () => pick(s));
        item.append(button);
        return item;
      }));
      count.textContent = shown.length + " of " + samples.length + " samples";
    };

    const fill = (select, values) => select.append(...values.map((v) => {
      const option = el("option", "", v);
      option.value = v;
      return option;
    }));

    async function load() {
      count.textContent = "Loading the samples...";
      try {
        const response = await fetch(API + "/api/dataset-samples");
        samples = (await response.json()).samples;
      } catch (error) {
        count.textContent = "The samples could not be loaded: the detector may be offline. Try again later.";
        return;
      }
      const distinct = (key) => [...new Set(samples.map((s) => s[key]))].sort();
      fill(filters.find((f) => f.dataset.samplesFilter === "dataset"), distinct("dataset"));
      fill(filters.find((f) => f.dataset.samplesFilter === "kind"), distinct("kind"));
      draw();
    }

    async function pick(sample) {
      dialog.close();
      const status = root.querySelector("[data-status]");
      status.textContent = "Loading the sample...";
      try {
        const response = await fetch(API + "/api/dataset-samples/" + sample.id + "/preview");
        if (!response.ok) throw new Error("preview " + response.status);
        const blob = await response.blob();
        const name = sample.dataset + ": " + sample.title;
        submit(root, new File([blob], name, { type: "video/mp4" }), "", sample);
      } catch (error) {
        status.textContent = "";
        render(root.querySelector("[data-result]"), { status: "rejected",
          reason: "The sample could not be loaded. The detector may be offline; try again later." });
      }
    }

    open.addEventListener("click", () => {
      dialog.showModal();
      if (!samples) load();
    });
    dialog.querySelector("[data-samples-close]").addEventListener("click", () => dialog.close());
    // A click on the backdrop lands on the dialog itself, outside its content.
    dialog.addEventListener("click", (event) => { if (event.target === dialog) dialog.close(); });
    search.addEventListener("input", () => samples && draw());
    filters.forEach((f) => f.addEventListener("change", () => samples && draw()));
  }

  /* live input: record a video or a voice, or take a picture, then check it like an upload ---- */
  // Raw microphone: the browser's noise suppression made a real voice score 0.99 (measured 11-18), because the
  // voice model learned that perfectly clean speech is synthetic.
  const LIVE = {
    video: { media: { video: true, audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: false } }, note: "Look at the camera and speak for 3 to 30 seconds, in a normal room." },
    voice: { media: { audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: false } }, note: "Speak for 3 to 30 seconds, in a normal room, without a noise-cancelling headset. Only the voice check runs." },
    picture: { media: { video: true }, note: "Face the camera. Only the face check runs on a picture." },
  };
  const MAX_SECONDS = 30, MIN_SECONDS = 4;   // a little over the 3 s the cleaning needs

  function liveInput(root) {
    const panel = root.querySelector("[data-live-panel]");
    if (!panel) return;
    const preview = panel.querySelector("[data-preview]");
    const note = panel.querySelector("[data-live-note]");
    const go = panel.querySelector("[data-live-go]");
    const cancel = panel.querySelector("[data-live-cancel]");
    let stream = null, recorder = null, chunks = [], kind = null, timer = null, started = 0;

    const stop = () => {
      clearInterval(timer);
      if (stream) stream.getTracks().forEach((track) => track.stop());
      stream = null; recorder = null; chunks = [];
      panel.hidden = true;
    };
    const fail = (reason) => {
      stop();
      render(root.querySelector("[data-result]"), { status: "rejected", reason });
    };

    root.querySelectorAll("[data-live]").forEach((button) => button.addEventListener("click", async () => {
      stop();
      kind = button.dataset.live;
      if (!navigator.mediaDevices || !window.MediaRecorder) {
        fail("This browser cannot record here. Recording needs a secure (https) page.");
        return;
      }
      try {
        stream = await navigator.mediaDevices.getUserMedia(LIVE[kind].media);
      } catch (error) {
        fail("The camera or microphone was not allowed. Allow it in the browser and try again.");
        return;
      }
      preview.hidden = kind === "voice";
      preview.srcObject = kind === "voice" ? null : stream;
      note.textContent = LIVE[kind].note;
      go.textContent = kind === "picture" ? "Take the picture" : "Start recording";
      go.disabled = false;
      panel.hidden = false;
    }));

    cancel.addEventListener("click", stop);

    go.addEventListener("click", () => {
      if (kind === "picture") {
        const canvas = document.createElement("canvas");
        canvas.width = preview.videoWidth; canvas.height = preview.videoHeight;
        canvas.getContext("2d").drawImage(preview, 0, 0);
        canvas.toBlob((blob) => {
          stop();
          submit(root, new File([blob], "picture.jpg", { type: "image/jpeg" }), "picture");
        }, "image/jpeg", 0.95);
        return;
      }
      if (recorder) {                       // the second press stops the recording
        if ((Date.now() - started) / 1000 < MIN_SECONDS) return;
        recorder.stop();
        return;
      }
      recorder = new MediaRecorder(stream);
      chunks = [];
      recorder.ondataavailable = (event) => event.data.size && chunks.push(event.data);
      recorder.onstop = () => {
        const blob = new Blob(chunks, { type: recorder.mimeType || "video/webm" });
        const mode = kind;
        stop();
        submit(root, new File([blob], "recording.webm", { type: blob.type }), mode);
      };
      recorder.start();
      started = Date.now();
      timer = setInterval(() => {
        const seconds = Math.floor((Date.now() - started) / 1000);
        go.textContent = seconds < MIN_SECONDS ? `Recording... ${seconds} s` : `Stop (${seconds} s)`;
        if (seconds >= MAX_SECONDS && recorder && recorder.state === "recording") recorder.stop();
      }, 250);
    });
  }

  /* lip-sync illustration (how it works): loudness and mouth opening over 2 s ---------------- */
  const svg = document.querySelector("[data-trace]");
  if (!svg) return;
  const W = 560, H = 250, N = 200, ns = "http://www.w3.org/2000/svg";
  const css = getComputedStyle(document.documentElement);
  const colour = (name) => css.getPropertyValue(name).trim();
  const env = (t) => {
    let v = 0;
    [[.08, .09], [.25, .07], [.42, .11], [.60, .08], [.77, .10], [.93, .08]].forEach(([c, w]) => {
      v += Math.exp(-((t - c) ** 2) / (2 * w * w * .25));
    });
    return Math.min(v, 1);
  };
  const sound = (i) => env(i / N) * (0.78 + 0.22 * Math.sin(i * 0.45));
  const mouth = (i, shift, amp) => env(Math.min(Math.max(i / N - shift, 0), 1)) * amp;
  const path = (f, y0, h) => Array.from({ length: N + 1 }, (_, i) =>
    (i ? "L" : "M") + (i / N * W).toFixed(1) + " " + (y0 - f(i) * h).toFixed(1)).join(" ");
  const line = (d, stroke, width) => {
    const p = document.createElementNS(ns, "path");
    p.setAttribute("d", d); p.setAttribute("fill", "none"); p.setAttribute("stroke", stroke);
    p.setAttribute("stroke-width", width); p.setAttribute("stroke-linejoin", "round");
    svg.append(p);
    return p;
  };
  for (let x = 0; x <= W; x += 70) {
    const g = document.createElementNS(ns, "line");
    g.setAttribute("x1", x); g.setAttribute("x2", x); g.setAttribute("y1", 8); g.setAttribute("y2", H - 8);
    g.setAttribute("stroke", colour("--line"));
    svg.append(g);
  }
  const loud = line(path(sound, 118, 100), colour("--sound"), 2);
  const lips = line(path((i) => mouth(i, 0, 1), 236, 100), colour("--face"), 2.5);
  const readout = document.querySelector("[data-readout]");
  const states = {
    real: { shift: 0, amp: 1, text: "In a real clip, the mouth opens when the sound gets loud." },
    fake: { shift: .09, amp: .75, text: "In a dubbed clip, the mouth lags the sound by about 180 ms. That gap is what the lip-sync check scores." },
  };
  const reduce = matchMedia("(prefers-reduced-motion: reduce)").matches;
  if (!reduce) {
    [loud, lips].forEach((p) => {
      const length = p.getTotalLength();
      p.style.strokeDasharray = length; p.style.strokeDashoffset = length;
      p.getBoundingClientRect();
      p.style.transition = "stroke-dashoffset 1.4s ease-out"; p.style.strokeDashoffset = 0;
    });
  }
  let current = { shift: 0, amp: 1 };
  function show(mode) {
    const target = states[mode];
    document.querySelectorAll(".toggle button").forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.mode === mode)));
    readout.textContent = target.text;
    lips.style.strokeDasharray = "none"; lips.style.transition = "none";
    if (reduce) { lips.setAttribute("d", path((i) => mouth(i, target.shift, target.amp), 236, 100)); current = target; return; }
    const from = { ...current }, t0 = performance.now();
    (function step(now) {
      const k = Math.min((now - t0) / 450, 1), e = 1 - (1 - k) ** 3;
      const shift = from.shift + (target.shift - from.shift) * e, amp = from.amp + (target.amp - from.amp) * e;
      lips.setAttribute("d", path((i) => mouth(i, shift, amp), 236, 100));
      if (k < 1) requestAnimationFrame(step); else current = target;
    })(t0);
  }
  document.querySelectorAll(".toggle button").forEach((b) => b.addEventListener("click", () => show(b.dataset.mode)));
})();
