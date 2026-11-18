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
  const MAX_MB = 90;
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

  function checkRows(checks, lines) {
    const box = el("div", "checks");
    const flag = lines ? lines.flag : null;
    for (const [key, label] of CHECKS) {
      const c = (checks || {})[key];
      const score = c && c.score != null ? c.score : null;
      const row = el("div", "crow " + key + (c && c.raised ? " raised" : ""));
      row.append(el("span", "cname", label));
      const track = el("span", "ctrack");
      const fill = el("i");
      fill.style.width = score != null ? score * 100 + "%" : "0%";
      track.append(fill);
      if (flag != null) {
        const tick = el("b", "flagtick");
        tick.style.left = flag * 100 + "%";
        track.append(tick);
      }
      row.append(track);
      row.append(el("span", "cval mono", score != null ? fmt(score) : "not checked"));
      if (c && c.reason) row.title = c.reason;
      box.append(row);
    }
    if (flag != null) {
      box.append(el("p", "scale small", "0 is real, 0.5 is neutral, 1 is fake. Below " + fmt(lines.lower)
        + ": no signs. " + fmt(lines.lower) + " to " + fmt(flag) + ": can't tell. From " + fmt(flag)
        + " (the mark on each bar): flagged. The flag mark is set so that about 5 in 100 real videos reach it."));
    }
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
      media.preload = "metadata";
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
    const readout = el("p", "readout mono");
    const show = (time) => {
      time = Math.min(Math.max(time, 0), end);
      cursor.style.left = time / end * 100 + "%";
      const part = partAt(timeline, time);
      const names = { face: "face", voice: "voice", sync: "lip sync" };
      const scores = Object.entries(part.checks).filter(([, v]) => v != null)
        .map(([k, v]) => names[k] + " " + fmt(v)).join(", ");
      readout.textContent = clock(time) + "  part " + clock(part.start) + " to " + clock(part.end) + ": "
        + (scores || "no model could run") + (part.reading != null
          ? "  (flag line " + fmt(lines.flag) + ")" : "");
      bar.setAttribute("aria-valuetext", readout.textContent);
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
    const legend = el("p", "legend small");
    legend.append(el("span", "key ok", "real side, below " + fmt(lines.lower)), el("span", "key unsure", "neutral, can't tell"),
      el("span", "key flag", "fake side, from " + fmt(lines.flag)), el("span", "key none", "not scored"));
    box.append(bar, readout, legend);
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
      const o = OUTCOMES[data.outcome] || OUTCOMES.cant_check;
      card.classList.add(o.cls);
      card.append(el("p", "outcome " + o.cls, o.text));
      if (data.summary) card.append(el("p", "", data.summary));
      if (data.file_check && data.file_check.length) {
        card.append(el("p", "filecheck", "File check: " + data.file_check.join("; ") + "."));
      }
      if (data.where) card.append(el("p", "where", data.where));
      if (file && data.outcome !== "cant_check") card.append(preview(file, data));
      card.append(checkRows(data.checks, data.lines));
      const trace = traceView(data.trace);
      if (trace) card.append(trace);
      if (data.how_sure) card.append(el("p", "sure", data.how_sure));
    }
    card.hidden = false;
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

  const STAGES = { received: "Received. Waiting for the detector", "file check": "Reading the file for AI labels",
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

  async function submit(root, file, mode) {
    const status = root.querySelector("[data-status]");
    const card = root.querySelector("[data-result]");
    card.hidden = true;
    if (file.size > MAX_MB * 1024 * 1024) {
      status.textContent = "";
      render(card, { status: "rejected", reason: "The file is over " + MAX_MB + " MB." });
      return;
    }
    status.textContent = "Uploading " + file.name + "...";
    const body = new FormData();
    body.append("file", file);
    if (mode) body.append("mode", mode);
    try {
      const response = await fetch(API + "/api/check", { method: "POST", body });
      let data = await response.json().catch(() => ({ status: "rejected", reason: "The server sent an unreadable answer." }));
      if (data.status === "queued") {
        status.textContent = STAGES.received + "...";
        data = await follow(status, data.job);
      }
      status.textContent = "";
      render(card, data, file);
    } catch (error) {
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
  });

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
