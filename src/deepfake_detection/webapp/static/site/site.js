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

  const el = (tag, cls, text) => {
    const node = document.createElement(tag);
    if (cls) node.className = cls;
    if (text !== undefined) node.textContent = text;
    return node;
  };

  function checkRows(checks) {
    const box = el("div", "checks");
    for (const [key, label] of CHECKS) {
      const c = (checks || {})[key];
      const row = el("div", "crow " + key + (c && c.raised ? " raised" : ""));
      row.append(el("span", "cname", label));
      const track = el("span", "ctrack");
      const fill = el("i");
      fill.style.width = c && c.percentile != null ? c.percentile + "%" : "0%";
      track.append(fill);
      row.append(track);
      row.append(el("span", "cval mono", c && c.percentile != null ? c.percentile + "th" : "not checked"));
      if (c && c.reason) row.title = c.reason;
      box.append(row);
    }
    return box;
  }

  function render(card, data) {
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
      if (data.where) card.append(el("p", "where", data.where));
      card.append(checkRows(data.checks));
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

  const STAGES = { received: "Received. Waiting for the detector", cleaning: "Cleaning the clip",
    "finding the face": "Finding the face", scoring: "Scoring", done: "Done" };
  const pause = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

  async function follow(status, job) {
    for (let tries = 0; tries < 400; tries++) {
      await pause(1500);
      const response = await fetch("/api/check/" + job);
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

  async function submit(root, file) {
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
    try {
      const response = await fetch("/api/check", { method: "POST", body });
      let data = await response.json().catch(() => ({ status: "rejected", reason: "The server sent an unreadable answer." }));
      if (data.status === "queued") {
        status.textContent = STAGES.received + "...";
        data = await follow(status, data.job);
      }
      status.textContent = "";
      render(card, data);
    } catch (error) {
      status.textContent = "";
      render(card, { status: "rejected", reason: "The upload failed. Check your connection and try again." });
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
  });

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
