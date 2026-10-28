"use strict";
/* Media Analysis dashboard for report schema 2 and 2.1. Plain JS, no network except the local server's own API.
   Every report string is written with textContent, never with markup insertion.
   Wording about evidence is NOT written here: it comes from the server (module "explanation" and report "summary",
   built by dfd.platform.explain from the module fields). This file arranges that text, rounds numbers for display only,
   and shows raw fields under "Technical details". Labels come from /api/vocabulary (decision D2: one copy, on the server). */
(function () {
  var VOCAB = { modules: {}, findings: {}, statuses: {}, groups: [] };
  var OBS_KIND = { file_metadata: "File metadata", heuristic: "Heuristic flag (descriptive only)",
                   marker_scan: "Marker scan (not a verification)", verification: "Verification result",
                   processing: "Processing record" };
  /* Fallback only: the server sends the same list as VOCAB.explanation_parts (dfd.platform.explain.EXPLANATION_PARTS). */
  var WHY_PARTS = [
    { id: "mock", label: "Scenario data", kinds: ["mock"] }, { id: "processing", label: "What we did", kinds: ["processing"] },
    { id: "examined", label: "What we examined", kinds: ["examined"] },
    { id: "observed", label: "What we observed", kinds: ["observation", "model_output"] },
    { id: "decision", label: "Why this was reported", kinds: ["decision"] }, { id: "interpretation", label: "What it means", kinds: ["interpretation"] },
    { id: "not_established", label: "What it does not establish", kinds: ["not_established"] }, { id: "limitation", label: "Limitations", kinds: ["limitation"] }
  ];
  function whyParts() { return Array.isArray(VOCAB.explanation_parts) && VOCAB.explanation_parts.length ? VOCAB.explanation_parts : WHY_PARTS; }
  var LIVE_TEXT = "Results come from the modules listed below; each states its own tested scope and limitations.";
  var SVGNS = "http://www.w3.org/2000/svg";
  var MODE = { live: false, accept: null, maxBytes: null };
  var current = { report: null, mock: false };

  function $(id) { return document.getElementById(id); }
  function el(tag, cls, text) {
    var e = document.createElement(tag);
    if (cls) e.className = cls;
    if (text !== undefined && text !== null) e.textContent = String(text);
    return e;
  }
  function svg(tag, attrs) {
    var e = document.createElementNS(SVGNS, tag);
    Object.keys(attrs || {}).forEach(function (k) { e.setAttribute(k, String(attrs[k])); });
    return e;
  }
  function clear(node) { while (node.firstChild) node.removeChild(node.firstChild); }
  function isObj(x) { return x !== null && typeof x === "object" && !Array.isArray(x); }
  function str(x) { return (x === null || x === undefined) ? "" : String(x); }
  function num(x) { return typeof x === "number" && isFinite(x); }
  function fmt(x, d) { return num(x) ? x.toFixed(d === undefined ? 2 : d) : ""; }
  function bytes(n) {
    if (!num(n)) return "";
    if (n < 1024) return n + " B";
    if (n < 1048576) return (n / 1024).toFixed(1) + " KB";
    if (n < 1073741824) return (n / 1048576).toFixed(1) + " MB";
    return (n / 1073741824).toFixed(2) + " GB";
  }
  function title(m) {
    if (isObj(m.explanation) && m.explanation.title) return m.explanation.title;
    var v = VOCAB.modules[m.module_id];
    return v && v.title ? v.title : str(m.module_id);
  }
  function moduleName(id) { var v = VOCAB.modules[id]; return v && v.title ? v.title : str(id); }

  /* ---------------- icons: one shape per group, so meaning never rests on colour alone ---------------- */
  function icon(group) {
    var s = svg("svg", { viewBox: "0 0 20 20", "class": "icon", "aria-hidden": "true", focusable: "false" });
    var stroke = { fill: "none", stroke: "currentColor", "stroke-width": "2", "stroke-linecap": "round", "stroke-linejoin": "round" };
    function add(tag, a) { var e = svg(tag, Object.assign({}, stroke, a)); s.appendChild(e); return e; }
    if (group === "attention") { add("path", { d: "M10 2.5 18 17H2z", fill: "currentColor", stroke: "none" });
      add("path", { d: "M10 8v4", stroke: "#fff" }); add("path", { d: "M10 14.6v.1", stroke: "#fff" }); }
    else if (group === "nothing_found" || group === "informational") { add("circle", { cx: 10, cy: 10, r: 7.5 }); add("path", { d: "M6.5 10h7" }); }
    else if (group === "credential") { add("path", { d: "M10 2 16.5 4.5V9c0 4.2-2.8 7.4-6.5 9-3.7-1.6-6.5-4.8-6.5-9V4.5z" }); add("path", { d: "M7 10l2 2 4-4" }); }
    else if (group === "inconclusive") { add("circle", { cx: 10, cy: 10, r: 7.5 }); add("path", { d: "M7.8 8a2.3 2.3 0 1 1 3.2 2.1c-.7.3-1 .8-1 1.5v.4" }); add("path", { d: "M10 14.6v.1" }); }
    else if (group === "experimental") { add("path", { d: "M7.5 2.5h5M8.5 2.5v5L4 16a1 1 0 0 0 .9 1.5h10.2A1 1 0 0 0 16 16l-4.5-8.5v-5" }); add("path", { d: "M6 13h8" }); }
    else if (group === "did_not_run") { add("path", { d: "M12.5 3.5a4 4 0 0 0-4.8 5.3L3 13.5 6.5 17l4.7-4.7a4 4 0 0 0 5.3-4.8l-2.6 2.6-2.4-.6-.6-2.4z" }); }
    else { add("circle", { cx: 10, cy: 10, r: 7.5 }); add("path", { d: "M4.7 15.3 15.3 4.7" }); }
    return s;
  }

  /* ---------------- report shape ---------------- */
  function shapeProblems(r) {
    var p = [];
    if (!isObj(r)) return ["The report is not an object."];
    ["schema_version", "input", "probe", "modules", "sections", "coverage", "source_summary", "status", "statement"].forEach(function (k) {
      if (!(k in r)) p.push("Missing key: " + k);
    });
    if (p.length) return p;
    var ok = Array.isArray(VOCAB.schema_versions) ? VOCAB.schema_versions : ["2", "2.1"];
    if (ok.indexOf(str(r.schema_version)) < 0) p.push("Unsupported schema_version: " + str(r.schema_version));
    if (!Array.isArray(r.modules)) p.push("modules is not a list.");
    else r.modules.forEach(function (m, i) {
      if (!isObj(m) || !m.module_id || !m.status || !m.section) p.push("Module " + i + " lacks module_id, status or section.");
    });
    if (!isObj(r.sections)) p.push("sections is not an object.");
    if (!isObj(r.coverage)) p.push("coverage is not an object.");
    if (!isObj(r.probe)) p.push("probe is not an object.");
    if (!isObj(r.status)) p.push("status is not an object.");
    if (!isObj(r.input)) p.push("input is not an object.");
    return p;
  }

  function showError(lines) {
    var box = $("error");
    clear(box);
    box.appendChild(el("strong", null, "This report cannot be displayed."));
    var ul = el("ul");
    lines.forEach(function (l) { ul.appendChild(el("li", null, l)); });
    box.appendChild(ul);
    box.hidden = false;
    $("report").hidden = true;
  }

  /* ---------------- banner: LIVE or MOCK, never ambiguous ---------------- */
  function isMockReport(r) {
    return r.source_summary === "mock" || r.source_summary === "mixed" || r.modules.some(function (m) { return m.source === "mock"; });
  }
  function renderBanner(r, live) {
    var b = $("banner");
    clear(b);
    b.className = "banner";
    var showMock = isMockReport(r);
    var showLive = live === true && (r.source_summary === "real" || r.source_summary === "mixed");
    if (showLive) {
      b.appendChild(el("div", "live-line", "LIVE ANALYSIS"));
      b.appendChild(el("div", "live-sub", LIVE_TEXT));
      b.className = "banner live";
    }
    if (showMock) {
      var t = r.source_summary === "mixed"
        ? "MOCK DATA: some module results below are invented scenario data, not measurements."
        : "MOCK DATA: every result below is invented scenario data. No model ran. Nothing here is a measurement.";
      b.appendChild(el("div", "mock-line", t));
      b.className = showLive ? "banner live mixed" : "banner mock";
    }
    b.hidden = !(showLive || showMock);
  }

  /* ---------------- A. file overview ---------------- */
  function factList(rows) {
    var dl = el("dl", "facts");
    rows.forEach(function (kv) {
      if (kv[1] === "" || kv[1] === null || kv[1] === undefined) return;
      var d = el("div");
      d.appendChild(el("dt", null, kv[0]));
      d.appendChild(el("dd", null, kv[1]));
      dl.appendChild(d);
    });
    return dl;
  }
  function stateText(st) {
    if (st === "ok") return "Complete";
    if (st === "partial") return "Partial: some checks did not complete";
    if (st === "failed") return "Report failed: no check completed";
    return str(st);
  }
  function renderMeta(r) {
    var s = $("meta"); clear(s);
    var inp = r.input, p = r.probe, st = r.status;
    var head = el("div", "ov-head");
    var left = el("div");
    left.appendChild(el("h2", "sr-only", "File analysed")).id = "overview-h";
    left.appendChild(el("p", "ov-file", str(inp.filename)));
    head.appendChild(left);
    s.appendChild(head);
    var kind = { video: "Video", image: "Image", audio: "Audio" }[p.media_type] || str(p.media_type);
    var streams = p.media_type === "video" ? (p.has_audio === true ? "Picture and sound" : (p.has_audio === false ? "Picture only, no sound" : "")) : "";
    s.appendChild(factList([
      ["Type", kind], ["Duration", num(p.duration_s) ? fmt(p.duration_s, 1) + " s" : ""],
      ["Dimensions", (p.width && p.height) ? p.width + " x " + p.height + " px" : ""], ["Streams", streams],
      ["Size", bytes(inp.size_bytes)], ["Analysis", stateText(st.state)]
    ]));
    var cov = r.coverage, cnt = function (k) { return Array.isArray(cov[k]) ? cov[k].length : 0; };
    if (st.state === "failed") {
      s.appendChild(el("p", "state-note failed", "No check completed, so nothing was analysed. This is not a statement about the file."));
    } else if (st.state === "partial") {
      var n = cnt("failed") + cnt("timed_out") + cnt("unsupported");
      var note = el("p", "state-note partial", n + " of " + r.modules.length + " checks did not complete (" + cnt("failed") + " failed, " +
        cnt("timed_out") + " timed out, " + cnt("unsupported") + " not available). The results below cover the other checks only.");
      note.id = "partial-note";
      s.appendChild(note);
    }
    var det = el("details");
    det.appendChild(el("summary", null, "File identifiers"));
    var kv = el("dl", "kv");
    [["Request", r.request_id], ["SHA-256", inp.sha256], ["MIME type", inp.mime_type], ["Container", p.container]].forEach(function (x) {
      if (!x[1]) return;
      kv.appendChild(el("dt", null, x[0]));
      kv.appendChild(el("dd", "mono", str(x[1])));
    });
    det.appendChild(kv);
    s.appendChild(det);
  }

  /* ---------------- B. plain-language summary (server-written) ---------------- */
  function renderSummary(r) {
    var s = $("summary"); clear(s);
    var h = el("h2", null, "What the checks found");
    h.id = "summary-h";
    s.appendChild(h);
    var sm = r.summary;
    if (!isObj(sm)) {
      s.appendChild(el("p", "empty", "This report was produced in an older format without a written summary. Each check is listed below."));
      return;
    }
    s.appendChild(el("p", "summary-head", sm.headline));
    var ul = el("ul", "counts");
    (sm.groups || []).forEach(function (g) {
      var li = el("li");
      var a = el("a", "g-" + g.id);
      a.href = "#group-" + g.id;
      a.appendChild(icon(g.id));
      a.appendChild(el("span", null, g.modules.length + " " + g.label.toLowerCase()));
      li.appendChild(a);
      ul.appendChild(li);
    });
    s.appendChild(ul);
    if (Array.isArray(sm.media_facts) && sm.media_facts.length) {
      var mf = el("div", "media-facts");
      sm.media_facts.forEach(function (f) {
        var d = el("p", "media-fact");
        d.appendChild(el("b", null, f.label));
        d.appendChild(el("span", null, f.text));
        mf.appendChild(d);
      });
      s.appendChild(mf);
    }
    s.appendChild(glance(r));
    if (sm.no_overall_score) s.appendChild(el("p", "no-score", sm.no_overall_score));
  }
  /* One row per check, in the order of the groups below; each row links to its card. Text is the server's headline and label. */
  function glance(r) {
    var wrap = el("div", "glance");
    wrap.appendChild(el("h3", "glance-h", "Every check at a glance"));
    var tbl = el("table", "glance-t");
    var th = el("tr");
    ["Check", "Result"].forEach(function (c) { var x = el("th", null, c); x.scope = "col"; th.appendChild(x); });
    var thead = el("thead"); thead.appendChild(th); tbl.appendChild(thead);
    var tb = el("tbody");
    var byId = {};
    r.modules.forEach(function (m) { byId[m.module_id] = m; });
    groupsOf(r).forEach(function (g) {
      g.modules.forEach(function (id) {
        var m = byId[id]; if (!m) return;
        var e = isObj(m.explanation) ? m.explanation : null;
        var tr = el("tr", "g-" + g.id);
        tr.setAttribute("data-module", str(id));
        var c1 = el("td"), a = el("a", null);
        a.href = "#check-" + str(id);
        a.appendChild(icon(g.id));
        a.appendChild(el("span", null, title(m)));
        if (experimental(m)) a.appendChild(el("span", "badge exp small", "EXPERIMENTAL"));
        if (m.source === "mock") a.appendChild(el("span", "tag mock badge small", "MOCK"));
        c1.appendChild(a);
        var c2 = el("td");
        c2.appendChild(el("span", "glance-label", assessmentOf(m) ? m.assessment.verdict_label : chipLabel(m, e)));
        if (e && e.headline) c2.appendChild(el("span", "glance-head", e.headline));
        tr.appendChild(c1); tr.appendChild(c2); tb.appendChild(tr);
      });
    });
    tbl.appendChild(tb);
    wrap.appendChild(tbl);
    return wrap;
  }
  function chipLabel(m, e) {
    return m.status === "completed" ? ((e && e.finding_label) || VOCAB.findings[m.finding] || str(m.finding))
      : ((e && e.status_label) || VOCAB.statuses[m.status] || str(m.status));
  }

  function renderConflicts(r) {
    var s = $("conflicts"); clear(s);
    var list = isObj(r.summary) && Array.isArray(r.summary.disagreements) ? r.summary.disagreements
      : (Array.isArray(r.conflicts) ? r.conflicts.map(function (c) { return { modules: c.modules, text: c.note }; }) : []);
    s.hidden = !list.length;
    if (!list.length) return;
    s.appendChild(el("h2", "h3", "Checks that point in different directions"));
    list.forEach(function (x) {
      s.appendChild(el("p", null, str(x.text)));
      s.appendChild(el("p", "what", "Checks involved: " + (Array.isArray(x.modules) ? x.modules.map(moduleName).join(", ") : "")));
    });
  }

  /* ---------------- evidence visuals (only real computed data from the module) ---------------- */
  function offsetCurve(ev) {
    var xs = ev.offsets, ys = ev.mean_dist;
    if (!Array.isArray(xs) || !Array.isArray(ys) || xs.length !== ys.length || xs.length < 2 || !ys.every(num)) return null;
    var W = 560, H = 200, L = 44, R = 12, T = 12, B = 36;
    var x0 = xs[0], x1 = xs[xs.length - 1];
    var lo = Math.min.apply(null, ys), hi = Math.max.apply(null, ys);
    if (hi - lo < 1e-9) { hi = lo + 1; }
    var pad = (hi - lo) * 0.08;
    lo -= pad; hi += pad;
    function X(v) { return L + (v - x0) / (x1 - x0) * (W - L - R); }
    function Y(v) { return T + (hi - v) / (hi - lo) * (H - T - B); }
    var root = svg("svg", { viewBox: "0 0 " + W + " " + H, "class": "curve", role: "img",
      "aria-label": "Embedding distance at each audio-visual offset from " + x0 + " to " + x1 + " frames; lowest at " + ev.best_offset +
        " frames; band reported as consistent " + ev.band[0] + " to " + ev.band[1] + " frames." });
    if (Array.isArray(ev.band)) root.appendChild(svg("rect", { "class": "band", x: X(ev.band[0] - 0.5), y: T, width: X(ev.band[1] + 0.5) - X(ev.band[0] - 0.5), height: H - T - B }));
    root.appendChild(svg("line", { "class": "axis", x1: L, y1: H - B, x2: W - R, y2: H - B }));
    root.appendChild(svg("line", { "class": "axis", x1: L, y1: T, x2: L, y2: H - B }));
    [x0, -10, -5, 0, 5, 10, x1].forEach(function (t) {
      if (t < x0 || t > x1) return;
      root.appendChild(svg("line", { "class": "axis", x1: X(t), y1: H - B, x2: X(t), y2: H - B + 4 }));
      var tx = svg("text", { x: X(t), y: H - B + 16, "text-anchor": "middle" }); tx.textContent = (t > 0 ? "+" : "") + t; root.appendChild(tx);
    });
    var xl = svg("text", { x: (L + W - R) / 2, y: H - 4, "text-anchor": "middle" }); xl.textContent = "audio-visual offset (frames, 25 per second)"; root.appendChild(xl);
    var yl = svg("text", { x: 12, y: (T + H - B) / 2, transform: "rotate(-90 12 " + (T + H - B) / 2 + ")", "text-anchor": "middle" });
    yl.textContent = "distance (lower = closer)"; root.appendChild(yl);
    root.appendChild(svg("polyline", { "class": "line", points: xs.map(function (x, i) { return X(x).toFixed(1) + "," + Y(ys[i]).toFixed(1); }).join(" ") }));
    var bi = xs.indexOf(ev.best_offset);
    if (bi >= 0) {
      root.appendChild(svg("circle", { "class": "best", cx: X(xs[bi]), cy: Y(ys[bi]), r: 4 }));
      var bl = svg("text", { x: X(xs[bi]), y: Math.max(T + 10, Y(ys[bi]) - 8), "text-anchor": "middle" });
      bl.textContent = "best " + (ev.best_offset > 0 ? "+" : "") + ev.best_offset; root.appendChild(bl);
    }
    var fig = el("figure", "figure");
    fig.appendChild(root);
    fig.appendChild(el("figcaption", null, ev.text));
    return fig;
  }
  function filmstrip(ev) {
    if (!Array.isArray(ev.frames)) return null;
    var fig = el("figure", "figure");
    var strip = el("div", "filmstrip");
    ev.frames.forEach(function (uri, i) {
      if (typeof uri !== "string" || uri.indexOf("data:image/jpeg;base64,") !== 0) return;
      var t = Array.isArray(ev.times_s) && num(ev.times_s[i]) ? fmt(ev.times_s[i], 2) + " s" : "";
      var f = el("figure");
      var img = el("img");
      img.src = uri;
      img.alt = "Model input frame" + (t ? " at " + t : "");
      img.width = 112; img.height = 112;
      f.appendChild(img);
      f.appendChild(el("span", null, t));
      strip.appendChild(f);
    });
    fig.appendChild(strip);
    fig.appendChild(el("figcaption", null, ev.text));
    return fig;
  }

  /* ---------------- D. one card per check ---------------- */
  function experimental(m) {
    return Array.isArray(m.warnings) && m.warnings.some(function (w) { return isObj(w) && /^EXPERIMENTAL/.test(str(w.code)); });
  }
  function whyPanel(m) {
    var det = el("details", "why-box");
    det.appendChild(el("summary", null, "How this result was reached"));
    var box = el("div", "why");
    var e = m.explanation;
    if (isObj(e) && Array.isArray(e.sentences)) {
      whyParts().forEach(function (part) {
        var kinds = Array.isArray(part.kinds) ? part.kinds : [part.id];
        var items = e.sentences.filter(function (x) { return isObj(x) && kinds.indexOf(x.kind) >= 0 && !(assessmentOf(m) && isScoreItem(x)); });
        if (!items.length) return;
        var sec = el("div", "why-part");
        sec.setAttribute("data-part", str(part.id));
        sec.appendChild(el("h4", null, part.label));
        var ul = el("ul");
        items.forEach(function (x) {
          var li = el("li", null, x.text + " ");
          li.title = "From: " + (x.sources || []).join(", ");
          li.appendChild(el("span", "src", "[from " + (x.sources || []).join(", ") + "]"));
          ul.appendChild(li);
        });
        sec.appendChild(ul);
        box.appendChild(sec);
      });
    } else {
      box.appendChild(el("p", "empty", "No plain-language explanation is stored in this report format. The raw fields are under Technical details."));
    }
    (Array.isArray(m.evidence) ? m.evidence : []).forEach(function (ev) {
      if (!isObj(ev)) return;
      var sec = el("div");
      sec.appendChild(el("h4", null, ev.kind === "frames" ? "Frames the model analysed" : "Audio-visual offset search"));
      var fig = ev.kind === "offset_curve" ? offsetCurve(ev) : (ev.kind === "frames" ? filmstrip(ev) : null);
      if (fig) { sec.appendChild(fig); box.appendChild(sec); }
    });
    det.appendChild(box);
    return det;
  }
  function kvRows(dl, rows) {
    rows.forEach(function (x) {
      if (x[1] === null || x[1] === undefined || x[1] === "") return;
      dl.appendChild(el("dt", null, x[0]));
      dl.appendChild(el("dd", x[2] || null, str(x[1])));
    });
  }
  function listBlock(parent, heading, items, cls, fmtItem) {
    if (!Array.isArray(items) || !items.length) return;
    var d = el("div");
    d.appendChild(el("h4", null, heading));
    var ul = el("ul");
    items.forEach(function (x) { ul.appendChild(el("li", cls, fmtItem ? fmtItem(x) : x)); });
    d.appendChild(ul);
    parent.appendChild(d);
  }
  function codeMsg(w) { return str(isObj(w) ? w.code : "") + ": " + str(isObj(w) ? w.message : w); }
  function techPanel(m) {
    var det = el("details", "tech-box");
    det.appendChild(el("summary", null, "Technical details"));
    var box = el("div", "tech");
    var dl = el("dl", "kv");
    var s = isObj(m.score) ? m.score : {};
    kvRows(dl, [["Module id", m.module_id, "mono"], ["Version", m.version, "mono"], ["Source", m.source], ["Status", m.status],
      ["Finding code", m.finding, "mono"], [assessmentOf(m) ? scoreKindLabel(m.assessment) : "Raw score", isObj(m.score) ? str(s.value) + (m.source === "mock" ? " (MOCK number)" : "") +
        (experimental(m) ? " (experimental, for inspection only; no decision is made from it)" : "") : "No score reported."],
      ["Score meaning", s.meaning], ["Calibrated", isObj(m.score) ? (s.calibrated ? "yes" : "no") : ""],
      ["Decision threshold", s.threshold], ["Units analysed", m.units_analysed], ["Time", num(m.timing_ms) ? fmt(m.timing_ms, 0) + " ms" : ""]]);
    if (isObj(m.model)) {
      kvRows(dl, [["Model", m.model.name], ["Model version", m.model.version, "mono"], ["Weights", m.model.weights_sha256, "mono"],
        ["Licence", m.model.licence], ["Evaluation", m.model.evaluation]]);
    }
    kvRows(dl, [["Tested scope", m.tested_scope], ["Reason", m.reason],
      ["Explanation template", isObj(m.explanation) ? m.explanation.template_id : "", "mono"]]);
    if (assessmentOf(m) && isObj(m.explanation)) {
      if (isScoreItem(m.explanation.key_figure)) kvRows(dl, [["Key figure (" + str(m.explanation.key_figure.label).toLowerCase() + ")", m.explanation.key_figure.text]]);
    }
    box.appendChild(dl);
    if (assessmentOf(m) && isObj(m.explanation) && Array.isArray(m.explanation.sentences)) {
      listBlock(box, "Score statements from the explanation", m.explanation.sentences.filter(isScoreItem), null, function (x) { return x.text; });
    }
    listBlock(box, "Observations", m.observations, null, function (o) {
      return (isObj(o) ? (OBS_KIND[o.kind] || str(o.kind)) : "") + ": " + str(isObj(o) ? o.text : o) +
        (isObj(o) && o.value !== null && o.value !== undefined && typeof o.value !== "object" ? " (" + str(o.value) + ")" : "");
    });
    listBlock(box, "Limitations", m.limitations, null);
    listBlock(box, "Warnings", m.warnings, "warn", codeMsg);
    listBlock(box, "Errors", m.errors, "err", codeMsg);
    listBlock(box, "Segments", m.segments, null, function (g) {
      return str(isObj(g) ? g.start_s : "") + " s to " + str(isObj(g) ? g.end_s : "") + " s" + (isObj(g) && g.note ? ": " + str(g.note) : "");
    });
    det.appendChild(box);
    return det;
  }
  function groupOf(m) {
    if (isObj(m.explanation) && m.explanation.group) return m.explanation.group;
    return m.status === "completed" ? "inconclusive" : ({ not_applicable: "not_applicable", unsupported: "unsupported" }[m.status] || "did_not_run");
  }

  /* ---------------- assessed cards (D-087, contract section 1) ----------------
     Order: verdict label, explanation headline, supporting signals and evidence, model and modality, limitations, uncertainty.
     The raw score appears only under Technical details. Everything written here comes from the report or /api/vocabulary. */
  /* A sentence or key figure built from score.value states the raw score; on an assessed card it lives only under Technical details. */
  function isScoreItem(x) { return isObj(x) && Array.isArray(x.sources) && x.sources.some(function (q) { return /^score\./.test(str(q)); }) && x.kind !== "decision"; }
  function assessmentOf(m) { return isObj(m.assessment) && str(m.assessment.verdict_label) ? m.assessment : null; }
  function verdictClasses(a) {
    var tier = str(a.tier) || "none", v = str(a.verdict), c = ["assessed", "tier-" + tier];
    if (tier === "experimental") return c;                  // an experimental card never takes the style of a detector flag
    if (/^(likely_|possible_)/.test(v)) { c.push("v-flag"); if (/^possible_/.test(v)) c.push("v-weak"); }
    else if (v === "credentials_invalid") c.push("v-bad");
    else if (v === "analysis_failed" || v === "unsupported_input") c.push("v-failed");
    else if (v === "inconclusive") c.push("v-inconclusive");
    else if (v === "credentials_verified") c.push("v-credential");
    return c;
  }
  function modalityText(m) {
    var v = VOCAB.modules[m.module_id];
    return v && Array.isArray(v.applies_to) && v.applies_to.length ? v.applies_to.join(", ") : "";
  }
  function scoreKindLabel(a) {
    var k = isObj(a) && isObj(a.raw) ? a.raw.score_kind : "";
    return k === "ranking_score" ? "Ranking score" : (k === "model_output" ? "Model output" : "Score");
  }
  function renderAssessedCard(m, a, g) {
    var e = isObj(m.explanation) ? m.explanation : null;
    var quiet = ["not_applicable", "unsupported"].indexOf(g) >= 0;
    var card = el("article", "card g-" + g + " " + verdictClasses(a).join(" ") + (m.finding === "potential_mismatch" ? " mismatch" : "") + (quiet ? " quiet" : ""));
    card.setAttribute("data-module", str(m.module_id));
    card.setAttribute("data-status", str(m.status));
    card.setAttribute("data-group", g);
    card.setAttribute("data-tier", str(a.tier));
    card.setAttribute("data-verdict", str(a.verdict));
    card.setAttribute("aria-label", title(m));
    card.id = "check-" + str(m.module_id);
    var head = el("div", "card-head");
    var t = el("div", "card-title");
    t.appendChild(icon(g));
    t.appendChild(el("span", "kicker", title(m)));
    head.appendChild(t);
    var badges = el("div", "badges");
    if (m.source === "mock") badges.appendChild(el("span", "tag mock badge", "MOCK"));
    if (experimental(m) || a.tier === "experimental") badges.appendChild(el("span", "badge exp", "EXPERIMENTAL"));
    if (e && Array.isArray(e.sentences) && e.sentences.some(function (x) { return (x.sources || []).some(function (s) { return /AUDIO_NOT_SPEECH_LIKE/.test(s); }); })) {
      badges.appendChild(el("span", "badge caveat", "Audio may not be speech"));
    }
    head.appendChild(badges);
    card.appendChild(head);
    /* (1) verdict label */
    var vl = el("h3", "verdict-label");
    vl.appendChild(icon(g));
    vl.appendChild(el("span", "verdict-text", a.verdict_label));
    card.appendChild(vl);
    if (a.meaning && !quiet) card.appendChild(el("p", "what verdict-meaning", a.meaning));
    if (m.status !== "completed" && m.reason) card.appendChild(el("p", "reason", "Reason: " + str(m.reason)));
    /* (2) explanation headline */
    card.appendChild(el("p", "headline", e ? e.headline : a.verdict_label));
    if (e && isObj(e.key_figure) && !isScoreItem(e.key_figure)) {
      var k = el("p", "keyfig");
      k.appendChild(el("b", null, e.key_figure.label + ": "));
      k.appendChild(el("span", null, e.key_figure.text));
      card.appendChild(k);
    }
    if (a.tier === "experimental") {
      card.appendChild(el("p", "exp-signal", "Experimental signal, not a finding. It is shown for inspection only and no decision is made from it. " +
        "The number is under Technical details."));
    }
    /* (3) supporting signals and evidence */
    var ev = Array.isArray(a.evidence) ? a.evidence.filter(function (x) { return isObj(x) && x.text; }) : [];
    var sup = [];
    if (e && Array.isArray(e.sentences)) {
      sup = e.sentences.filter(function (x) { return isObj(x) && ["observation", "decision"].indexOf(x.kind) >= 0 && !isScoreItem(x); });
    }
    if (ev.length || sup.length) {
      var sg = el("div", "signals");
      sg.setAttribute("data-part", "signals");
      sg.appendChild(el("h4", "sec-h", "Supporting signals and evidence"));
      var ul = el("ul");
      sup.forEach(function (x) { ul.appendChild(el("li", "sig", x.text)); });
      ev.forEach(function (x) { var li = el("li", "ev", x.text); li.setAttribute("data-evidence", str(x.id)); ul.appendChild(li); });
      sg.appendChild(ul);
      card.appendChild(sg);
    }
    /* (4) model and modality */
    var ml = el("div", "model-line");
    ml.setAttribute("data-part", "model");
    var mname = isObj(m.model) ? m.model.name : "";
    [["Check", title(m)], ["Model", mname], ["Applies to", modalityText(m)]].forEach(function (x) {
      if (!x[1]) return;
      var sp = el("span");
      sp.appendChild(el("b", null, x[0] + ": "));
      sp.appendChild(document.createTextNode(str(x[1])));
      ml.appendChild(sp);
    });
    card.appendChild(ml);
    /* (5) limitations */
    var lim = (Array.isArray(m.limitations) ? m.limitations : []).filter(function (x) { return str(x); });
    if (lim.length && !quiet) {
      var lb = el("div", "limits-list");
      lb.setAttribute("data-part", "limitations");
      lb.appendChild(el("h4", "sec-h", "Limitations"));
      var lu = el("ul");
      lim.forEach(function (x) { lu.appendChild(el("li", null, x)); });
      lb.appendChild(lu);
      card.appendChild(lb);
    }
    /* (6) uncertainty */
    var u = isObj(a.uncertainty) ? a.uncertainty : null, cal = isObj(a.calibration) ? a.calibration : null;
    if ((u && u.text) || (cal && cal.statement)) {
      var ub = el("div", "uncert");
      ub.setAttribute("data-part", "uncertainty");
      ub.appendChild(el("h4", "sec-h", "Uncertainty"));
      if (u && u.text) {
        var p1 = el("p");
        p1.appendChild(el("span", "lvl", ({ moderate: "Moderate uncertainty. ", high: "High uncertainty. ", not_assessed: "Uncertainty not assessed. " })[u.level] || ""));
        p1.appendChild(document.createTextNode(u.text));
        ub.appendChild(p1);
      }
      if (cal && cal.statement) ub.appendChild(el("p", null, cal.statement));
      card.appendChild(ub);
    }
    card.appendChild(whyPanel(m));
    card.appendChild(techPanel(m));
    return card;
  }
  function renderCard(m) {
    var e = isObj(m.explanation) ? m.explanation : null;
    var g = groupOf(m);
    var asm = assessmentOf(m);
    if (asm) return renderAssessedCard(m, asm, g);
    var card = el("article", "card g-" + g + (m.finding === "potential_mismatch" ? " mismatch" : "") +
      (["not_applicable", "unsupported"].indexOf(g) >= 0 ? " quiet" : ""));
    card.setAttribute("data-module", str(m.module_id));
    card.setAttribute("data-status", str(m.status));
    card.setAttribute("data-group", g);
    card.id = "check-" + str(m.module_id);
    var head = el("div", "card-head");
    var t = el("div", "card-title");
    t.appendChild(icon(g));
    t.appendChild(el("h3", null, title(m)));
    head.appendChild(t);
    var badges = el("div", "badges");
    if (m.source === "mock") badges.appendChild(el("span", "tag mock badge", "MOCK"));
    if (experimental(m)) badges.appendChild(el("span", "badge exp", "EXPERIMENTAL"));
    var label = chipLabel(m, e);
    var chip = el("span", "chip", label);
    if (m.status === "completed") chip.setAttribute("data-finding", str(m.finding));
    badges.appendChild(chip);
    head.appendChild(badges);
    card.appendChild(head);
    card.appendChild(el("p", "headline", e ? e.headline : label));
    var quiet = ["not_applicable", "unsupported"].indexOf(g) >= 0;
    var what = e && e.what_it_checks ? e.what_it_checks : ((VOCAB.modules[m.module_id] || {}).what_it_checks || "");
    if (what && !quiet) card.appendChild(el("p", "what", what));
    if (m.status !== "completed" && m.reason) card.appendChild(el("p", "reason", "Reason: " + str(m.reason)));
    if (e && isObj(e.key_figure)) {
      var k = el("p", "keyfig");
      k.appendChild(el("b", null, e.key_figure.label + ": "));
      k.appendChild(el("span", null, e.key_figure.text));
      card.appendChild(k);
    }
    if (experimental(m) && isObj(m.score) && num(m.score.value)) {
      card.appendChild(el("p", "exp-signal", "Experimental signal, not a finding: ranking score " + fmt(m.score.value) +
        ". Open “How this result was reached” for why it cannot be relied on."));
    }
    if (e && Array.isArray(e.sentences) && e.sentences.some(function (x) { return (x.sources || []).some(function (s) { return /AUDIO_NOT_SPEECH_LIKE/.test(s); }); })) {
      badges.insertBefore(el("span", "badge caveat", "Audio may not be speech"), chip);
    }
    if (e && e.does_not_mean && !quiet) card.appendChild(el("p", "dnm", e.does_not_mean));
    card.appendChild(whyPanel(m));
    card.appendChild(techPanel(m));
    return card;
  }

  /* ---------------- C. results grouped by meaning ---------------- */
  function groupsOf(r) {
    if (isObj(r.summary) && Array.isArray(r.summary.groups)) return r.summary.groups;
    var order = ["inconclusive", "not_applicable", "unsupported", "did_not_run"], by = {};
    r.modules.forEach(function (m) { var g = groupOf(m); (by[g] = by[g] || []).push(m.module_id); });
    var label = { inconclusive: "Results", not_applicable: "Did not apply to this file", unsupported: "Not available", did_not_run: "Did not complete" };
    return order.filter(function (g) { return by[g]; }).map(function (g) { return { id: g, label: label[g], modules: by[g] }; });
  }
  function renderEvidence(r) {
    var host = $("evidence"); clear(host);
    var byId = {};
    r.modules.forEach(function (m) { byId[m.module_id] = m; });
    groupsOf(r).forEach(function (g) {
      var sec = el("section", "group g-" + g.id);
      sec.id = "group-" + g.id;
      var h = el("h2", "h3");
      h.appendChild(icon(g.id));
      h.appendChild(el("span", null, g.label));
      h.appendChild(el("span", "count", "(" + g.modules.length + ")"));
      sec.appendChild(h);
      var cards = el("div", "cards");
      g.modules.forEach(function (id) { if (byId[id]) cards.appendChild(renderCard(byId[id])); });
      sec.appendChild(cards);
      host.appendChild(sec);
    });
    if (!r.modules.length) host.appendChild(el("p", "empty", "No checks are listed in this report."));
  }

  /* ---------------- F. report-level technical details ---------------- */
  function renderReportTech(r) {
    var s = $("report-tech"); clear(s);
    var dl = el("dl", "kv");
    kvRows(dl, [["Request id", r.request_id, "mono"], ["Schema version", r.schema_version], ["Result source", r.source_summary],
      ["Processing time", num(r.status.processing_time_ms) ? fmt(r.status.processing_time_ms, 0) + " ms" : ""], ["Report state", r.status.state]]);
    s.appendChild(dl);
    var c = $("coverage"); clear(c);
    c.appendChild(el("h3", "h3", "Coverage: every check by status"));
    var grid = el("div", "covgrid");
    ["completed", "not_applicable", "unsupported", "failed", "timed_out", "skipped"].forEach(function (st) {
      var items = Array.isArray(r.coverage[st]) ? r.coverage[st] : [];
      var box = el("div", "covbox st-" + st);
      box.setAttribute("data-coverage", st);
      box.appendChild(el("h3", null, (VOCAB.statuses[st] || st) + " (" + items.length + ")"));
      if (!items.length) box.appendChild(el("p", "empty", "None."));
      else {
        var ul = el("ul");
        items.forEach(function (i) {
          ul.appendChild(el("li", null, moduleName(isObj(i) ? i.module_id : "") + (isObj(i) && i.reason ? ": " + str(i.reason) : "")));
        });
        box.appendChild(ul);
      }
      grid.appendChild(box);
    });
    c.appendChild(grid);
  }

  function renderReport(r, live) {
    $("error").hidden = true;
    var probs = shapeProblems(r);
    if (probs.length) { $("banner").hidden = true; showError(probs); return false; }
    try {
      renderBanner(r, live === true);
      renderMeta(r);
      renderSummary(r);
      renderConflicts(r);
      renderEvidence(r);
      renderReportTech(r);
      $("statement").textContent = str(r.statement);
      current.report = r;
      current.mock = isMockReport(r);
      $("again-btn").hidden = !MODE.live;
      $("report").hidden = false;
      return true;
    } catch (e) {
      $("banner").hidden = true;
      showError(["Rendering failed: " + (e && e.message ? e.message : "unknown error")]);
      return false;
    }
  }

  var fixtureToken = 0;   // only the latest scenario request may render, so a slow earlier load cannot replace a newer choice
  function loadFixture(name) {
    var t = ++fixtureToken;
    return fetch("/api/fixtures/" + encodeURIComponent(name)).then(function (res) {
      if (!res.ok) throw new Error("Server answered " + res.status + " for " + name);
      return res.json();
    }).then(function (r) { return t === fixtureToken ? renderReport(r, false) : false; }).catch(function (e) {
      if (t !== fixtureToken) return;
      $("banner").hidden = true;
      showError(["Could not load scenario " + name + ": " + e.message]);
    });
  }

  /* ---------------- upload and live analysis ----------------
     Needs the server routes /api/mode, /api/analyse and /api/jobs/<id>. If /api/mode is missing or fails, the page behaves as
     scenarios-only. Test hooks via the query string (read once at load):
       ?poll_interval_ms=N   polling interval, default 1000
       ?poll_timeout_ms=N    give up polling after N ms, default 600000 (10 minutes) */
  function qnum(name, dflt) {
    try {
      var v = Number(new URLSearchParams(window.location.search).get(name));
      return (isFinite(v) && v > 0) ? v : dflt;
    } catch (e) { return dflt; }
  }
  var POLL_INTERVAL_MS = qnum("poll_interval_ms", 1000);
  var POLL_TIMEOUT_MS = qnum("poll_timeout_ms", 600000);
  var pollToken = 0;
  var previewUrl = null;
  var seenSteps = [];

  function setLive(kind, text, detail) {
    var s = $("live-status");
    clear(s);
    s.className = "live-status " + kind;
    if (text) s.appendChild(document.createTextNode(text));
    if (detail) s.appendChild(el("span", "st-msg", detail));
    s.hidden = !text;
  }
  function setBusy(busy) {
    $("analyse-btn").disabled = busy;
    $("analyse-file").disabled = busy;
    $("change-btn").disabled = busy;
    $("dropzone").setAttribute("aria-busy", busy ? "true" : "false");
  }
  var FRIENDLY = {
    UNSUPPORTED_MEDIA_TYPE: "This kind of file is not supported.", UNREADABLE_FILE: "The file could not be read; it may be damaged.",
    EMPTY_FILE: "The file is empty.", FILE_TOO_LARGE: "The file is larger than the upload limit.", NO_FILE: "Choose a file first.",
    QUEUE_FULL: "The server is busy with other analyses. Try again shortly.", NETWORK: "The server could not be reached."
  };
  function errText(e) {
    return isObj(e) ? (str(e.code) + ": " + str(e.message)) : "Unknown error";
  }
  function liveFail(prefix, e) {
    setLive("failed", prefix + " " + errText(e), isObj(e) && FRIENDLY[e.code] ? FRIENDLY[e.code] : "");
    if (isObj(e) && isObj(e.assessment) && e.assessment.verdict_label) {      // the same category the module cards use (D-087)
      var lab = el("span", "st-label", e.assessment.verdict_label);
      lab.setAttribute("data-error-verdict", str(e.assessment.verdict));
      $("live-status").insertBefore(lab, $("live-status").firstChild);
    }
    setBusy(false);
    $("steps").hidden = true;
  }
  /* A new live analysis hides the report on screen (a MOCK scenario or an earlier file's result), so a refused, failed or timed-out
     analysis never sits next to a report that belongs to something else. Choosing a scenario renders it again. */
  function hideReport() {
    $("report").hidden = true;
    $("banner").hidden = true;
    $("error").hidden = true;
    current.report = null;
  }
  function readJson(res) {
    return res.json().catch(function () { return null; }).then(function (j) { return { ok: res.ok, status: res.status, j: j }; });
  }
  function renderSteps(p) {
    var ol = $("steps");
    if (p.module_id && seenSteps.indexOf(p.module_id) < 0) seenSteps.push(p.module_id);
    clear(ol);
    seenSteps.forEach(function (id) {
      var running = id === p.module_id;
      ol.appendChild(el("li", running ? "running" : "done", moduleName(id) + (running ? " (running)" : "")));
    });
    ol.hidden = !seenSteps.length;
  }

  function pollJob(jobId, fname, token, started) {
    if (token !== pollToken) return;
    if (Date.now() - started > POLL_TIMEOUT_MS) {
      setLive("failed", "Timed out: no result after " + (Math.round(POLL_TIMEOUT_MS / 6000) / 10) +
        " minutes of waiting. The server may still be working; no result is shown for " + fname + ".");
      setBusy(false);
      $("steps").hidden = true;
      return;
    }
    fetch("/api/jobs/" + encodeURIComponent(jobId)).then(readJson).then(function (x) {
      if (token !== pollToken) return;
      if (!x.ok || !isObj(x.j)) {
        liveFail("Could not read job status.", isObj(x.j) && x.j.error ? x.j.error : { code: "HTTP_" + x.status, message: "Unexpected answer from server." });
        return;
      }
      var j = x.j;
      if (j.state === "done") {
        if (!isObj(j.report)) { liveFail("Analysis ended without a report.", { code: "NO_REPORT", message: "The job finished but returned no report." }); return; }
        setBusy(false);
        $("steps").hidden = true;
        setLive("done", "Done: " + fname);
        $("live-panel").classList.add("compact");
        if (renderReport(j.report, true) && $("meta").scrollIntoView) $("meta").scrollIntoView({ block: "start" });
        return;
      }
      if (j.state === "failed") {
        liveFail("Failed: " + fname + ".", j.error || { code: "UNKNOWN", message: "The job failed without details." });
        return;
      }
      var p = isObj(j.progress) ? j.progress : {};
      if (j.state === "running") {
        setLive("running", "Running: " + (p.module_id ? moduleName(p.module_id) + " " : "") + "(" + str(p.completed) + " of " + str(p.total) +
          " checks done) for " + fname);
        renderSteps(p);
      } else {
        setLive("queued", "Queued: " + fname + " is waiting for its turn.");
      }
      setTimeout(function () { pollJob(jobId, fname, token, started); }, POLL_INTERVAL_MS);
    }).catch(function (e) {
      if (token !== pollToken) return;
      liveFail("Could not read job status.", { code: "NETWORK", message: e && e.message ? e.message : "request failed" });
    });
  }

  function startAnalysis() {
    var inp = $("analyse-file");
    var f = inp.files && inp.files[0];
    if (!f) { setLive("failed", "NO_FILE: choose a file first."); return; }
    var fname = f.name;
    var token = ++pollToken; fixtureToken++;   // a scenario still loading must not replace the live result
    hideReport();
    $("scenario").value = "";
    seenSteps = [];
    $("steps").hidden = true;
    setBusy(true);
    setLive("queued", "Uploading " + fname + " ...");
    var fd = new FormData();
    fd.append("file", f, f.name);
    fetch("/api/analyse", { method: "POST", body: fd }).then(readJson).then(function (x) {
      if (token !== pollToken) return;
      if (!x.ok || !isObj(x.j) || !x.j.job_id) {
        liveFail("Refused.", isObj(x.j) && x.j.error ? x.j.error : { code: "HTTP_" + x.status, message: "The server did not accept the file." });
        return;
      }
      setLive("queued", "Queued: " + fname + " is waiting for its turn.");
      pollJob(str(x.j.job_id), fname, token, Date.now());
    }).catch(function (e) {
      if (token !== pollToken) return;
      liveFail("Upload failed.", { code: "NETWORK", message: e && e.message ? e.message : "request failed" });
    });
  }

  function extOf(name) { var i = name.lastIndexOf("."); return i >= 0 ? name.slice(i).toLowerCase() : ""; }
  function kindOf(name, mime) {
    var a = MODE.accept, e = extOf(name);
    if (!a) return /^(image|video|audio)\//.test(str(mime)) ? str(mime).split("/")[0] : "";   // server did not list types: use the browser's type
    return ["image", "video", "audio"].filter(function (k) { return Array.isArray(a[k]) && a[k].indexOf(e) >= 0; })[0] || "";
  }
  function showFile(f) {
    var dz = $("dropzone"), pv = $("preview");
    if (previewUrl) { try { URL.revokeObjectURL(previewUrl); } catch (e) { /* ignore */ } previewUrl = null; }
    clear(pv);
    if (!f) { dz.setAttribute("data-state", "empty"); $("file-card").hidden = true; return; }
    dz.setAttribute("data-state", "file");
    $("file-card").hidden = false;
    $("file-name").textContent = f.name;
    var kind = kindOf(f.name, f.type);
    var sub = [kind ? kind.charAt(0).toUpperCase() + kind.slice(1) : "File", bytes(f.size)];
    var notes = [];
    if (MODE.accept && !kind) notes.push("This file type is not in the supported list, so the server is expected to refuse it.");
    if (num(MODE.maxBytes) && f.size > MODE.maxBytes) notes.push("The file is larger than the " + bytes(MODE.maxBytes) + " upload limit.");
    $("file-sub").textContent = sub.join(", ") + (notes.length ? ". " + notes.join(" ") : "");
    willRun(kind);
    try {
      if (kind && window.URL && URL.createObjectURL) {
        previewUrl = URL.createObjectURL(f);
        var m = kind === "image" ? el("img") : el(kind === "video" ? "video" : "audio");
        if (kind === "image") m.alt = "Preview of the chosen image";
        else { m.controls = true; m.preload = "metadata"; if (kind === "video") m.muted = true; }
        if (kind === "video") m.addEventListener("loadedmetadata", function () {
          if (isFinite(m.duration)) $("file-sub").textContent = sub.concat([fmt(m.duration, 1) + " s"]).join(", ") + (notes.length ? ". " + notes.join(" ") : "");
        });
        m.src = previewUrl;
        pv.appendChild(m);
        pv.setAttribute("aria-hidden", kind === "image" ? "true" : "false");
      } else pv.appendChild(el("span", null, "No preview"));
    } catch (e) { pv.appendChild(el("span", null, "No preview")); }
  }
  function moduleIds() { return Array.isArray(VOCAB.module_order) ? VOCAB.module_order : Object.keys(VOCAB.modules || {}); }
  function checksFor(kind) {
    return moduleIds().filter(function (k) {
      var v = VOCAB.modules[k];
      return isObj(v) && Array.isArray(v.applies_to) && v.applies_to.indexOf(kind) >= 0;
    });
  }
  function willRun(kind) {
    var box = $("will-run"); clear(box);
    var ids = kind ? checksFor(kind) : [];
    box.hidden = !ids.length;
    if (!ids.length) return;
    box.appendChild(el("p", "will-h", "When you press Analyse, these checks will try to run on this " + (kind === "audio" ? "audio file" : kind) + ":"));
    var ul = el("ul");
    ids.forEach(function (k) { ul.appendChild(el("li", null, moduleName(k))); });
    box.appendChild(ul);
    box.appendChild(el("p", "will-note", "A check that needs something this file does not have (for example a face or a sound track) will say that it did not apply."));
  }
  function renderWhichChecks() {
    var host = $("which-checks-body"); if (!host) return;
    clear(host);
    var ids = moduleIds().filter(function (k) { return isObj(VOCAB.modules[k]) && Array.isArray(VOCAB.modules[k].applies_to); });
    $("which-checks").hidden = !ids.length;
    if (!ids.length) return;
    var tbl = el("table", "which-t");
    var tr = el("tr");
    ["Check", "What it looks at", "Runs on"].forEach(function (c) { var x = el("th", null, c); x.scope = "col"; tr.appendChild(x); });
    var thead = el("thead"); thead.appendChild(tr); tbl.appendChild(thead);
    var tb = el("tbody");
    ids.forEach(function (k) {
      var v = VOCAB.modules[k], row = el("tr");
      row.appendChild(el("td", null, v.title));
      row.appendChild(el("td", null, v.what_it_checks));
      row.appendChild(el("td", null, v.needs));
      tb.appendChild(row);
    });
    tbl.appendChild(tb);
    host.appendChild(tbl);
  }
  function onFileChosen() {
    var inp = $("analyse-file");
    var f = inp.files && inp.files[0];
    showFile(f || null);
    if (f) setLive("", "");
  }
  function resetUpload() {
    pollToken++; fixtureToken++;
    hideReport();
    $("live-panel").classList.remove("compact");
    $("steps").hidden = true;
    setLive("", "");
    try { $("analyse-file").value = ""; } catch (e) { /* ignore */ }
    showFile(null);
    setBusy(!MODE.live);
    window.scrollTo(0, 0);
  }
  function acceptText(a, maxBytes) {
    function up(xs) { return xs.map(function (x) { return x.replace(".", "").toUpperCase(); }).join(", "); }
    return "Images: " + up(a.image) + ". Video: " + up(a.video) + ". Audio: " + up(a.audio) + "." +
      (num(maxBytes) ? " Up to " + bytes(maxBytes) + "." : "");
  }

  function initLive() {
    var panel = $("live-panel"), inp = $("analyse-file"), dz = $("dropzone");
    $("analyse-btn").addEventListener("click", startAnalysis);
    $("change-btn").addEventListener("click", function () { inp.click(); });
    $("again-btn").addEventListener("click", resetUpload);
    $("again-top").addEventListener("click", resetUpload);
    $("sources-toggle").addEventListener("change", function () { $("report").classList.toggle("show-sources", this.checked); });
    inp.addEventListener("change", onFileChosen);
    dz.querySelector(".dz-empty").addEventListener("click", function (e) { if (e.target.tagName !== "LABEL" && !inp.disabled) inp.click(); });
    ["dragenter", "dragover"].forEach(function (t) {
      dz.addEventListener(t, function (e) { if (inp.disabled) return; e.preventDefault(); dz.classList.add("drag"); });
    });
    ["dragleave", "drop"].forEach(function (t) { dz.addEventListener(t, function () { dz.classList.remove("drag"); }); });
    dz.addEventListener("drop", function (e) {
      if (inp.disabled || !e.dataTransfer || !e.dataTransfer.files.length) return;
      e.preventDefault();
      try { inp.files = e.dataTransfer.files; } catch (err) { /* older browsers: use the file picker */ return; }
      onFileChosen();
    });
    var pill = $("mode-pill");
    var unavailable = function () {
      MODE.live = false;
      inp.disabled = true;
      $("analyse-btn").disabled = true;
      $("live-unavailable").hidden = false;
      panel.setAttribute("data-live", "false");
      pill.className = "pill mockonly"; pill.textContent = "Demo data only"; pill.hidden = false;
      return false;
    };
    return fetch("/api/mode").then(function (res) {
      if (!res.ok) throw new Error("mode " + res.status);
      return res.json();
    }).then(function (m) {
      if (isObj(m) && isObj(m.accept)) {
        MODE.accept = m.accept; MODE.maxBytes = m.max_bytes;
        $("accept-text").textContent = acceptText(m.accept, m.max_bytes);
        inp.setAttribute("accept", [].concat(m.accept.image || [], m.accept.video || [], m.accept.audio || []).join(","));
      }
      if (isObj(m) && m.live === true) {
        MODE.live = true;
        panel.setAttribute("data-live", "true");
        $("live-unavailable").hidden = true;
        inp.disabled = false;
        $("analyse-btn").disabled = false;
        pill.className = "pill live"; pill.textContent = "Live analysis"; pill.hidden = false;
        return true;
      }
      return unavailable();
    }).catch(unavailable);
  }

  function initActions() {
    $("download-btn").addEventListener("click", function () {
      if (!current.report) return;
      var blob = new Blob([JSON.stringify(current.report, null, 1)], { type: "application/json" });
      var a = el("a");
      a.href = URL.createObjectURL(blob);
      a.download = (current.mock ? "MOCK_" : "") + "report_" + str(current.report.request_id).replace(/[^\w-]/g, "_") + ".json";
      document.body.appendChild(a); a.click(); a.remove();
      setTimeout(function () { URL.revokeObjectURL(a.href); }, 1000);
    });
    $("print-btn").addEventListener("click", function () { window.print(); });
    window.addEventListener("beforeprint", function () {
      Array.prototype.forEach.call(document.querySelectorAll("#report details"), function (d) { d.setAttribute("data-was", d.open ? "1" : "0"); d.open = true; });
    });
    window.addEventListener("afterprint", function () {
      Array.prototype.forEach.call(document.querySelectorAll("#report details[data-was]"), function (d) { d.open = d.getAttribute("data-was") === "1"; });
    });
  }


  /* ---------------- top navigation: two views, "Analyse a file" and "Evaluation Lab" ---------------- */
  var VIEW = "analyse";
  function setView(name, push) {
    VIEW = name === "lab" ? "lab" : "analyse";
    $("view-analyse").hidden = VIEW !== "analyse";
    $("view-lab").hidden = VIEW !== "lab";
    document.body.classList.toggle("view-lab", VIEW === "lab");
    if ($("mode-pill")) $("mode-pill").classList.toggle("lab-hidden", VIEW === "lab");   // the mode pill describes analysis data; the lab reads recorded scores
    $("nav-analyse").setAttribute("aria-current", VIEW === "analyse" ? "page" : "false");
    $("nav-lab").setAttribute("aria-current", VIEW === "lab" ? "page" : "false");
    if ($("nav-analyse").getAttribute("aria-current") === "false") $("nav-analyse").removeAttribute("aria-current");
    if ($("nav-lab").getAttribute("aria-current") === "false") $("nav-lab").removeAttribute("aria-current");
    document.title = VIEW === "lab" ? "Evaluation Lab | Media Analysis" : "Media Analysis";
    if (push) {
      try { history.replaceState(null, "", VIEW === "lab" ? "#lab" : window.location.pathname + window.location.search); } catch (e) { /* ignore */ }
    }
    if (VIEW === "lab") {
      if (window.evallab && typeof window.evallab.open === "function") window.evallab.open($("lab-root"));
      else { clear($("lab-root")); $("lab-root").appendChild(el("p", "error-box", "The Evaluation Lab script is not available on this server.")); }
    }
    return VIEW;
  }
  function initNav() {
    $("nav-analyse").addEventListener("click", function () { setView("analyse", true); window.scrollTo(0, 0); });
    $("nav-lab").addEventListener("click", function () { setView("lab", true); window.scrollTo(0, 0); });
    window.addEventListener("hashchange", function () {
      if (window.location.hash === "#lab" && VIEW !== "lab") setView("lab", false);
    });
    if (window.location.hash === "#lab") setView("lab", false);
  }
  // Scroll offset = the real height of the sticky header (it wraps to two rows at some widths and is not sticky on
  // narrow screens), so a scrolled-to section such as the report overview is never hidden under it.
  function syncHeaderOffset() {
    var top = document.querySelector("header.top");
    if (!top || !window.getComputedStyle) return;
    var pinned = window.getComputedStyle(top).position === "sticky";
    document.documentElement.style.setProperty("--head-h", (pinned ? Math.ceil(top.getBoundingClientRect().height) : 0) + 12 + "px");
  }
  function initHeaderOffset() {
    syncHeaderOffset();
    window.addEventListener("resize", syncHeaderOffset);
    if (window.ResizeObserver && document.querySelector("header.top")) new ResizeObserver(syncHeaderOffset).observe(document.querySelector("header.top"));
  }

  function init() {
    initActions();
    initNav();
    initHeaderOffset();
    var vocab = fetch("/api/vocabulary").then(function (r) { return r.ok ? r.json() : null; }).then(function (v) {
      if (isObj(v)) VOCAB = v;
      renderWhichChecks();
    }).catch(function () { /* labels fall back to the explanation text and raw codes */ });
    var mode = initLive();
    var sel = $("scenario");
    Promise.all([vocab, mode]).then(function (res) {
      var live = res[1] === true;
      return fetch("/api/fixtures").then(function (r) { return r.json(); }).then(function (names) {
        names.forEach(function (n) {
          var o = document.createElement("option");
          o.value = n; o.textContent = "MOCK: " + n.replace(/_/g, " ");
          sel.appendChild(o);
        });
        sel.addEventListener("change", function () {
          if (sel.value) { pollToken++; setBusy(!MODE.live); $("steps").hidden = true; setLive("", ""); loadFixture(sel.value); }
          else hideReport();
        });
        /* Live mode opens on the upload panel (D14): no invented report is shown until someone asks for a scenario.
           A scenarios-only server has nothing else to show, so it opens the first scenario under the MOCK banner. */
        if (!live && names.length) { sel.value = names[0]; loadFixture(names[0]); }
        else if (!names.length && !live) showError(["No scenarios available."]);
      });
    }).catch(function (e) { showError(["Could not list scenarios: " + e.message]); });
  }

  window.dashboard = { renderReport: renderReport, shapeProblems: shapeProblems, config: { pollIntervalMs: POLL_INTERVAL_MS, pollTimeoutMs: POLL_TIMEOUT_MS },
                       loadFixture: loadFixture, setView: setView, view: function () { return VIEW; } };
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init);
  else init();
})();
