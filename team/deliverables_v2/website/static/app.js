"use strict";
/* Media Analysis, front end v2 (deliverables_v2/website). Plain JS, no build step, no network except this server's own API.
   The backend, the report schema and every API route are the unchanged ones of src/dfd/platform. This file only arranges what the
   API returns. Rules kept from the original page:
   - every string from a report is written with textContent, never with markup insertion;
   - wording about evidence comes from the server (module "explanation", "assessment" and report "summary"), not from this file;
   - no report-level verdict, no overall or fused score, no probability: a number is a "model output" or a "ranking score" and is never
     turned into a percentage or combined with another number;
   - the "Checks by outcome" chart counts checks per outcome group of THIS report. It does not use any score. */
(function () {
  var VOCAB = { modules: {}, findings: {}, statuses: {}, groups: [] };
  var OBS_KIND = { file_metadata: "File metadata", heuristic: "Heuristic flag (descriptive only)",
                   marker_scan: "Marker scan (not a verification)", verification: "Verification result",
                   processing: "Processing record" };
  var WHY_PARTS = [      /* fallback only: the server sends the same list as VOCAB.explanation_parts */
    { id: "mock", label: "Scenario data", kinds: ["mock"] }, { id: "processing", label: "What we did", kinds: ["processing"] },
    { id: "examined", label: "What we examined", kinds: ["examined"] },
    { id: "observed", label: "What we observed", kinds: ["observation", "model_output"] },
    { id: "decision", label: "Why this was reported", kinds: ["decision"] }, { id: "interpretation", label: "What it means", kinds: ["interpretation"] },
    { id: "not_established", label: "What it does not establish", kinds: ["not_established"] }, { id: "limitation", label: "Limitations", kinds: ["limitation"] }
  ];
  function whyParts() { return Array.isArray(VOCAB.explanation_parts) && VOCAB.explanation_parts.length ? VOCAB.explanation_parts : WHY_PARTS; }
  var GROUP_ORDER = ["attention", "nothing_found", "credential", "inconclusive", "informational", "experimental", "not_applicable", "unsupported", "did_not_run"];
  var GROUP_ABOUT = {
    attention: "A check flagged something at its fixed decision line, inside its tested scope only. This is a possible sign, not a determination.",
    nothing_found: "The check ran and found nothing it looks for. It does not show that the file is unmanipulated.",
    credential: "A signed content credential is present and verified. This concerns where a file came from, not whether it is true.",
    inconclusive: "The check ran but reached no conclusion.",
    informational: "Facts about the file. Not evidence either way.",
    experimental: "A research prototype. It makes no finding and no decision is taken from it.",
    not_applicable: "The check does not apply to this file, for example because it has no face or no sound track.",
    unsupported: "A model or file the check needs is not available on this computer.",
    did_not_run: "The check started but could not complete."
  };
  var LIVE_TEXT = "Results come from the checks listed below; each states its own tested scope and limitations.";
  var SVGNS = "http://www.w3.org/2000/svg";
  var MODE = { live: false, accept: null, maxBytes: null };
  var current = { report: null, mock: false };
  var TYPE = "auto";           /* client-side choice only: filters the file picker and sets expectations; never sent to the server */

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

  /* ---------------- simple view (D-097) ----------------
     The page shows only these checks, together in one section, in this order. Every other check still runs on the server and stays
     in the downloaded JSON and in "Technical details for the whole report"; nothing is removed. Set SIMPLE_VIEW to false to show
     every check again, grouped by outcome as before. */
  var SIMPLE_VIEW = true;
  var SHOWN_MODULES = ["face_video", "speech", "media_metadata", "generated_video"];
  function isShown(id) { return !SIMPLE_VIEW || SHOWN_MODULES.indexOf(id) >= 0; }
  function shownModules(r) {
    var ms = (Array.isArray(r.modules) ? r.modules : []).filter(function (m) { return isShown(m.module_id); });
    if (SIMPLE_VIEW) ms.sort(function (a, b) { return SHOWN_MODULES.indexOf(a.module_id) - SHOWN_MODULES.indexOf(b.module_id); });
    return ms;
  }
  function cap(s) { return s ? s.charAt(0).toUpperCase() + s.slice(1) : s; }

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

  /* ---------------- stages of the Analyse view ---------------- */
  function showStage(name) {
    $("stage-upload").hidden = name !== "upload";
    $("stage-progress").hidden = name !== "progress";
    if (name !== "results") { $("report").hidden = true; $("banner").hidden = true; $("error").hidden = true; }
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
  function stateText(st) {
    if (st === "ok") return "Complete";
    if (st === "partial") return "Partial: some checks did not complete";
    if (st === "failed") return "Report failed: no check completed";
    return str(st);
  }
  function factChips(rows) {
    var ul = el("ul", "facts");
    rows.forEach(function (kv) {
      if (kv[1] === "" || kv[1] === null || kv[1] === undefined) return;
      var li = el("li");
      li.appendChild(el("span", "fact-k", kv[0]));
      li.appendChild(el("span", "fact-v", kv[1]));
      ul.appendChild(li);
    });
    return ul;
  }
  /* The uploaded file, played from the browser's local copy (nothing is fetched again). Shown only when the report is for the
     file chosen in this page, so a demo scenario never shows another file. */
  function reportPreview(filename) {
    if (!previewUrl || !previewKind || str(filename) !== previewName) return null;
    var box = el("div", "ov-preview");
    var m = previewKind === "image" ? el("img") : el(previewKind === "video" ? "video" : "audio");
    if (previewKind === "image") m.alt = "The analysed image";
    else { m.controls = true; m.preload = "metadata"; m.setAttribute("aria-label", "The analysed " + (previewKind === "video" ? "video" : "audio")); }
    m.src = previewUrl;
    box.appendChild(m);
    return box;
  }
  function renderMeta(r) {
    var s = $("meta"); clear(s);
    var inp = r.input, p = r.probe, st = r.status;
    var head = el("div", "ov-head");
    var left = el("div", "ov-title");
    left.appendChild(el("p", "kicker", "File analysed"));
    var h = el("h2", "ov-file", str(inp.filename));
    h.id = "overview-h";
    left.appendChild(h);
    head.appendChild(left);
    var pm = reportPreview(inp.filename);
    if (pm) head.appendChild(pm);
    s.appendChild(head);
    var kind = { video: "Video", image: "Image", audio: "Audio" }[p.media_type] || str(p.media_type);
    var streams = p.media_type === "video" ? (p.has_audio === true ? "Picture and sound" : (p.has_audio === false ? "Picture only, no sound" : "")) : "";
    s.appendChild(factChips([
      ["Type", kind], ["Duration", num(p.duration_s) ? fmt(p.duration_s, 1) + " s" : ""],
      ["Dimensions", (p.width && p.height) ? p.width + " x " + p.height + " px" : ""], ["Streams", streams],
      ["Size", bytes(inp.size_bytes)], ["Analysis", stateText(st.state)]
    ]));
    var cov = r.coverage, cnt = function (k) { return Array.isArray(cov[k]) ? cov[k].filter(function (i) { return isShown(isObj(i) ? i.module_id : ""); }).length : 0; };
    if (st.state === "failed") {
      s.appendChild(el("p", "state-note failed", "No check completed, so nothing was analysed. This is not a statement about the file."));
    } else if (st.state === "partial") {
      var n = cnt("failed") + cnt("timed_out") + cnt("unsupported");
      var note = el("p", "state-note partial", n + " of " + shownModules(r).length + " checks did not complete (" + cnt("failed") + " failed, " +
        cnt("timed_out") + " timed out, " + cnt("unsupported") + " not available). The results below cover the other checks only.");
      note.id = "partial-note";
      s.appendChild(note);
    }
    var det = el("details", "plain");
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

  /* ---------------- B. the group counts and the "Checks by outcome" chart ---------------- */
  function groupOf(m) {
    if (isObj(m.explanation) && m.explanation.group) return m.explanation.group;
    return m.status === "completed" ? "inconclusive" : ({ not_applicable: "not_applicable", unsupported: "unsupported" }[m.status] || "did_not_run");
  }
  /* The same grouping the original page uses: the server's summary.groups when present, otherwise the group of each module result. */
  function groupsOf(r) {
    if (isObj(r.summary) && Array.isArray(r.summary.groups)) {
      return r.summary.groups.map(function (g) {
        return { id: g.id, label: g.label, modules: (Array.isArray(g.modules) ? g.modules : []).filter(isShown) };
      }).filter(function (g) { return g.modules.length; });
    }
    var by = {};
    shownModules(r).forEach(function (m) { var g = groupOf(m); (by[g] = by[g] || []).push(m.module_id); });
    var label = { inconclusive: "Results", not_applicable: "Did not apply to this file", unsupported: "Not available", did_not_run: "Did not complete" };
    return GROUP_ORDER.filter(function (g) { return by[g]; }).map(function (g) { return { id: g, label: label[g] || g, modules: by[g] }; });
  }
  function outcomeCounts(r) {
    return groupsOf(r).map(function (g) { return { id: g.id, label: g.label, n: g.modules.length }; }).filter(function (g) { return g.n > 0; });
  }
  function patternDefs() {
    var defs = svg("defs");
    function pat(id, extra) {
      var p = svg("pattern", { id: "pat-" + id, width: 6, height: 6, patternUnits: "userSpaceOnUse", patternTransform: "rotate(45)" });
      p.appendChild(svg("rect", { width: 6, height: 6, "class": "pbase d-" + id }));
      extra(p);
      defs.appendChild(p);
    }
    pat("experimental", function (p) { p.appendChild(svg("rect", { width: 2, height: 6, "class": "pmark" })); });
    pat("not_applicable", function (p) { p.appendChild(svg("circle", { cx: 3, cy: 3, r: 1, "class": "pmark" })); });
    pat("unsupported", function (p) { p.appendChild(svg("rect", { width: 6, height: 1.6, y: 2, "class": "pmark" })); });
    return defs;
  }
  var PATTERNED = { experimental: true, not_applicable: true, unsupported: true };
  function fillFor(id) { return PATTERNED[id] ? "url(#pat-" + id + ")" : ""; }
  function swatch(id) {
    var s = svg("svg", { viewBox: "0 0 14 14", width: 14, height: 14, "class": "dsw", "aria-hidden": "true", focusable: "false" });
    var r = svg("rect", { width: 14, height: 14, rx: 3, "class": "dsw-rect d-" + id });
    if (PATTERNED[id]) r.setAttribute("style", "fill:" + fillFor(id));
    s.appendChild(r);
    return s;
  }
  function donut(counts) {
    var total = counts.reduce(function (a, c) { return a + c.n; }, 0);
    var R = 44, C = 2 * Math.PI * R, SW = 22, GAP = counts.length > 1 ? 2 : 0;
    var desc = total + (total === 1 ? " check" : " checks") + " by outcome: " + counts.map(function (c) { return c.n + " " + c.label.toLowerCase(); }).join(", ") + ".";
    var s = svg("svg", { viewBox: "0 0 120 120", "class": "donut", role: "img", "aria-label": desc });
    s.appendChild(patternDefs());
    s.appendChild(svg("circle", { cx: 60, cy: 60, r: R, fill: "none", "class": "donut-track", "stroke-width": SW }));
    var done = 0;
    counts.forEach(function (c) {
      var len = c.n / total * C;
      var seg = svg("circle", { cx: 60, cy: 60, r: R, fill: "none", "stroke-width": SW, "class": "dseg d-" + c.id,
        "stroke-dasharray": Math.max(len - GAP, 0.01).toFixed(3) + " " + (C - Math.max(len - GAP, 0.01)).toFixed(3),
        "stroke-dashoffset": (-done).toFixed(3), transform: "rotate(-90 60 60)" });
      if (PATTERNED[c.id]) seg.setAttribute("style", "stroke:" + fillFor(c.id));
      seg.setAttribute("data-group", c.id);
      s.appendChild(seg);
      done += len;
    });
    var t1 = svg("text", { x: 60, y: 62, "text-anchor": "middle", "class": "donut-n" }); t1.textContent = String(total); s.appendChild(t1);
    var t2 = svg("text", { x: 60, y: 77, "text-anchor": "middle", "class": "donut-u" }); t2.textContent = total === 1 ? "check" : "checks"; s.appendChild(t2);
    return s;
  }
  function renderOutcome(r) {
    var s = $("outcome"); clear(s);
    var counts = outcomeCounts(r);
    var h = el("h2", "h3", "Checks by outcome");
    h.id = "outcome-h";
    s.appendChild(h);
    if (!counts.length) { s.appendChild(el("p", "empty", "No checks are listed in this report.")); return; }
    var fig = el("figure", "outcome-fig");
    fig.appendChild(donut(counts));
    var ul = el("ul", "legend");
    ul.setAttribute("aria-label", "Number of checks in each outcome group");
    counts.forEach(function (c) {
      var li = el("li", "legend-item");
      li.setAttribute("data-group", c.id);
      var a = el("a", null);
      a.href = "#group-" + c.id;
      a.appendChild(swatch(c.id));
      a.appendChild(el("span", "legend-label", c.label));
      a.appendChild(el("span", "legend-n", c.n));
      li.appendChild(a);
      ul.appendChild(li);
    });
    fig.appendChild(ul);
    s.appendChild(fig);
    s.appendChild(el("p", "outcome-cap", "This chart counts the checks in this report by outcome. It is not a probability, a score or an overall verdict."));
  }

  function renderSummary(r) {
    var s = $("summary"); clear(s);
    var h = el("h2", "h3", "What the checks found");
    h.id = "summary-h";
    s.appendChild(h);
    var sm = r.summary;
    if (!isObj(sm)) {
      s.appendChild(el("p", "empty", "This report was produced in an older format without a written summary. Each check is listed below."));
      return;
    }
    s.appendChild(el("p", "summary-head", SIMPLE_VIEW ? shownHeadline(r) : sm.headline));
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
    if (sm.no_overall_score) s.appendChild(el("p", "no-score", sm.no_overall_score));
  }

  function plural(n, one, many) { return n + " " + (n === 1 ? one : many); }
  /* Mirror of explain.summarise for the checks this page shows. */
  function shownHeadline(r) {
    var c = {};
    shownModules(r).forEach(function (m) { var g = groupOf(m); c[g] = (c[g] || 0) + 1; });
    var n = function (g) { return c[g] || 0; };
    var kind = { video: "video", image: "image", audio: "audio file" }[isObj(r.probe) ? r.probe.media_type : ""] || "file";
    var parts = [];
    if (n("attention")) parts.push(plural(n("attention"), "check found a point that merits attention", "checks found points that merit attention"));
    if (n("nothing_found")) parts.push(plural(n("nothing_found"), "found nothing within its tested scope", "found nothing within their tested scope"));
    if (n("credential")) parts.push(plural(n("credential"), "verified a content credential", "verified content credentials"));
    if (n("inconclusive")) parts.push(plural(n("inconclusive"), "was inconclusive", "were inconclusive"));
    if (n("informational")) parts.push(plural(n("informational"), "described the file", "described the file"));
    if (n("experimental")) parts.push(plural(n("experimental"), "is experimental and makes no finding", "are experimental and make no finding"));
    var ran = shownModules(r).length - n("not_applicable") - n("unsupported") - n("did_not_run");
    var head = plural(ran, "check", "checks") + " ran on this " + kind + ".";
    if (parts.length) head += " " + cap(parts.join("; ")) + ".";
    var tail = [];
    if (n("not_applicable")) tail.push(plural(n("not_applicable"), "did not apply", "did not apply"));
    if (n("unsupported")) tail.push(plural(n("unsupported"), "was not available", "were not available"));
    if (n("did_not_run")) tail.push(plural(n("did_not_run"), "did not complete", "did not complete"));
    if (tail.length) head += " " + cap(tail.join(", ")) + ".";
    return head;
  }

  function renderConflicts(r) {
    var s = $("conflicts"); clear(s);
    var list = isObj(r.summary) && Array.isArray(r.summary.disagreements) ? r.summary.disagreements
      : (Array.isArray(r.conflicts) ? r.conflicts.map(function (c) { return { modules: c.modules, text: c.note }; }) : []);
    list = list.filter(function (x) { return !Array.isArray(x.modules) || x.modules.every(isShown); });   /* only disagreements among the shown checks */
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
    var W = 560, H = 210, L = 44, R = 12, T = 12, B = 38;
    var x0 = xs[0], x1 = xs[xs.length - 1];
    var lo = Math.min.apply(null, ys), hi = Math.max.apply(null, ys);
    if (hi - lo < 1e-9) { hi = lo + 1; }
    var pad = (hi - lo) * 0.08;
    lo -= pad; hi += pad;
    function X(v) { return L + (v - x0) / (x1 - x0) * (W - L - R); }
    function Y(v) { return T + (hi - v) / (hi - lo) * (H - T - B); }
    var bandTxt = Array.isArray(ev.band) ? "; band reported as consistent " + ev.band[0] + " to " + ev.band[1] + " frames" : "";
    var root = svg("svg", { viewBox: "0 0 " + W + " " + H, "class": "curve", role: "img",
      "aria-label": "Embedding distance at each audio-visual offset from " + x0 + " to " + x1 + " frames; lowest at " + ev.best_offset + " frames" + bandTxt + "." });
    if (Array.isArray(ev.band)) {
      root.appendChild(svg("rect", { "class": "band", x: X(ev.band[0] - 0.5), y: T, width: X(ev.band[1] + 0.5) - X(ev.band[0] - 0.5), height: H - T - B }));
      var bt = svg("text", { x: (X(ev.band[0] - 0.5) + X(ev.band[1] + 0.5)) / 2, y: T + 12, "text-anchor": "middle", "class": "band-label" });
      bt.textContent = "band " + ev.band[0] + " to +" + ev.band[1]; root.appendChild(bt);
    }
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
      var bl = svg("text", { x: X(xs[bi]), y: Math.max(T + 26, Y(ys[bi]) - 8), "text-anchor": "middle" });
      bl.textContent = "best " + (ev.best_offset > 0 ? "+" : "") + ev.best_offset; root.appendChild(bl);
    }
    var fig = el("figure", "figure");
    fig.setAttribute("data-evidence-kind", "offset_curve");
    fig.appendChild(root);
    fig.appendChild(el("figcaption", null, ev.text));
    return fig;
  }
  function filmstrip(ev) {
    if (!Array.isArray(ev.frames)) return null;
    var fig = el("figure", "figure");
    fig.setAttribute("data-evidence-kind", "frames");
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

  /* ---------------- score bar: a number from 0 to 1 with the module's decision line. Never a percentage. ---------------- */
  function assessmentOf(m) { return isObj(m.assessment) && str(m.assessment.verdict_label) ? m.assessment : null; }
  function experimental(m) {
    return Array.isArray(m.warnings) && m.warnings.some(function (w) { return isObj(w) && /^EXPERIMENTAL/.test(str(w.code)); });
  }
  function scoreBar(m, a) {
    var raw = isObj(a.raw) ? a.raw : {};
    if (!num(raw.score) || ["model_output", "ranking_score"].indexOf(raw.score_kind) < 0) return null;      /* no bar for a number that is neither */
    var exp = a.tier === "experimental";
    var kind = exp ? "Experimental ranking score, no finding" : "Model output, not a probability";
    var hasLine = num(raw.threshold) && !exp;
    var wrap = el("figure", "sbar" + (exp ? " sbar-exp" : ""));
    wrap.setAttribute("data-score-kind", str(raw.score_kind));
    var label = kind + ": " + fmt(raw.score) + (hasLine ? ". Decision line " + fmt(raw.threshold) + "." : ".") +
      (m.source === "mock" ? " MOCK number, not a measurement." : "");
    var top = el("div", "sbar-top");
    top.appendChild(el("span", "sbar-kind", kind));
    top.appendChild(el("span", "sbar-val", fmt(raw.score)));
    wrap.appendChild(top);
    var track = el("div", "sbar-track");
    track.setAttribute("role", "img");
    track.setAttribute("aria-label", label);
    var fill = el("span", "sbar-fill");
    fill.style.width = Math.max(0, Math.min(1, raw.score)) * 100 + "%";
    track.appendChild(fill);
    if (hasLine) {
      var mark = el("span", "sbar-mark");
      mark.style.left = Math.max(0, Math.min(1, raw.threshold)) * 100 + "%";
      track.appendChild(mark);
    }
    wrap.appendChild(track);
    var scale = el("div", "sbar-scale");
    scale.setAttribute("aria-hidden", "true");
    scale.appendChild(el("span", null, "0"));
    if (hasLine) {
      var tl = el("span", "sbar-line-label", "decision line " + fmt(raw.threshold));
      tl.style.left = Math.max(0.12, Math.min(0.88, raw.threshold)) * 100 + "%";
      scale.appendChild(tl);
    }
    scale.appendChild(el("span", null, "1"));
    wrap.appendChild(scale);
    var cap2 = el("figcaption", "sbar-cap");
    cap2.textContent = exp ? "Shown for inspection only. No decision is made from it and it is not a probability."
      : (m.source === "mock" ? "MOCK number, not a measurement. A bar position is not a chance of being fake." : "A bar position is not a chance of being fake. It is the model's raw output on a 0 to 1 scale.");
    wrap.appendChild(cap2);
    return wrap;
  }

  /* ---------------- cards ---------------- */
  var UNC = { moderate: "Moderate uncertainty", high: "High uncertainty" };
  function pseudoAssessment(m) {      /* reports written before assessments existed still render, with the old chip text as the label */
    var e = isObj(m.explanation) ? m.explanation : null;
    var label = m.status === "completed" ? ((e && e.finding_label) || VOCAB.findings[m.finding] || str(m.finding))
      : ((e && e.status_label) || VOCAB.statuses[m.status] || str(m.status));
    return { verdict: "legacy", verdict_label: label, meaning: "", tier: "none", uncertainty: { level: "not_assessed", text: "" }, raw: { score: null, score_kind: "none", threshold: null },
             calibration: { is_probability: false, statement: null }, evidence: [] };
  }
  function verdictClasses(a) {
    var tier = str(a.tier) || "none", v = str(a.verdict), c = ["assessed", "tier-" + tier];
    if (tier === "experimental") return c;                  /* an experimental card never takes the look of a detector flag */
    if (/^(likely_|possible_)/.test(v)) { c.push("v-flag"); if (/^possible_/.test(v)) c.push("v-weak"); }
    else if (v === "credentials_invalid") c.push("v-bad");
    else if (v === "analysis_failed" || v === "unsupported_input") c.push("v-failed");
    else if (v === "inconclusive") c.push("v-inconclusive");
    else if (v === "credentials_verified") c.push("v-credential");
    return c;
  }
  function isScoreItem(x) { return isObj(x) && Array.isArray(x.sources) && x.sources.some(function (q) { return /^score\./.test(str(q)); }) && x.kind !== "decision"; }
  function modalityText(m) {
    var v = VOCAB.modules[m.module_id];
    return v && Array.isArray(v.applies_to) && v.applies_to.length ? v.applies_to.join(", ") : "";
  }
  function scoreKindLabel(a) {
    var k = isObj(a) && isObj(a.raw) ? a.raw.score_kind : "";
    return k === "ranking_score" ? "Ranking score" : (k === "model_output" ? "Model output" : "Score");
  }
  function disclosure(cls, summaryText, node) {
    var d = el("details", "disc " + cls);
    d.appendChild(el("summary", null, summaryText));
    d.appendChild(node);
    return d;
  }
  function whyBox(m, a) {
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
    var ev = Array.isArray(a.evidence) ? a.evidence.filter(function (x) { return isObj(x) && x.text; }) : [];
    if (ev.length) {
      var es = el("div", "why-part");
      es.setAttribute("data-part", "signals");
      es.appendChild(el("h4", null, "Evidence behind this reading"));
      var eu = el("ul");
      ev.forEach(function (x) { var li = el("li", null, x.text); li.setAttribute("data-evidence", str(x.id)); eu.appendChild(li); });
      es.appendChild(eu);
      box.appendChild(es);
    }
    var u = isObj(a.uncertainty) ? a.uncertainty : null, cal = isObj(a.calibration) ? a.calibration : null;
    if ((u && u.text) || (cal && cal.statement)) {
      var ub = el("div", "why-part");
      ub.setAttribute("data-part", "uncertainty");
      ub.appendChild(el("h4", null, "Uncertainty"));
      if (u && u.text) ub.appendChild(el("p", null, u.text));
      if (cal && cal.statement) ub.appendChild(el("p", null, cal.statement));
      box.appendChild(ub);
    }
    (Array.isArray(m.evidence) ? m.evidence : []).forEach(function (ev2) {
      if (!isObj(ev2) || ev2.kind !== "frames") return;      /* the offset curve is drawn on the card itself */
      var fig = filmstrip(ev2);
      if (fig) { var sec2 = el("div", "why-part"); sec2.appendChild(el("h4", null, "Frames the model analysed")); sec2.appendChild(fig); box.appendChild(sec2); }
    });
    return box;
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
  function techBox(m, a) {
    var box = el("div", "tech");
    if (a.tier === "experimental") {      /* the original page keeps an experimental number out of the card face: the muted bar lives here */
      var eb = scoreBar(m, a);
      if (eb) box.appendChild(eb);
    }
    var dl = el("dl", "kv");
    var s = isObj(m.score) ? m.score : {};
    kvRows(dl, [["Module id", m.module_id, "mono"], ["Version", m.version, "mono"], ["Source", m.source], ["Status", m.status],
      ["Finding code", m.finding, "mono"], [scoreKindLabel(a), isObj(m.score) ? str(s.value) + (m.source === "mock" ? " (MOCK number)" : "") +
        (experimental(m) ? " (experimental, for inspection only; no decision is made from it)" : "") : "No score reported."],
      ["Score kind", isObj(a.raw) && a.raw.score_kind !== "none" ? a.raw.score_kind : ""],
      ["Score meaning", s.meaning], ["Calibrated", isObj(m.score) ? (s.calibrated ? "yes" : "no") : ""],
      ["Is a probability", isObj(a.calibration) ? (a.calibration.is_probability ? "yes" : "no") : ""],
      ["Decision threshold", s.threshold], ["Units analysed", m.units_analysed], ["Time", num(m.timing_ms) ? fmt(m.timing_ms, 0) + " ms" : ""]]);
    if (isObj(m.model)) {
      kvRows(dl, [["Model", m.model.name], ["Model version", m.model.version, "mono"], ["Weights SHA-256", m.model.weights_sha256, "mono"],
        ["Licence", m.model.licence], ["Evaluation", m.model.evaluation]]);
    }
    kvRows(dl, [["Tested scope", m.tested_scope], ["Reason", m.reason],
      ["Explanation template", isObj(m.explanation) ? m.explanation.template_id : "", "mono"]]);
    box.appendChild(dl);
    listBlock(box, "Observations", m.observations, null, function (o) {
      return (isObj(o) ? (OBS_KIND[o.kind] || str(o.kind)) : "") + ": " + str(isObj(o) ? o.text : o) +
        (isObj(o) && o.value !== null && o.value !== undefined && typeof o.value !== "object" ? " (" + str(o.value) + ")" : "");
    });
    listBlock(box, "Warnings", m.warnings, "warn", codeMsg);
    listBlock(box, "Errors", m.errors, "err", codeMsg);
    listBlock(box, "Segments", m.segments, null, function (g) {
      return str(isObj(g) ? g.start_s : "") + " s to " + str(isObj(g) ? g.end_s : "") + " s" + (isObj(g) && g.note ? ": " + str(g.note) : "");
    });
    return box;
  }
  function renderCard(m) {
    var a = assessmentOf(m) || pseudoAssessment(m);
    var e = isObj(m.explanation) ? m.explanation : null;
    var g = groupOf(m);
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
    t.appendChild(el("span", "card-name", title(m)));
    head.appendChild(t);
    var badges = el("div", "badges");
    if (m.source === "mock") badges.appendChild(el("span", "badge mock", "MOCK"));
    if (experimental(m) || a.tier === "experimental") badges.appendChild(el("span", "badge exp", "EXPERIMENTAL"));
    if (e && Array.isArray(e.sentences) && e.sentences.some(function (x) { return (x.sources || []).some(function (q) { return /AUDIO_NOT_SPEECH_LIKE/.test(q); }); })) {
      badges.appendChild(el("span", "badge caveat", "Audio may not be speech"));
    }
    head.appendChild(badges);
    card.appendChild(head);

    var vl = el("h3", "verdict-label");
    vl.appendChild(el("span", "verdict-text", a.verdict_label));
    card.appendChild(vl);

    if (quiet || (m.status !== "completed" && m.reason)) {
      if (m.reason) card.appendChild(el("p", "reason", str(m.reason)));
    }
    if (!quiet) {
      var u = isObj(a.uncertainty) ? a.uncertainty : {};
      if (UNC[u.level]) {
        var chips = el("div", "chips");
        var uc = el("span", "chip unc unc-" + u.level, UNC[u.level]);
        uc.setAttribute("data-uncertainty", u.level);
        chips.appendChild(uc);
        card.appendChild(chips);
      }
      if (a.meaning) card.appendChild(el("p", "meaning", a.meaning));
      if (a.tier === "experimental") {
        card.appendChild(el("p", "exp-signal", "Experimental signal, not a finding. It is shown for inspection only and no decision is made from it."));
      }
      if (e && isObj(e.key_figure) && !isScoreItem(e.key_figure) && a.tier !== "experimental") {
        var k = el("p", "keyfig");
        k.appendChild(el("b", null, e.key_figure.label + ": "));
        k.appendChild(el("span", null, e.key_figure.text));
        card.appendChild(k);
      }
      if (a.tier === "detector" || a.tier === "limited") {
        var bar = scoreBar(m, a);
        if (bar) card.appendChild(bar);
      }
      (Array.isArray(m.evidence) ? m.evidence : []).forEach(function (ev) {
        if (isObj(ev) && ev.kind === "offset_curve") { var fig = offsetCurve(ev); if (fig) card.appendChild(fig); }
      });
    }

    var more = el("div", "more");
    if (!quiet) {
      more.appendChild(disclosure("why-box", "How this was reached", whyBox(m, a)));
      var lim = (Array.isArray(m.limitations) ? m.limitations : []).filter(function (x) { return str(x); });
      if (lim.length) {
        var lb = el("div", "limits-list");
        lb.setAttribute("data-part", "limitations");
        var lu = el("ul");
        lim.forEach(function (x) { lu.appendChild(el("li", null, x)); });
        lb.appendChild(lu);
        more.appendChild(disclosure("lim-box", "Limitations", lb));
      }
    }
    more.appendChild(disclosure("tech-box", "Technical details", techBox(m, a)));
    card.appendChild(more);
    return card;
  }

  /* ---------------- C. results grouped by meaning ---------------- */
  function renderEvidence(r) {
    var host = $("evidence"); clear(host);
    if (SIMPLE_VIEW) {
      var ms = shownModules(r);
      var one = el("section", "group g-all wide");
      one.id = "group-all";
      var oh = el("h2", "group-h");
      oh.appendChild(el("span", null, "Results of the checks"));
      oh.appendChild(el("span", "count", "(" + ms.length + ")"));
      one.appendChild(oh);
      var oc = el("div", "cards");
      ms.forEach(function (m) { oc.appendChild(renderCard(m)); });
      one.appendChild(oc);
      if (ms.length) host.appendChild(one);
      else host.appendChild(el("p", "empty", "No checks are listed in this report."));
      return;
    }
    var byId = {};
    r.modules.forEach(function (m) { byId[m.module_id] = m; });
    groupsOf(r).forEach(function (g) {
      var sec = el("section", "group g-" + g.id);
      sec.id = "group-" + g.id;
      if (g.modules.length > 1 || g.id === "not_applicable") sec.className += " wide";      /* single-card groups sit two to a row */
      var h = el("h2", "group-h");
      h.appendChild(icon(g.id));
      h.appendChild(el("span", null, g.label));
      h.appendChild(el("span", "count", "(" + g.modules.length + ")"));
      var cards = el("div", "cards");
      g.modules.forEach(function (id) { if (byId[id]) cards.appendChild(renderCard(byId[id])); });
      if (g.id === "not_applicable") {      /* checks that did not apply are listed but folded away: fewer cards on screen */
        var d = el("details", "group-fold");
        var sm = el("summary");
        sm.appendChild(icon(g.id));
        sm.appendChild(el("span", null, g.label));
        sm.appendChild(el("span", "count", "(" + g.modules.length + ")"));
        d.appendChild(sm);
        d.appendChild(cards);
        sec.appendChild(d);
      } else {
        sec.appendChild(h);
        sec.appendChild(cards);
      }
      host.appendChild(sec);
    });
    if (!r.modules.length) host.appendChild(el("p", "empty", "No checks are listed in this report."));
  }

  /* ---------------- F. limits and report-level technical details ---------------- */
  function renderLimits(r) {
    $("statement").textContent = str(r.statement);
    fillLimitsExtra($("limits-extra"));
  }
  function fillLimitsExtra(ul) {
    clear(ul);
    var a = isObj(VOCAB.assessment) ? VOCAB.assessment : {};
    [a.not_established, a.probability_statement].forEach(function (t) { if (t) ul.appendChild(el("li", null, t)); });
  }
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
      box.appendChild(el("h4", null, (VOCAB.statuses[st] || st) + " (" + items.length + ")"));
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
      showStage("results");
      renderBanner(r, live === true);
      renderMeta(r);
      renderOutcome(r);
      renderSummary(r);
      renderConflicts(r);
      renderEvidence(r);
      renderLimits(r);
      renderReportTech(r);
      current.report = r;
      current.mock = isMockReport(r);
      $("report").hidden = false;
      return true;
    } catch (e) {
      $("banner").hidden = true;
      showError(["Rendering failed: " + (e && e.message ? e.message : "unknown error")]);
      return false;
    }
  }

  var fixtureToken = 0;   /* only the latest scenario request may render, so a slow earlier load cannot replace a newer choice */
  function loadFixture(name) {
    var t = ++fixtureToken;
    return fetch("/api/fixtures/" + encodeURIComponent(name)).then(function (res) {
      if (!res.ok) throw new Error("Server answered " + res.status + " for " + name);
      return res.json();
    }).then(function (r) { return t === fixtureToken ? renderReport(r, false) : false; }).catch(function (e) {
      if (t !== fixtureToken) return;
      showStage("results");
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
  var previewUrl = null, previewKind = "", previewName = "";   /* kept so the report can show the analysed file again */
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
    $("analyse-btn").disabled = busy || mismatchActive();
    $("analyse-file").disabled = busy || !MODE.live;
    $("change-btn").disabled = busy;
    $("dropzone").setAttribute("aria-busy", busy ? "true" : "false");
  }
  var FRIENDLY = {
    UNSUPPORTED_MEDIA_TYPE: "This kind of file is not supported.", UNREADABLE_FILE: "The file could not be read; it may be damaged.",
    EMPTY_FILE: "The file is empty.", FILE_TOO_LARGE: "The file is larger than the upload limit.", NO_FILE: "Choose a file first.",
    QUEUE_FULL: "The server is busy with other analyses. Try again shortly.", NETWORK: "The server could not be reached."
  };
  function errText(e) { return isObj(e) ? (str(e.code) + ": " + str(e.message)) : "Unknown error"; }

  /* progress stage */
  function resetProgress(fname) {
    $("progress-h").textContent = fname;
    $("progress-status").textContent = "";
    $("progress-status").className = "progress-status";
    $("progress-meter").hidden = true;
    $("steps").hidden = true;
    clear($("steps"));
    $("progress-error").hidden = true;
    clear($("progress-error"));
    $("progress-back").hidden = true;
  }
  function setProgressStatus(kind, text, detail) {
    var s = $("progress-status");
    clear(s);
    s.className = "progress-status " + kind;
    s.appendChild(el("span", "ps-main", text));
    if (detail) s.appendChild(el("span", "ps-detail", detail));
  }
  function setMeter(done, total) {
    var m = $("progress-meter");
    if (!num(total) || total <= 0) { m.hidden = true; return; }
    m.hidden = false;
    m.setAttribute("aria-valuemax", String(total));
    m.setAttribute("aria-valuenow", String(done));
    m.setAttribute("aria-valuetext", done + " of " + total + " checks done");
    m.firstChild.style.width = Math.max(0, Math.min(1, done / total)) * 100 + "%";
  }
  function liveFail(prefix, e) {
    setBusy(false);
    $("progress-meter").hidden = true;
    $("steps").hidden = true;
    setProgressStatus("failed", "The analysis did not finish");
    var box = $("progress-error");
    clear(box);
    if (isObj(e) && isObj(e.assessment) && e.assessment.verdict_label) {      /* the same category the module cards use */
      var lab = el("p", "err-label", e.assessment.verdict_label);
      lab.setAttribute("data-error-verdict", str(e.assessment.verdict));
      box.appendChild(lab);
    }
    box.appendChild(el("p", "err-main", (prefix ? prefix + " " : "") + errText(e)));
    if (isObj(e) && FRIENDLY[e.code]) box.appendChild(el("p", "err-friendly", FRIENDLY[e.code]));
    box.hidden = false;
    $("progress-back").hidden = false;
    $("progress-back").focus();
  }
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
    if (p.module_id && isShown(p.module_id) && seenSteps.indexOf(p.module_id) < 0) seenSteps.push(p.module_id);
    clear(ol);
    seenSteps.forEach(function (id) {
      var running = id === p.module_id;
      var li = el("li", running ? "running" : "done");
      li.appendChild(el("span", "step-dot"));
      li.appendChild(el("span", null, moduleName(id)));
      li.appendChild(el("span", "step-state", running ? "running" : "done"));
      ol.appendChild(li);
    });
    ol.hidden = !seenSteps.length;
  }

  function pollJob(jobId, fname, token, started) {
    if (token !== pollToken) return;
    if (Date.now() - started > POLL_TIMEOUT_MS) {
      liveFail("Timed out.", { code: "TIMEOUT", message: "No result after " + (Math.round(POLL_TIMEOUT_MS / 6000) / 10) +
        " minutes of waiting. The server may still be working; no result is shown for " + fname + "." });
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
        setLive("done", "Done: " + fname);
        if (renderReport(j.report, true)) {
          var h = $("overview-h"); if (h) { h.setAttribute("tabindex", "-1"); }
          window.scrollTo(0, 0);
        }
        return;
      }
      if (j.state === "failed") {
        liveFail("Failed.", j.error || { code: "UNKNOWN", message: "The job failed without details." });
        return;
      }
      var p = isObj(j.progress) ? j.progress : {};
      if (j.state === "running") {
        setProgressStatus("running", p.module_id && isShown(p.module_id) ? "Running: " + moduleName(p.module_id) : (p.module_id ? "Running the checks" : "Starting the checks"),
          num(p.total) && p.total > 0 ? str(p.completed) + " of " + str(p.total) + " checks done" : "");
        setMeter(num(p.completed) ? p.completed : 0, p.total);
        renderSteps(p);
      } else {
        setProgressStatus("queued", "Queued", "The file is waiting for its turn.");
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
    var token = ++pollToken; fixtureToken++;   /* a scenario still loading must not replace the live result */
    hideReport();
    $("scenario").value = "";
    seenSteps = [];
    resetProgress(fname);
    showStage("progress");
    setBusy(true);
    setProgressStatus("queued", "Uploading", fname);
    var fd = new FormData();
    fd.append("file", f, f.name);      /* the request is the file and nothing else: the content-type choice never leaves the browser */
    fetch("/api/analyse", { method: "POST", body: fd }).then(readJson).then(function (x) {
      if (token !== pollToken) return;
      if (!x.ok || !isObj(x.j) || !x.j.job_id) {
        liveFail("Refused.", isObj(x.j) && x.j.error ? x.j.error : { code: "HTTP_" + x.status, message: "The server did not accept the file." });
        return;
      }
      setProgressStatus("queued", "Queued", "The file is waiting for its turn.");
      pollJob(str(x.j.job_id), fname, token, Date.now());
    }).catch(function (e) {
      if (token !== pollToken) return;
      liveFail("Upload failed.", { code: "NETWORK", message: e && e.message ? e.message : "request failed" });
    });
  }

  /* ---------------- content-type choice (client side only) and file handling ---------------- */
  function extOf(name) { var i = name.lastIndexOf("."); return i >= 0 ? name.slice(i).toLowerCase() : ""; }
  function kindOf(name, mime) {
    var a = MODE.accept, e = extOf(name);
    if (!a) return /^(image|video|audio)\//.test(str(mime)) ? str(mime).split("/")[0] : "";
    return ["image", "video", "audio"].filter(function (k) { return Array.isArray(a[k]) && a[k].indexOf(e) >= 0; })[0] || "";
  }
  function kindWord(k) { return k === "audio" ? "audio file" : k; }
  function moduleIds() { return (Array.isArray(VOCAB.module_order) ? VOCAB.module_order : Object.keys(VOCAB.modules || {})).filter(isShown); }
  function checksFor(kind) {
    return moduleIds().filter(function (k) {
      var v = VOCAB.modules[k];
      return isObj(v) && Array.isArray(v.applies_to) && v.applies_to.indexOf(kind) >= 0;
    });
  }
  function acceptList(kind) {
    var a = MODE.accept; if (!a) return [];
    return kind === "auto" ? [].concat(a.image || [], a.video || [], a.audio || []) : (a[kind] || []);
  }
  function up(xs) { return xs.map(function (x) { return x.replace(".", "").toUpperCase(); }).join(", "); }
  function updateTypeUi() {
    var a = MODE.accept, inp = $("analyse-file");
    if (a) {
      inp.setAttribute("accept", acceptList(TYPE).join(","));
      var t = TYPE === "auto"
        ? "Images: " + up(a.image) + ". Video: " + up(a.video) + ". Audio: " + up(a.audio) + "."
        : cap(TYPE) + " files: " + up(a[TYPE] || []) + ".";
      $("accept-text").textContent = t + (num(MODE.maxBytes) ? " Up to " + bytes(MODE.maxBytes) + " per file." : "");
    }
    var note = $("type-note"); clear(note);
    if (TYPE === "auto") {
      note.appendChild(document.createTextNode("The page reads the file extension to see what the file is. This choice only filters the file picker and sets expectations; the file is sent to the server unchanged and the server decides which checks apply."));
    } else {
      var ids = checksFor(TYPE);
      note.appendChild(document.createTextNode("Checks that can run on " + (TYPE === "audio" ? "an audio file" : (TYPE === "image" ? "an image" : "a video")) + ": " +
        ids.map(moduleName).join("; ") + ". A check that needs something the file lacks, such as a face or a sound track, will say it did not apply. This choice only filters the file picker; the file is sent unchanged."));
    }
    checkMismatch();
  }
  function mismatchActive() { return !$("type-mismatch").hidden; }
  function checkMismatch() {
    var box = $("type-mismatch"), inp = $("analyse-file");
    var f = inp.files && inp.files[0];
    var msg = "";
    if (f && TYPE !== "auto") {
      var k = kindOf(f.name, f.type);
      if (k !== TYPE) {
        msg = k ? "This looks like " + (k === "audio" ? "an audio file" : (k === "image" ? "an image" : "a video")) + ", but you chose " + TYPE + ". Change the choice above, or choose another file."
                : "This file type is not a supported " + TYPE + " type. Change the choice above, or choose another file.";
      }
    }
    box.textContent = msg;
    box.hidden = !msg;
    $("analyse-btn").disabled = !MODE.live || !f || !!msg;
  }
  function showFile(f) {
    var dz = $("dropzone"), pv = $("preview");
    if (previewUrl) { try { URL.revokeObjectURL(previewUrl); } catch (e) { /* ignore */ } previewUrl = null; }
    previewKind = ""; previewName = "";
    var oldPv = document.querySelector("#meta .ov-preview");
    if (oldPv && oldPv.parentNode) oldPv.parentNode.removeChild(oldPv);   /* its URL was just revoked */
    clear(pv);
    if (!f) { dz.setAttribute("data-state", "empty"); $("file-card").hidden = true; willRun(""); checkMismatch(); return; }
    dz.setAttribute("data-state", "file");
    $("file-card").hidden = false;
    $("file-name").textContent = f.name;
    var kind = kindOf(f.name, f.type);
    var sub = [kind ? cap(kind) : "File", bytes(f.size)];
    var notes = [];
    if (MODE.accept && !kind) notes.push("This file type is not in the supported list, so the server is expected to refuse it.");
    if (num(MODE.maxBytes) && f.size > MODE.maxBytes) notes.push("The file is larger than the " + bytes(MODE.maxBytes) + " upload limit.");
    $("file-sub").textContent = sub.join(", ") + (notes.length ? ". " + notes.join(" ") : "");
    willRun(kind);
    checkMismatch();
    try {
      if (kind && window.URL && URL.createObjectURL) {
        previewUrl = URL.createObjectURL(f);
        previewKind = kind; previewName = f.name;
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
  function willRun(kind) {
    var box = $("will-run"); clear(box);
    var ids = kind ? checksFor(kind) : [];
    box.hidden = !ids.length;
    if (!ids.length) return;
    box.appendChild(el("p", "will-h", "These checks will try to run on this " + kindWord(kind) + ":"));
    var ul = el("ul", "will-list");
    ids.forEach(function (k) { ul.appendChild(el("li", null, moduleName(k))); });
    box.appendChild(ul);
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
    showStage("upload");
    resetProgress("");
    setLive("", "");
    try { $("analyse-file").value = ""; } catch (e) { /* ignore */ }
    $("scenario").value = "";
    showFile(null);
    setBusy(!MODE.live);
    window.scrollTo(0, 0);
  }

  function initLive() {
    var panel = $("live-panel"), inp = $("analyse-file"), dz = $("dropzone");
    $("analyse-btn").addEventListener("click", startAnalysis);
    $("change-btn").addEventListener("click", function () { inp.click(); });
    $("again-btn").addEventListener("click", resetUpload);
    $("progress-back").addEventListener("click", resetUpload);
    $("sources-toggle").addEventListener("change", function () { $("report").classList.toggle("show-sources", this.checked); });
    inp.addEventListener("change", onFileChosen);
    Array.prototype.forEach.call(document.querySelectorAll("input[name=content-type]"), function (r) {
      r.addEventListener("change", function () { if (r.checked) { TYPE = r.value; updateTypeUi(); } });
    });
    dz.querySelector(".dz-empty").addEventListener("click", function (e) { if (e.target.tagName !== "LABEL" && !inp.disabled) inp.click(); });
    ["dragenter", "dragover"].forEach(function (t) {
      dz.addEventListener(t, function (e) { if (inp.disabled) return; e.preventDefault(); dz.classList.add("drag"); });
    });
    ["dragleave", "drop"].forEach(function (t) { dz.addEventListener(t, function () { dz.classList.remove("drag"); }); });
    dz.addEventListener("drop", function (e) {
      if (inp.disabled || !e.dataTransfer || !e.dataTransfer.files.length) return;
      e.preventDefault();
      try { inp.files = e.dataTransfer.files; } catch (err) { return; }
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
      if (isObj(m) && isObj(m.accept)) { MODE.accept = m.accept; MODE.maxBytes = m.max_bytes; }
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

  /* ---------------- About the checks ---------------- */
  function renderAbout() {
    var host = $("about-checks"); clear(host);
    var ids = moduleIds().filter(function (k) { return isObj(VOCAB.modules[k]) && Array.isArray(VOCAB.modules[k].applies_to); });
    if (!ids.length) { host.appendChild(el("p", "empty", "The list of checks could not be loaded from the server.")); }
    else {
      var tbl = el("table", "about-t");
      var tr = el("tr");
      ["Check", "What it looks at", "Runs on"].forEach(function (c) { var x = el("th", null, c); x.scope = "col"; tr.appendChild(x); });
      var thead = el("thead"); thead.appendChild(tr); tbl.appendChild(thead);
      var tb = el("tbody");
      ids.forEach(function (k) {
        var v = VOCAB.modules[k], row = el("tr");
        var c1 = el("th", null, v.title); c1.scope = "row"; row.appendChild(c1);
        var c2 = el("td", null, v.what_it_checks); c2.setAttribute("data-label", "What it looks at"); row.appendChild(c2);
        var c3 = el("td", null, v.needs); c3.setAttribute("data-label", "Runs on"); row.appendChild(c3);
        tb.appendChild(row);
      });
      tbl.appendChild(tb);
      host.appendChild(tbl);
    }
    var dl = $("about-groups"); clear(dl);
    var groups = Array.isArray(VOCAB.groups) && VOCAB.groups.length ? VOCAB.groups : GROUP_ORDER.map(function (g) { return { id: g, label: g }; });
    groups.forEach(function (g) {
      if (!GROUP_ABOUT[g.id]) return;
      var d = el("div", "about-group g-" + g.id);
      var dt = el("dt");
      dt.appendChild(icon(g.id));
      dt.appendChild(el("span", null, g.label));
      d.appendChild(dt);
      d.appendChild(el("dd", null, GROUP_ABOUT[g.id]));
      dl.appendChild(d);
    });
    $("about-statement").textContent = str(VOCAB.statement);
    fillLimitsExtra($("about-extra"));
  }

  /* ---------------- top navigation: Analyse a file, Evaluation Lab, About the checks ---------------- */
  var VIEW = "analyse";
  var VIEW_TITLE = { analyse: "Media Analysis", lab: "Evaluation Lab | Media Analysis", about: "About the checks | Media Analysis" };
  function setView(name, push) {
    VIEW = (name === "lab" || name === "about") ? name : "analyse";
    ["analyse", "lab", "about"].forEach(function (v) {
      $("view-" + v).hidden = VIEW !== v;
      var n = $("nav-" + v);
      if (VIEW === v) n.setAttribute("aria-current", "page"); else n.removeAttribute("aria-current");
    });
    document.body.classList.toggle("view-lab", VIEW === "lab");
    $("mode-pill").classList.toggle("view-hidden", VIEW !== "analyse");      /* the mode pill describes analysis data only */
    document.title = VIEW_TITLE[VIEW];
    if (push) {
      try { history.replaceState(null, "", VIEW === "analyse" ? window.location.pathname + window.location.search : "#" + VIEW); } catch (e) { /* ignore */ }
    }
    if (VIEW === "lab") {
      if (window.evallab && typeof window.evallab.open === "function") window.evallab.open($("lab-root"));
      else { clear($("lab-root")); $("lab-root").appendChild(el("p", "error-box", "The Evaluation Lab script is not available on this server.")); }
    }
    if (VIEW === "about") renderAbout();
    return VIEW;
  }
  function initNav() {
    ["analyse", "lab", "about"].forEach(function (v) {
      $("nav-" + v).addEventListener("click", function () { setView(v, true); window.scrollTo(0, 0); });
    });
    window.addEventListener("hashchange", function () {
      var h = window.location.hash;
      if ((h === "#lab" || h === "#about") && VIEW !== h.slice(1)) setView(h.slice(1), false);
    });
    var h0 = window.location.hash;
    if (h0 === "#lab" || h0 === "#about") setView(h0.slice(1), false);
  }
  /* Scroll offset = the real height of the sticky header, so a scrolled-to section is never hidden under it. */
  function syncHeaderOffset() {
    var top = document.querySelector("header.site-header");
    if (!top || !window.getComputedStyle) return;
    var pinned = window.getComputedStyle(top).position === "sticky";
    document.documentElement.style.setProperty("--head-h", (pinned ? Math.ceil(top.getBoundingClientRect().height) : 0) + 12 + "px");
  }
  function initHeaderOffset() {
    syncHeaderOffset();
    window.addEventListener("resize", syncHeaderOffset);
    if (window.ResizeObserver && document.querySelector("header.site-header")) new ResizeObserver(syncHeaderOffset).observe(document.querySelector("header.site-header"));
  }

  function init() {
    initActions();
    initNav();
    initHeaderOffset();
    var vocab = fetch("/api/vocabulary").then(function (r) { return r.ok ? r.json() : null; }).then(function (v) {
      if (isObj(v)) VOCAB = v;
      if (VIEW === "about") renderAbout();
    }).catch(function () { /* labels fall back to the explanation text and raw codes */ });
    var mode = initLive();
    var sel = $("scenario");
    Promise.all([vocab, mode]).then(function (res) {
      var live = res[1] === true;
      updateTypeUi();
      return fetch("/api/fixtures").then(function (r) { return r.json(); }).then(function (names) {
        names.forEach(function (n) {
          var o = document.createElement("option");
          o.value = n; o.textContent = "MOCK: " + n.replace(/_/g, " ");
          sel.appendChild(o);
        });
        sel.addEventListener("change", function () {
          if (sel.value) { pollToken++; setBusy(!MODE.live); setLive("", ""); loadFixture(sel.value); }
          else resetUpload();
        });
        /* Live mode opens on the upload stage: no invented report is shown until someone asks for a scenario.
           A scenarios-only server has nothing else to show, so it opens the first scenario under the MOCK banner. */
        if (!live && names.length) { sel.value = names[0]; loadFixture(names[0]); }
        else if (!names.length && !live) { showStage("results"); showError(["No scenarios available."]); }
      });
    }).catch(function (e) { showStage("results"); showError(["Could not list scenarios: " + e.message]); });
  }

  window.dashboard = { renderReport: renderReport, shapeProblems: shapeProblems, outcomeCounts: outcomeCounts,
                       config: { pollIntervalMs: POLL_INTERVAL_MS, pollTimeoutMs: POLL_TIMEOUT_MS },
                       loadFixture: loadFixture, setView: setView, view: function () { return VIEW; } };
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init);
  else init();
})();
