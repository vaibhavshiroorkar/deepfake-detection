"use strict";
/* Results dashboard for report schema v2. Plain JS, no network except the local fixture API.
   Every report string is written with textContent, never with markup insertion. */
(function () {
  var SECTIONS = [
    ["authenticity", "Authenticity Analysis"],
    ["cross_modal", "Cross-Modal Consistency"],
    ["provenance", "Provenance and Supporting Checks"]
  ];
  var STATUS_LABEL = {
    completed: "Completed", not_applicable: "Not applicable", unsupported: "Unsupported",
    failed: "Failed", timed_out: "Timed out", skipped: "Skipped"
  };
  var STATUS_ORDER = ["completed", "not_applicable", "unsupported", "failed", "timed_out", "skipped"];
  var FINDING_LABEL = {
    indicates_manipulation: "Indicates manipulation",
    no_indication_within_tested_scope: "No indication within tested scope",
    inconclusive: "Inconclusive",
    consistent: "No mismatch found within tested scope",
    potential_mismatch: "Potential mismatch",
    credentials_valid: "Credentials present and valid",
    credentials_invalid: "Credentials present but invalid",
    no_credentials_found: "No credentials found"
  };
  var FINDING_CLASS = {
    indicates_manipulation: "f-indicates_manipulation",
    potential_mismatch: "f-potential_mismatch",
    credentials_invalid: "f-potential_mismatch"
  };

  function $(id) { return document.getElementById(id); }
  function el(tag, cls, text) {
    var e = document.createElement(tag);
    if (cls) e.className = cls;
    if (text !== undefined && text !== null) e.textContent = String(text);
    return e;
  }
  function clear(node) { while (node.firstChild) node.removeChild(node.firstChild); }
  function isObj(x) { return x !== null && typeof x === "object" && !Array.isArray(x); }
  function str(x) { return (x === null || x === undefined) ? "" : String(x); }
  function yn(x) { return x === true ? "yes" : (x === false ? "no" : "unknown"); }
  function statusClass(s) { return STATUS_LABEL[s] ? s : "unknown"; }

  /* Basic shape check. Returns a list of problems; empty means renderable. */
  function shapeProblems(r) {
    var p = [];
    if (!isObj(r)) return ["The report is not an object."];
    ["schema_version", "input", "probe", "modules", "sections", "coverage", "source_summary", "status", "statement"].forEach(function (k) {
      if (!(k in r)) p.push("Missing key: " + k);
    });
    if (p.length) return p;
    if (str(r.schema_version) !== "2") p.push("Unsupported schema_version: " + str(r.schema_version));
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

  function renderBanner(r) {
    var b = $("banner");
    var anyMock = r.modules.some(function (m) { return m.source === "mock"; });
    if (r.source_summary === "mock" || r.source_summary === "mixed" || anyMock) {
      b.textContent = r.source_summary === "mixed"
        ? "MOCK DATA: some module results below are invented scenario data, not measurements."
        : "MOCK DATA: every result below is invented scenario data. No model ran. Nothing here is a measurement.";
      b.hidden = false;
    } else {
      b.textContent = "";
      b.hidden = true;
    }
  }

  function renderMeta(r) {
    var s = $("meta"); clear(s);
    s.appendChild(el("h2", null, "Request"));
    var inp = r.input, st = r.status;
    var rows = [
      ["Request", r.request_id], ["File", inp.filename], ["Media type", inp.media_type],
      ["Size (bytes)", inp.size_bytes], ["SHA-256", inp.sha256],
      ["Report state", st.state], ["Processing time (ms)", st.processing_time_ms],
      ["Source summary", r.source_summary]
    ];
    rows.forEach(function (kv) {
      var d = el("div", "kv");
      d.appendChild(el("b", null, kv[0] + ": "));
      d.appendChild(el(kv[0] === "SHA-256" ? "code" : "span", null, str(kv[1])));
      s.appendChild(d);
    });
    if (st.state === "failed") {
      s.appendChild(el("p", "err", "Report state is failed: no module completed. This is not a statement about the media."));
    } else if (st.state === "partial") {
      s.appendChild(el("p", "warn", "Report state is partial: at least one module failed, timed out or was unsupported."));
    }
  }

  function appendModuleList(parent, list, withReason) {
    if (!list.length) { parent.appendChild(el("p", "empty", "None.")); return; }
    var ul = el("ul");
    list.forEach(function (m) {
      var li = el("li");
      li.appendChild(el("code", null, m.module_id));
      li.appendChild(el("span", "tag status st-" + statusClass(m.status), STATUS_LABEL[m.status] || str(m.status)));
      if (withReason) li.appendChild(el("span", "reason", " " + str(m.reason)));
      ul.appendChild(li);
    });
    parent.appendChild(ul);
  }

  function renderRouting(r) {
    var s = $("routing"); clear(s);
    s.appendChild(el("h2", null, "Media routing"));
    var p = r.probe;
    var rows = [
      ["Media type", p.media_type], ["Has video", yn(p.has_video)], ["Has audio", yn(p.has_audio)],
      ["Speech detected", yn(p.speech_detected)], ["Faces found", p.n_faces],
      ["Duration (s)", p.duration_s], ["Resolution", (p.width && p.height) ? p.width + " x " + p.height : ""],
      ["Container", p.container]
    ];
    var g = el("div", "covgrid");
    rows.forEach(function (kv) {
      var d = el("div", "kv");
      d.appendChild(el("b", null, kv[0] + ": "));
      d.appendChild(el("span", null, (kv[1] === undefined || kv[1] === null || kv[1] === "") ? "unknown" : kv[1]));
      g.appendChild(d);
    });
    s.appendChild(g);
    var ran = [], notRan = [];
    r.modules.forEach(function (m) {
      if (m.status === "completed" || m.status === "failed" || m.status === "timed_out") ran.push(m);
      else notRan.push(m);
    });
    s.appendChild(el("h3", null, "Modules that ran"));
    appendModuleList(s, ran, false);
    s.appendChild(el("h3", null, "Modules that did not run"));
    appendModuleList(s, notRan, true);
  }

  function listBlock(card, title, items, cls, fmt) {
    if (!Array.isArray(items) || !items.length) return;
    card.appendChild(el("div", "kv", title));
    var ul = el("ul");
    items.forEach(function (x) { ul.appendChild(el("li", cls, fmt ? fmt(x) : x)); });
    card.appendChild(ul);
  }
  function codeMsg(w) { return str(isObj(w) ? w.code : "") + ": " + str(isObj(w) ? w.message : w); }

  function renderCard(m) {
    var status = statusClass(m.status);
    var card = el("article", "card st-" + status);
    card.setAttribute("data-module", str(m.module_id));
    card.setAttribute("data-status", str(m.status));
    var head = el("div", "head");
    head.appendChild(el("h3", null, str(m.module_id)));
    var tags = el("div");
    if (m.source === "mock") tags.appendChild(el("span", "tag mock", "MOCK"));
    tags.appendChild(el("span", "tag status st-" + status, STATUS_LABEL[m.status] || str(m.status)));
    head.appendChild(tags);
    card.appendChild(head);
    card.appendChild(el("div", "kv", "version " + str(m.version) + " | source " + str(m.source)));

    if (m.status === "completed") {
      var f = el("div", "finding " + (FINDING_CLASS[m.finding] || "f-neutral"), FINDING_LABEL[m.finding] || "Inconclusive");
      f.setAttribute("data-finding", str(m.finding));
      card.appendChild(f);
      if (isObj(m.score)) {
        var sc = el("div", "kv");
        sc.appendChild(el("b", null, "Score: "));
        sc.appendChild(el("span", null, str(m.score.value) + (m.source === "mock" ? " (MOCK number)" : "")));
        card.appendChild(sc);
        card.appendChild(el("div", "kv", "Meaning: " + str(m.score.meaning)));
        card.appendChild(el("div", "kv", "Calibrated: " + (m.score.calibrated ? "yes" : "no") +
          ((m.score.threshold === null || m.score.threshold === undefined) ? "" : " | threshold " + str(m.score.threshold))));
      } else {
        card.appendChild(el("div", "kv", "No score reported."));
      }
      if (m.units_analysed !== null && m.units_analysed !== undefined) card.appendChild(el("div", "kv", "Units analysed: " + str(m.units_analysed)));
      var scope = el("div", "kv");
      scope.appendChild(el("b", null, "Tested scope: "));
      scope.appendChild(el("span", null, str(m.tested_scope)));
      card.appendChild(scope);
    } else {
      var rs = el("div", "kv");
      rs.appendChild(el("b", null, "Reason: "));
      rs.appendChild(el("span", "reason", str(m.reason)));
      card.appendChild(rs);
      card.appendChild(el("div", "kv", "No finding is reported for this status."));
    }
    var OBS = { file_metadata: "File metadata", heuristic: "Heuristic flag (descriptive only)", marker_scan: "Marker scan (not a verification)", verification: "Verified" };
    listBlock(card, "Observations", m.observations, null, function (o) {
      return (isObj(o) ? (OBS[o.kind] || str(o.kind)) : "") + ": " + str(isObj(o) ? o.text : o);
    });
    listBlock(card, "Limitations", m.limitations, null);
    listBlock(card, "Warnings", m.warnings, "warn", codeMsg);
    listBlock(card, "Errors", m.errors, "err", codeMsg);
    listBlock(card, "Segments", m.segments, null, function (g) {
      return str(isObj(g) ? g.start_s : "") + " s to " + str(isObj(g) ? g.end_s : "") + " s" + (isObj(g) && g.note ? ": " + str(g.note) : "");
    });
    return card;
  }

  function renderSections(r) {
    var byId = {};
    r.modules.forEach(function (m) { byId[m.module_id] = m; });
    SECTIONS.forEach(function (pair) {
      var key = pair[0], host = $("sec-" + key);
      clear(host);
      host.appendChild(el("h2", null, pair[1]));
      var ids = Array.isArray(r.sections[key]) ? r.sections[key] : [];
      var mods = ids.map(function (i) { return byId[i]; }).filter(Boolean);
      if (!mods.length) { host.appendChild(el("p", "empty", "No modules in this section.")); return; }
      var cards = el("div", "cards");
      mods.forEach(function (m) { cards.appendChild(renderCard(m)); });
      host.appendChild(cards);
    });
  }

  function renderCoverage(r) {
    var s = $("coverage"); clear(s);
    s.appendChild(el("h2", null, "Coverage"));
    var g = el("div", "covgrid");
    STATUS_ORDER.forEach(function (st) {
      var items = Array.isArray(r.coverage[st]) ? r.coverage[st] : [];
      var box = el("div", "covbox st-" + st);
      box.setAttribute("data-coverage", st);
      box.appendChild(el("h3", null, STATUS_LABEL[st] + " (" + items.length + ")"));
      if (!items.length) box.appendChild(el("div", "empty", "None."));
      else {
        var ul = el("ul");
        items.forEach(function (i) {
          var li = el("li");
          li.appendChild(el("code", null, str(isObj(i) ? i.module_id : "")));
          if (isObj(i) && i.reason) li.appendChild(el("span", "reason", ": " + str(i.reason)));
          ul.appendChild(li);
        });
        box.appendChild(ul);
      }
      g.appendChild(box);
    });
    s.appendChild(g);
  }

  function renderConflicts(r) {
    var s = $("conflicts"); clear(s);
    s.appendChild(el("h2", null, "Conflicts between modules"));
    var c = Array.isArray(r.conflicts) ? r.conflicts : [];
    if (!c.length) { s.appendChild(el("p", "empty", "No conflicts recorded.")); return; }
    c.forEach(function (x) {
      var d = el("div", "covbox st-skipped");
      d.appendChild(el("div", "kv", "Modules: " + (isObj(x) && Array.isArray(x.modules) ? x.modules.join(", ") : "")));
      d.appendChild(el("div", null, str(isObj(x) ? x.note : "")));
      s.appendChild(d);
    });
  }

  function renderReport(r) {
    $("error").hidden = true;
    var probs = shapeProblems(r);
    if (probs.length) { $("banner").hidden = true; showError(probs); return false; }
    try {
      renderBanner(r);
      renderMeta(r);
      renderRouting(r);
      renderSections(r);
      renderCoverage(r);
      renderConflicts(r);
      $("statement").textContent = str(r.statement);
      $("report").hidden = false;
      return true;
    } catch (e) {
      $("banner").hidden = true;
      showError(["Rendering failed: " + (e && e.message ? e.message : "unknown error")]);
      return false;
    }
  }

  function loadFixture(name) {
    return fetch("/api/fixtures/" + encodeURIComponent(name)).then(function (res) {
      if (!res.ok) throw new Error("Server answered " + res.status + " for " + name);
      return res.json();
    }).then(renderReport).catch(function (e) {
      $("banner").hidden = true;
      showError(["Could not load scenario " + name + ": " + e.message]);
    });
  }

  function init() {
    var sel = $("scenario");
    fetch("/api/fixtures").then(function (res) { return res.json(); }).then(function (names) {
      clear(sel);
      names.forEach(function (n) {
        var o = document.createElement("option");
        o.value = n; o.textContent = n;
        sel.appendChild(o);
      });
      sel.addEventListener("change", function () { loadFixture(sel.value); });
      if (names.length) loadFixture(names[0]);
      else showError(["No scenarios available."]);
    }).catch(function (e) { showError(["Could not list scenarios: " + e.message]); });
  }

  window.dashboard = { renderReport: renderReport, shapeProblems: shapeProblems, loadFixture: loadFixture };
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init);
  else init();
})();
