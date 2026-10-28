"use strict";
/* Evaluation Lab view (docs/EVAL_LAB_AND_UI_CONTRACT.md section 3, decision D-088).
   Plain JS, no network except the local server's own /api/evallab/... routes. Every string from the API is written with textContent.
   Nothing numeric is hard-coded here: metrics, intervals, confusion matrix, ROC points, breakdown rows, warnings and skipped rows all come
   from the run record. The only fixed text is explanation of what a number means. The lab re-analyses recorded score files; it does not
   run models and it never approves anything. Test hook: ?lab_poll_ms=N sets the polling interval (default 800). */
(function () {
  var SVGNS = "http://www.w3.org/2000/svg";
  var POLL_MS = (function () {
    try { var v = Number(new URLSearchParams(window.location.search).get("lab_poll_ms")); return (isFinite(v) && v > 0) ? v : 800; } catch (e) { return 800; }
  })();
  var METRICS = {
    auc: ["AUC", "How well the scores rank fake samples above real ones (0.5 is chance, 1 is perfect). It does not depend on a threshold."],
    eer: ["EER", "Error rate at the point where false alarms and misses are equal. Lower is better. It does not depend on the chosen threshold."],
    precision: ["Precision", "Of the samples flagged as fake, the share that really were fake. Depends on how common fakes are in this block."],
    recall: ["Recall", "Of the fake samples, the share that were flagged."],
    specificity: ["Specificity", "Of the real samples, the share that were not flagged."],
    fpr: ["False positive rate", "Real samples flagged as fake, as a share of the real samples."],
    fnr: ["False negative rate", "Fake samples that were missed, as a share of the fake samples."],
    f1: ["F1", "Balance of precision and recall at the threshold."],
    accuracy: ["Accuracy", "Share of all samples classified correctly. It depends on the class balance of the block."],
    balanced_accuracy: ["Balanced accuracy", "Mean of recall and specificity; not affected by class balance."]
  };
  var root = null, built = false;
  var S = { cat: null, fileId: null, blockKey: null, run: null, runs: [], token: 0, busy: false };

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
  function clear(n) { while (n.firstChild) n.removeChild(n.firstChild); }
  function isObj(x) { return x !== null && typeof x === "object" && !Array.isArray(x); }
  function str(x) { return (x === null || x === undefined) ? "" : String(x); }
  function num(x) { return typeof x === "number" && isFinite(x); }
  function f4(x) { return num(x) ? x.toFixed(4) : "not defined"; }
  function $(id) { return document.getElementById(id); }
  function field(label, control, hint, id) {
    var w = el("div", "lab-field");
    var l = el("label", null, label);
    if (id) l.setAttribute("for", id);
    w.appendChild(l);
    w.appendChild(control);
    if (hint) w.appendChild(el("p", "lab-hint", hint));
    return w;
  }
  function kv(rows) {
    var dl = el("dl", "kv lab-kv");
    rows.forEach(function (r) {
      if (r[1] === null || r[1] === undefined || r[1] === "") return;
      dl.appendChild(el("dt", null, r[0]));
      dl.appendChild(el("dd", r[2] || null, str(r[1])));
    });
    return dl;
  }
  function api(method, url, body) {
    var opt = { method: method, headers: {} };
    if (body !== undefined) { opt.headers["Content-Type"] = "application/json"; opt.body = JSON.stringify(body); }
    return fetch(url, opt).then(function (res) {
      return res.json().catch(function () { return null; }).then(function (j) { return { ok: res.ok, status: res.status, j: j }; });
    });
  }
  function errMsg(x) {
    var e = isObj(x.j) && isObj(x.j.error) ? x.j.error : (isObj(x.j) && x.j.reason ? { code: "REFUSED", message: x.j.reason } : null);
    return e ? (str(e.code) + ": " + str(e.message)) : ("HTTP " + x.status + ": the server gave no details.");
  }
  function roleClass(split) {
    if (/^test/.test(str(split))) return "final";
    if (split === "val") return "validation";
    return "development";
  }
  function roleBadge(block, cat) {
    var cls = roleClass(block.split);
    var text = cls === "final" ? "Final held-out (sealed)" : (cls === "validation" ? "Validation" : "Development");
    var b = el("span", "role-badge role-" + cls, text);
    b.title = str(block.role) || (isObj(cat.roles) ? str(cat.roles[block.split]) : "");
    return b;
  }
  function blockKey(b) { return str(b.dataset) + "|" + str(b.split); }
  function curFile() { return S.cat && S.cat.score_files.filter(function (f) { return f.id === S.fileId; })[0] || null; }
  function curBlock() { var f = curFile(); return f && f.blocks.filter(function (b) { return blockKey(b) === S.blockKey; })[0] || null; }

  /* ---------------- skeleton ---------------- */
  function build() {
    clear(root);
    var hero = el("section", "panel lab-hero");
    hero.appendChild(el("p", "hero-kicker", "Research tool"));
    var h = el("h1", null, "Evaluation Lab");
    h.id = "lab-h";
    hero.appendChild(h);
    hero.appendChild(el("p", "lede", "Re-analyse the recorded scores of a model with the project's own metrics. Pick a score file and one block of it, review the settings, run, and read the result with its uncertainty."));
    var how = el("ul", "lab-explain");
    [["What data", "Score files recorded by earlier evaluations: one row per sample with its label, group and score. Nothing is uploaded and no model is run."],
     ["Is it unseen?", "Each block shows its role. Development and validation blocks may have been used to tune a model; only a final held-out block is sealed, and the lab opens it only to re-analyse an evaluation that is already logged."],
     ["What the numbers mean", "Every metric has an interval from a bootstrap that resamples whole groups (a person or source), so near-identical samples do not count twice. A wide interval means the block is too small to say much."],
     ["What it cannot tell you", "A score file may not record what the model was trained on, so the lab checks only what the file records (groups, splits, hashes). A good number here is not an approval of any model."]
    ].forEach(function (x) {
      var li = el("li");
      li.appendChild(el("b", null, x[0] + ". "));
      li.appendChild(document.createTextNode(x[1]));
      how.appendChild(li);
    });
    hero.appendChild(how);
    root.appendChild(hero);

    var steps = el("ol", "lab-stepper");
    steps.id = "lab-stepper";
    steps.setAttribute("aria-label", "Evaluation steps");
    ["Choose data", "Review settings", "Run", "Results", "Compare and export"].forEach(function (t, i) {
      var li = el("li", null, t);
      li.setAttribute("data-step", String(i + 1));
      steps.appendChild(li);
    });
    root.appendChild(steps);

    [["lab-catalogue", "1. Choose a score file and a block"], ["lab-config", "2. Review the settings"], ["lab-run", "3. Run"],
     ["lab-results", "4. Results"], ["lab-compare", "5. Compare with a baseline and export"]].forEach(function (s) {
      var sec = el("section", "panel lab-section");
      sec.id = s[0];
      var hh = el("h2", null, s[1]);
      hh.id = s[0] + "-h";
      sec.setAttribute("aria-labelledby", hh.id);
      sec.appendChild(hh);
      sec.appendChild(el("div", "lab-body"));
      root.appendChild(sec);
    });
    ["lab-config", "lab-run", "lab-results", "lab-compare"].forEach(function (id) { $(id).hidden = true; });
    $("lab-run").hidden = true;
    built = true;
  }
  function body(id) { return $(id).querySelector(".lab-body"); }
  function setStep(n) {
    Array.prototype.forEach.call($("lab-stepper").children, function (li) {
      var k = Number(li.getAttribute("data-step"));
      li.className = k < n ? "done" : (k === n ? "current" : "");
      if (k === n) li.setAttribute("aria-current", "step"); else li.removeAttribute("aria-current");
    });
  }

  /* ---------------- step 1: catalogue ---------------- */
  function loadCatalogue() {
    var b = body("lab-catalogue");
    clear(b);
    b.appendChild(el("p", "lab-loading", "Looking for score files ..."));
    return api("GET", "/api/evallab/catalogue").then(function (x) {
      clear(b);
      if (!x.ok || !isObj(x.j)) {
        var box = el("div", "error-box");
        box.setAttribute("role", "alert");
        box.appendChild(el("strong", null, "The score-file catalogue could not be loaded. "));
        box.appendChild(document.createTextNode(errMsg(x)));
        b.appendChild(box);
        var again = el("button", "btn", "Try again");
        again.type = "button";
        again.addEventListener("click", loadCatalogue);
        b.appendChild(again);
        setStep(1);
        return;
      }
      S.cat = x.j;
      if (!Array.isArray(S.cat.score_files)) S.cat.score_files = [];
      renderCatalogue();
    }).catch(function (e) {
      clear(b);
      var box = el("div", "error-box", "The Evaluation Lab API could not be reached: " + (e && e.message ? e.message : "request failed"));
      box.setAttribute("role", "alert");
      b.appendChild(box);
    });
  }
  function renderCatalogue() {
    var b = body("lab-catalogue");
    clear(b);
    var cat = S.cat;
    if (!cat.score_files.length) {
      var em = el("div", "lab-empty");
      em.id = "lab-empty";
      em.appendChild(el("h3", null, "No score files found"));
      em.appendChild(el("p", null, "The lab looks for recorded score files in these folders on this computer and found none. Nothing is shown here until a score file exists, and no example data is invented."));
      var ul = el("ul");
      (Array.isArray(cat.default_dirs) ? cat.default_dirs : []).forEach(function (d) { ul.appendChild(el("li", "mono", d)); });
      em.appendChild(ul);
      em.appendChild(el("p", "lab-hint", "A score file has one row per sample with the columns run_id, iteration, seed, modality, dataset, split, sample_id, group_id, label and score. The search folders can be changed with the server option or the environment variable DFD_EVALLAB_DIRS."));
      b.appendChild(em);
      $("lab-config").hidden = true;
      setStep(1);
      return;
    }
    if (!S.fileId || !curFile()) S.fileId = cat.score_files[0].id;
    b.appendChild(el("p", "lab-hint", "A score file holds the recorded scores of one or more runs. Choose one, then choose the block (dataset and split) to evaluate."));
    var list = el("div", "lab-files");
    list.id = "lab-files";
    list.setAttribute("role", "radiogroup");
    list.setAttribute("aria-label", "Score files");
    cat.score_files.forEach(function (f) {
      var lab = el("label", "pick" + (f.id === S.fileId ? " on" : ""));
      var r = el("input");
      r.type = "radio"; r.name = "lab-file"; r.value = f.id; r.checked = f.id === S.fileId;
      r.addEventListener("change", function () { S.fileId = f.id; S.blockKey = null; renderCatalogue(); });
      var t = el("span", "pick-body");
      t.appendChild(el("b", "pick-title", str(f.path)));
      t.appendChild(el("span", "pick-sub", f.n_rows + " rows; iterations: " + (f.iterations || []).join(", ") + "; modalities: " + (f.modalities || []).join(", ")));
      t.appendChild(el("span", "pick-sub mono", "SHA-256 " + str(f.sha256).slice(0, 16) + (str(f.sha256).length > 16 ? "..." : "")));
      lab.appendChild(r);
      lab.appendChild(t);
      list.appendChild(lab);
    });
    b.appendChild(list);
    var f0 = curFile();
    b.appendChild(el("h3", "lab-sub", "Blocks in this file"));
    var grid = el("div", "lab-blocks");
    grid.id = "lab-blocks";
    grid.setAttribute("role", "radiogroup");
    grid.setAttribute("aria-label", "Blocks");
    (f0.blocks || []).forEach(function (bl) {
      var key = blockKey(bl);
      var lab = el("label", "pick block" + (key === S.blockKey ? " on" : ""));
      lab.setAttribute("data-split", str(bl.split));
      var r = el("input");
      r.type = "radio"; r.name = "lab-block"; r.value = key; r.checked = key === S.blockKey;
      r.addEventListener("change", function () { S.blockKey = key; renderCatalogue(); renderConfig(); });
      var t = el("span", "pick-body");
      var top = el("span", "pick-top");
      top.appendChild(el("b", "pick-title", str(bl.dataset) + " / " + str(bl.split)));
      top.appendChild(roleBadge(bl, cat));
      t.appendChild(top);
      t.appendChild(el("span", "pick-sub", bl.n_samples + " samples (" + bl.n_fake + " fake, " + bl.n_real + " real) in " + bl.n_groups + " groups"));
      t.appendChild(el("span", "pick-sub", "Runs: " + (bl.runs || []).length + (Array.isArray(bl.seeds) && bl.seeds.length ? "; seeds " + bl.seeds.join(", ") : "")));
      lab.appendChild(r);
      lab.appendChild(t);
      grid.appendChild(lab);
    });
    if (!(f0.blocks || []).length) grid.appendChild(el("p", "empty", "This file lists no blocks."));
    b.appendChild(grid);
    if (isObj(cat.roles)) {
      var rl = el("details", "lab-roles");
      rl.appendChild(el("summary", null, "What the role badges mean"));
      var rul = el("ul");
      Object.keys(cat.roles).forEach(function (k) { var li = el("li"); li.appendChild(el("b", null, k + ": ")); li.appendChild(document.createTextNode(str(cat.roles[k]))); rul.appendChild(li); });
      rl.appendChild(rul);
      b.appendChild(rl);
    }
    setStep(S.blockKey ? 2 : 1);
  }

  /* ---------------- step 2: configuration ---------------- */
  function renderConfig() {
    var sec = $("lab-config"), b = body("lab-config");
    var f = curFile(), bl = curBlock();
    sec.hidden = !bl;
    if (!bl) return;
    clear(b);
    var final = roleClass(bl.split) === "final";
    var form = el("form", "lab-form");
    form.id = "lab-form";
    form.addEventListener("submit", function (ev) { ev.preventDefault(); startRun(); });

    var ssum = el("div", "lab-summary");
    ssum.id = "lab-block-summary";
    ssum.appendChild(el("h3", null, "Block: " + str(bl.dataset) + " / " + str(bl.split)));
    ssum.appendChild(roleBadge(bl, S.cat));
    var share = bl.n_samples ? (bl.n_fake / bl.n_samples) : null;
    ssum.appendChild(kv([["Samples", bl.n_samples], ["Fake (positive)", bl.n_fake], ["Real (negative)", bl.n_real],
      ["Fake share of the block", num(share) ? share.toFixed(3) : ""], ["Groups", bl.n_groups]]));
    if (final) {
      var fn = el("p", "lab-note final", "This block is sealed. The lab will run it only as a re-analysis of an evaluation that is already in the ledger. It never makes a first look at sealed data, and it writes nothing to the ledger.");
      ssum.appendChild(fn);
    } else if (roleClass(bl.split) === "validation") {
      ssum.appendChild(el("p", "lab-note", "Validation rows are where thresholds and temperatures are fitted, so results here are optimistic and are not a test of unseen data."));
    } else {
      ssum.appendChild(el("p", "lab-note", "A development block can be repeated while a model is being built. Results here are not a final claim."));
    }
    form.appendChild(ssum);

    /* Candidates (iteration and modality with their runs) come from the catalogue; a file without them falls back to the block's runs. */
    var cands = Array.isArray(f.candidates) ? f.candidates.filter(function (c) {
      return !Array.isArray(c.blocks) || c.blocks.some(function (q) { return q[0] === bl.dataset && q[1] === bl.split; });
    }) : null;
    var iters = cands ? cands.map(function (c) { return c.iteration; }).filter(function (x, i, a) { return a.indexOf(x) === i; }) : (f.iterations || []);
    var grid = el("div", "lab-grid");
    var itSel = el("select"); itSel.id = "lab-iteration";
    iters.forEach(function (x) { var o = el("option", null, x); o.value = x; itSel.appendChild(o); });
    grid.appendChild(field("Model iteration", itSel, "The model version whose scores are evaluated.", "lab-iteration"));
    var moSel = el("select"); moSel.id = "lab-modality";
    grid.appendChild(field("Modality", moSel, "Image, video, audio and fused scores are never mixed.", "lab-modality"));
    form.appendChild(grid);

    var runs = el("fieldset", "lab-runs");
    runs.id = "lab-runs";
    runs.appendChild(el("legend", null, "Runs (seeds of one candidate)"));
    var runBox = el("div", "lab-runbox"); runs.appendChild(runBox);
    runs.appendChild(el("p", "lab-hint", "All runs are used by default. The metrics describe the pooled candidate, with a bootstrap over groups."));
    function fillModalities() {
      clear(moSel);
      var mods = cands ? cands.filter(function (c) { return c.iteration === itSel.value; }).map(function (c) { return c.modality; }) : (f.modalities || []);
      mods.forEach(function (x) { var o = el("option", null, x); o.value = x; moSel.appendChild(o); });
      fillRuns();
    }
    function fillRuns() {
      clear(runBox);
      var cand = cands ? cands.filter(function (c) { return c.iteration === itSel.value && c.modality === moSel.value; })[0] : null;
      var ids = cand ? cand.run_ids : (cands ? [] : (bl.runs || []));
      var seeds = cand ? cand.seeds : bl.seeds;
      ids.forEach(function (rid, i) {
        var l = el("label", "check");
        var c = el("input"); c.type = "checkbox"; c.value = rid; c.checked = true; c.name = "lab-run-id";
        l.appendChild(c);
        var seed = Array.isArray(seeds) && seeds.length === ids.length ? " (seed " + seeds[i] + ")" : "";
        l.appendChild(el("span", "mono", rid + seed));
        runBox.appendChild(l);
      });
      if (!ids.length) runBox.appendChild(el("p", "empty", "No runs are listed for this combination."));
    }
    itSel.addEventListener("change", fillModalities);
    moSel.addEventListener("change", fillRuns);
    fillModalities();
    form.appendChild(runs);

    var th = el("fieldset", "lab-thr");
    th.appendChild(el("legend", null, "Decision threshold"));
    var o1 = el("label", "check"); var r1 = el("input"); r1.type = "radio"; r1.name = "lab-thr"; r1.value = "val_fit"; r1.checked = true;
    o1.appendChild(r1);
    o1.appendChild(el("span", null, "Fit on the validation rows of the same runs (maximum balanced accuracy)"));
    var o2 = el("label", "check"); var r2 = el("input"); r2.type = "radio"; r2.name = "lab-thr"; r2.value = "fixed";
    o2.appendChild(r2);
    o2.appendChild(el("span", null, "Use a fixed value, with its provenance"));
    th.appendChild(o1); th.appendChild(o2);
    var fx = el("div", "lab-fixed"); fx.id = "lab-fixed"; fx.hidden = true;
    var tv = el("input"); tv.type = "number"; tv.step = "any"; tv.id = "lab-thr-value";
    fx.appendChild(field("Threshold value", tv, "", "lab-thr-value"));
    var tp = el("textarea"); tp.id = "lab-thr-prov";
    fx.appendChild(field("Where this value comes from", tp, "Required. For example the experiment that fixed it. The threshold must not have been chosen on the rows evaluated here.", "lab-thr-prov"));
    th.appendChild(fx);
    var tn = el("p", "lab-hint", bl.split === "val"
      ? "The evaluated block is itself validation data, so a threshold fitted on it is marked as fitted on these rows (optimistic)."
      : "The threshold is never fitted on the rows being evaluated.");
    th.appendChild(tn);
    [r1, r2].forEach(function (r) { r.addEventListener("change", function () { fx.hidden = !r2.checked; }); });
    form.appendChild(th);

    var g2 = el("div", "lab-grid");
    var pur = el("select"); pur.id = "lab-purpose";
    [["development", "Development"], ["validation", "Validation"], ["final_reanalysis", "Final re-analysis of a logged evaluation"]].forEach(function (p) {
      var o = el("option", null, p[1]); o.value = p[0];
      if (final ? p[0] !== "final_reanalysis" : p[0] === "final_reanalysis") o.disabled = true;
      pur.appendChild(o);
    });
    pur.value = final ? "final_reanalysis" : (bl.split === "val" ? "validation" : "development");
    g2.appendChild(field("Purpose", pur, final ? "Sealed blocks allow only a re-analysis." : "Recorded in the run so a result is never mistaken for a final one.", "lab-purpose"));
    var lbl = el("input"); lbl.type = "text"; lbl.id = "lab-label"; lbl.maxLength = 120;
    g2.appendChild(field("Label for this run", lbl, "Optional, for finding it again.", "lab-label"));
    var nb = el("input"); nb.type = "number"; nb.id = "lab-nboot"; nb.min = "100"; nb.max = "5000"; nb.step = "100"; nb.value = "2000";
    g2.appendChild(field("Bootstrap resamples", nb, "Between 100 and 5000. More resamples give smoother intervals and take longer.", "lab-nboot"));
    var sd = el("input"); sd.type = "number"; sd.id = "lab-seed"; sd.step = "1"; sd.value = "20261027";
    g2.appendChild(field("Bootstrap seed", sd, "Fixed so the same settings give the same intervals.", "lab-seed"));
    form.appendChild(g2);

    var msg = el("p", "lab-formerr"); msg.id = "lab-formerr"; msg.setAttribute("role", "alert"); msg.hidden = true;
    form.appendChild(msg);
    var run = el("button", "btn primary", "Run evaluation");
    run.type = "submit"; run.id = "lab-run-btn";
    form.appendChild(run);
    b.appendChild(form);
  }

  function readConfig() {
    var f = curFile(), bl = curBlock();
    var thr = document.querySelector("input[name=lab-thr]:checked").value;
    var runIds = Array.prototype.map.call(document.querySelectorAll("input[name=lab-run-id]:checked"), function (c) { return c.value; });
    var cfg = { score_file_id: f.id, iteration: $("lab-iteration").value, modality: $("lab-modality").value, dataset: bl.dataset, split: bl.split,
      run_ids: runIds, threshold_source: thr, n_boot: Number($("lab-nboot").value), seed: Number($("lab-seed").value),
      label: $("lab-label").value, purpose: $("lab-purpose").value };
    var err = "";
    if (!runIds.length) err = "Choose at least one run.";
    if (!(cfg.n_boot >= 100 && cfg.n_boot <= 5000)) err = err || "Bootstrap resamples must be between 100 and 5000.";
    if (!isFinite(cfg.seed)) err = err || "The bootstrap seed must be a number.";
    if (thr === "fixed") {
      var v = $("lab-thr-value").value;
      if (v === "" || !isFinite(Number(v))) err = err || "Enter the fixed threshold value.";
      else cfg.threshold = Number(v);
      cfg.threshold_provenance = $("lab-thr-prov").value.trim();
      if (!cfg.threshold_provenance) err = err || "A fixed threshold needs a provenance text (where the value comes from).";
    }
    return { cfg: cfg, err: err };
  }

  /* ---------------- step 3: run and progress ---------------- */
  function startRun() {
    if (S.busy) return;
    var rc = readConfig(), fe = $("lab-formerr");
    fe.hidden = !rc.err; fe.textContent = rc.err;
    if (rc.err) return;
    S.busy = true;
    $("lab-run-btn").disabled = true;
    var token = ++S.token;
    $("lab-results").hidden = true; $("lab-compare").hidden = true;
    var sec = $("lab-run"), b = body("lab-run");
    sec.hidden = false; clear(b);
    setStep(3);
    var status = el("div", "lab-status"); status.id = "lab-status"; status.setAttribute("role", "status");
    status.textContent = "Submitting the run ...";
    b.appendChild(status);
    api("POST", "/api/evallab/runs", rc.cfg).then(function (x) {
      if (token !== S.token) return;
      var rec = isObj(x.j) ? x.j : null;
      var id = rec && (rec.id || rec.run_id);
      if (rec && rec.state === "refused") { showRefused(rec); return; }
      if (!x.ok || !id) { showFailure(isObj(x.j) && isObj(x.j.error) ? x.j.error : { code: "HTTP_" + x.status, message: "The server did not accept the run." }, true); return; }
      poll(str(id), token);
    }).catch(function (e) { if (token === S.token) showFailure({ code: "NETWORK", message: e && e.message ? e.message : "request failed" }, false); });
  }
  function endBusy() { S.busy = false; var r = $("lab-run-btn"); if (r) r.disabled = false; }
  function showFailure(err, refused) {
    endBusy();
    var b = body("lab-run"); clear(b);
    var box = el("div", "error-box");
    box.id = "lab-error";
    box.setAttribute("role", "alert");
    box.appendChild(el("strong", null, refused ? "The run was not started. " : "The run failed. "));
    box.appendChild(document.createTextNode(str(err.code) + ": " + str(err.message)));
    b.appendChild(box);
    b.appendChild(el("p", "lab-hint", "Nothing was computed or stored for a refused run. Change the settings above and run again."));
  }
  function showRefused(rec) {
    showFailure(isObj(rec.error) ? rec.error : { code: "REFUSED", message: "The lab refused this configuration." }, true);
  }
  function poll(id, token) {
    if (token !== S.token) return;
    api("GET", "/api/evallab/runs/" + encodeURIComponent(id)).then(function (x) {
      if (token !== S.token) return;
      if (!x.ok || !isObj(x.j)) { showFailure({ code: "HTTP_" + x.status, message: "Could not read the run." }, false); return; }
      var r = x.j;
      if (r.state === "done") { endBusy(); S.run = r; clear(body("lab-run")); body("lab-run").appendChild(el("p", "lab-status done", "Finished: run " + str(r.id) + ".")); renderResults(r); return; }
      if (r.state === "failed") { showFailure(isObj(r.error) ? r.error : { code: "FAILED", message: "The run failed without details." }, false); return; }
      if (r.state === "refused") { showRefused(r); return; }
      var b = body("lab-run"); clear(b);
      var p = isObj(r.progress) ? r.progress : {};
      var st = el("div", "lab-status"); st.id = "lab-status"; st.setAttribute("role", "status");
      st.textContent = (r.state === "queued" ? "Queued" : "Running") + (p.stage ? ": " + p.stage : "") + (num(p.total) && p.total > 0 ? " (" + p.done + " of " + p.total + ")" : "");
      b.appendChild(st);
      if (num(p.total) && p.total > 0) {
        var bar = el("div", "lab-bar"); bar.setAttribute("role", "progressbar"); bar.setAttribute("aria-valuemin", "0");
        bar.setAttribute("aria-valuemax", String(p.total)); bar.setAttribute("aria-valuenow", String(p.done || 0));
        var fill = el("span"); fill.style.width = Math.max(0, Math.min(100, 100 * (p.done || 0) / p.total)) + "%";
        bar.appendChild(fill); b.appendChild(bar);
      }
      setTimeout(function () { poll(id, token); }, POLL_MS);
    }).catch(function (e) { if (token === S.token) showFailure({ code: "NETWORK", message: e && e.message ? e.message : "request failed" }, false); });
  }

  /* ---------------- step 4: results ---------------- */
  function interval(m) {
    if (!isObj(m)) return "";
    return num(m.low) && num(m.high) ? "[" + f4(m.low) + ", " + f4(m.high) + "]" : "no interval";
  }
  function metricsTable(r) {
    var tbl = el("table", "lab-t");
    tbl.id = "lab-metrics";
    var cap = el("caption", null, "Metrics with bootstrap intervals (resampling whole groups, " + str(isObj(r.config) ? r.config.n_boot : "") + " resamples, seed " + str(isObj(r.config) ? r.config.seed : "") + ")");
    tbl.appendChild(cap);
    var hr = el("tr");
    ["Metric", "Estimate", "Interval", "What it measures"].forEach(function (c) { var th = el("th", null, c); th.scope = "col"; hr.appendChild(th); });
    var th = el("thead"); th.appendChild(hr); tbl.appendChild(th);
    var tb = el("tbody");
    Object.keys(isObj(r.metrics) ? r.metrics : {}).forEach(function (k) {
      var m = r.metrics[k], tr = el("tr");
      tr.setAttribute("data-metric", k);
      var lab = METRICS[k] || [k, ""];
      tr.appendChild(el("th", null, lab[0])); tr.cells[0].scope = "row";
      tr.appendChild(el("td", "num", isObj(m) ? f4(m.estimate) : f4(m)));
      tr.appendChild(el("td", "num", interval(m)));
      tr.appendChild(el("td", "lab-what", lab[1]));
      tb.appendChild(tr);
    });
    tbl.appendChild(tb);
    var wrap = el("div", "lab-scroll"); wrap.appendChild(tbl);
    return wrap;
  }
  function confusion(c) {
    var tbl = el("table", "lab-t lab-conf");
    tbl.id = "lab-confusion";
    tbl.appendChild(el("caption", null, "Confusion matrix at the threshold (counts of samples" +
      (str(c.aggregation) && c.aggregation !== "single run" ? "; " + c.aggregation + ", so counts can be fractional)" : ")")));
    var hr = el("tr");
    ["", "Flagged as fake", "Not flagged"].forEach(function (t, i) { var th = el("th", null, t); th.scope = "col"; if (!i) th.className = "sr-only"; hr.appendChild(th); });
    var thead = el("thead"); thead.appendChild(hr); tbl.appendChild(thead);
    var tb = el("tbody");
    function row(name, a, an, b, bn) {
      var tr = el("tr"); var h = el("th", null, name); h.scope = "row"; tr.appendChild(h);
      var c1 = el("td", "num cell-good"); c1.setAttribute("data-cell", an); c1.appendChild(el("b", null, str(a))); c1.appendChild(el("span", "cell-name", an.toUpperCase()));
      var c2 = el("td", "num cell-bad"); c2.setAttribute("data-cell", bn); c2.appendChild(el("b", null, str(b))); c2.appendChild(el("span", "cell-name", bn.toUpperCase()));
      tr.appendChild(c1); tr.appendChild(c2); tb.appendChild(tr);
    }
    row("Actually fake", c.tp, "tp", c.fn, "fn");
    row("Actually real", c.fp, "fp", c.tn, "tn");
    tbl.appendChild(tb);
    var wrap = el("div", "lab-scroll"); wrap.appendChild(tbl);
    return wrap;
  }
  function rocPlot(r) {
    var roc = r.roc;
    if (!isObj(roc) || !Array.isArray(roc.fpr) || !Array.isArray(roc.tpr) || roc.fpr.length !== roc.tpr.length || roc.fpr.length < 2) {
      return el("p", "empty", "No ROC curve is stored in this run.");
    }
    var W = 360, H = 340, L = 48, R = 14, T = 14, B = 44, pw = W - L - R, ph = H - T - B;
    function X(v) { return L + v * pw; }
    function Y(v) { return T + (1 - v) * ph; }
    var auc = isObj(r.metrics) && isObj(r.metrics.auc) ? r.metrics.auc.estimate : null;
    var s = svg("svg", { viewBox: "0 0 " + W + " " + H, "class": "roc", role: "img", id: "lab-roc",
      "aria-label": "ROC curve: true positive rate against false positive rate" + (num(auc) ? "; AUC " + auc.toFixed(4) : "") + "; " + roc.fpr.length + " points." });
    s.appendChild(svg("rect", { x: L, y: T, width: pw, height: ph, "class": "frame" }));
    [0, 0.25, 0.5, 0.75, 1].forEach(function (t) {
      s.appendChild(svg("line", { x1: X(t), y1: T, x2: X(t), y2: T + ph, "class": "grid" }));
      s.appendChild(svg("line", { x1: L, y1: Y(t), x2: L + pw, y2: Y(t), "class": "grid" }));
      var tx = svg("text", { x: X(t), y: T + ph + 16, "text-anchor": "middle" }); tx.textContent = t.toFixed(2).replace(/0$/, ""); s.appendChild(tx);
      var ty = svg("text", { x: L - 8, y: Y(t) + 4, "text-anchor": "end" }); ty.textContent = t.toFixed(2).replace(/0$/, ""); s.appendChild(ty);
    });
    s.appendChild(svg("line", { x1: X(0), y1: Y(0), x2: X(1), y2: Y(1), "class": "chance" }));
    s.appendChild(svg("polyline", { "class": "curve-line", points: roc.fpr.map(function (x, i) { return X(x).toFixed(1) + "," + Y(roc.tpr[i]).toFixed(1); }).join(" ") }));
    var c = r.confusion;
    if (isObj(c) && num(c.fp) && num(c.tn) && num(c.tp) && num(c.fn) && (c.fp + c.tn) > 0 && (c.tp + c.fn) > 0) {
      var ox = c.fp / (c.fp + c.tn), oy = c.tp / (c.tp + c.fn);
      var dot = svg("circle", { cx: X(ox), cy: Y(oy), r: 5, "class": "op" });
      dot.appendChild(svg("title", {}));
      dot.firstChild.textContent = "Operating point at the threshold";
      s.appendChild(dot);
    }
    var xl = svg("text", { x: L + pw / 2, y: H - 6, "text-anchor": "middle" }); xl.textContent = "False positive rate"; s.appendChild(xl);
    var yl = svg("text", { x: 12, y: T + ph / 2, "text-anchor": "middle", transform: "rotate(-90 12 " + (T + ph / 2) + ")" }); yl.textContent = "True positive rate"; s.appendChild(yl);
    var fig = el("figure", "lab-fig");
    fig.appendChild(s);
    fig.appendChild(el("figcaption", null, "ROC curve drawn from the stored points. The dashed line is chance; the dot is the operating point at the threshold used."));
    return fig;
  }
  function breakdownTable(bd) {
    if (!isObj(bd) || !Array.isArray(bd.rows) || !bd.rows.length) return null;
    var tbl = el("table", "lab-t");
    tbl.id = "lab-breakdown";
    tbl.appendChild(el("caption", null, "Breakdown by " + str(bd.column)));
    var hr = el("tr");
    [str(bd.column), "Fake", "Real", "AUC", "False positive rate", "False negative rate"].forEach(function (c) { var th = el("th", null, c); th.scope = "col"; hr.appendChild(th); });
    var thead = el("thead"); thead.appendChild(hr); tbl.appendChild(thead);
    var tb = el("tbody");
    bd.rows.forEach(function (x) {
      var tr = el("tr");
      var th = el("th", null, x.value); th.scope = "row"; tr.appendChild(th);
      tr.appendChild(el("td", "num", x.n_fake)); tr.appendChild(el("td", "num", x.n_real));
      tr.appendChild(el("td", "num", f4(x.auc))); tr.appendChild(el("td", "num", f4(x.fpr))); tr.appendChild(el("td", "num", f4(x.fnr)));
      tb.appendChild(tr);
    });
    tbl.appendChild(tb);
    var wrap = el("div", "lab-scroll"); wrap.appendChild(tbl);
    var out = el("div");
    out.appendChild(wrap);
    out.appendChild(el("p", "lab-hint", "Not defined means the group has only one class or too few samples for that measure. Small groups give unstable numbers."));
    return out;
  }
  function renderResults(r) {
    setStep(4);
    var sec = $("lab-results"), b = body("lab-results");
    sec.hidden = false; clear(b);
    var cfg = isObj(r.config) ? r.config : {};
    var role = str(r.role);
    var final = roleClass(cfg.split) === "final" || (isObj(r.final_guard) && r.final_guard.required === true);
    var head = el("div", "lab-rhead");
    head.appendChild(el("p", "mono", "Run " + str(r.id)));
    if (cfg.label) head.appendChild(el("p", "lab-rlabel", cfg.label));
    if (final) {
      var fb = el("span", "role-badge role-final", "Final held-out, re-analysis of a logged evaluation");
      fb.id = "lab-final-badge";
      head.appendChild(fb);
    } else {
      head.appendChild(el("span", "role-badge role-" + roleClass(cfg.split), roleClass(cfg.split) === "validation" ? "Validation" : "Development"));
    }
    b.appendChild(head);

    var d = isObj(r.data) ? r.data : {};
    var intro = el("div", "lab-note-box");
    var lines = [];
    lines.push("Evaluated: " + str(cfg.dataset) + " / " + str(cfg.split) + ", " + str(d.n_samples) + " samples (" + str(d.n_fake) + " fake, " + str(d.n_real) + " real) in " + str(d.n_groups) + " groups, " + (Array.isArray(cfg.run_ids) ? cfg.run_ids.length : "") + " run(s) of iteration " + str(cfg.iteration) + ".");
    if (final) {
      var led = isObj(r.final_guard) && Array.isArray(r.final_guard.ledger_entries) ? r.final_guard.ledger_entries : [];
      lines.push("These rows are sealed. This is a re-analysis of an evaluation that is already logged" + (led.length ? " (ledger: " + led.join(", ") + ")" : "") +
        ", not a new test, and it changes no ledger entry.");
    }
    else if (isObj(r.threshold) && r.threshold.fitted_on_these_rows) lines.push("The threshold was fitted on these same rows, so threshold-dependent numbers are optimistic. AUC and EER do not depend on it.");
    else lines.push("The threshold was not fitted on these rows.");
    lines.push(role ? "Role of the block: " + role + "." : "");
    lines.forEach(function (t) { if (t) intro.appendChild(el("p", null, t)); });
    b.appendChild(intro);

    if (isObj(r.threshold)) {
      var t = r.threshold;
      var tb = el("div", "lab-thr-box"); tb.id = "lab-threshold";
      tb.appendChild(el("h3", null, "Threshold and its provenance"));
      tb.appendChild(kv([["Value", num(t.value) ? t.value.toFixed(4) : ""], ["Source", t.source], ["Provenance", t.provenance],
        ["Fitted on these rows", t.fitted_on_these_rows === true ? "yes (optimistic)" : (t.fitted_on_these_rows === false ? "no" : "")]]));
      b.appendChild(tb);
    }
    if (Array.isArray(d.warnings) && d.warnings.length) {
      var wb = el("div", "lab-warnings"); wb.id = "lab-warnings"; wb.setAttribute("role", "note");
      wb.appendChild(el("h3", null, "Warnings (" + d.warnings.length + ")"));
      var wu = el("ul"); d.warnings.forEach(function (w) { wu.appendChild(el("li", null, isObj(w) ? (str(w.code) + ": " + str(w.message)) : w)); });
      wb.appendChild(wu); b.appendChild(wb);
    }
    b.appendChild(el("h3", "lab-sub", "Metrics"));
    b.appendChild(metricsTable(r));
    var two = el("div", "lab-two");
    var l = el("div"); l.appendChild(el("h3", "lab-sub", "Confusion matrix"));
    if (isObj(r.confusion)) l.appendChild(confusion(r.confusion)); else l.appendChild(el("p", "empty", "No confusion matrix is stored."));
    two.appendChild(l);
    var rr = el("div"); rr.appendChild(el("h3", "lab-sub", "ROC curve")); rr.appendChild(rocPlot(r)); two.appendChild(rr);
    b.appendChild(two);
    var bd = breakdownTable(r.breakdown);
    if (bd) { b.appendChild(el("h3", "lab-sub", "Breakdown")); b.appendChild(bd); }
    else b.appendChild(el("p", "lab-hint", "No breakdown is stored for this run (the score file has no source or generator column)."));

    var sk = Array.isArray(d.skipped) ? d.skipped : [];
    var skd = el("details", "lab-skipped"); skd.id = "lab-skipped";
    skd.appendChild(el("summary", null, "Skipped rows (" + sk.length + ")"));
    if (sk.length) {
      var su = el("ul");
      sk.slice(0, 200).forEach(function (x) { su.appendChild(el("li", null, str(isObj(x) ? x.sample_id : x) + (isObj(x) && x.reason ? ": " + str(x.reason) : ""))); });
      skd.appendChild(su);
      if (sk.length > 200) skd.appendChild(el("p", "lab-hint", "Showing the first 200 of " + sk.length + ". The full list is in the JSON export."));
    } else skd.appendChild(el("p", "empty", "No rows were skipped."));
    b.appendChild(skd);

    var tech = el("details", "lab-tech");
    tech.appendChild(el("summary", null, "Reproducibility and checks"));
    var lk = isObj(d.leakage) ? d.leakage : null;
    tech.appendChild(kv([["Configuration SHA-256", r.config_sha256, "mono"], ["Score file", isObj(r.score_file) ? r.score_file.path : "", "mono"],
      ["Score file SHA-256", isObj(r.score_file) ? r.score_file.sha256 : "", "mono"], ["Rows in file", isObj(r.score_file) ? r.score_file.n_rows : ""],
      ["Purpose", r.purpose], ["Ledger entries", isObj(r.final_guard) && Array.isArray(r.final_guard.ledger_entries) ? r.final_guard.ledger_entries.join(", ") : ""],
      ["Run time", num(r.runtime_s) ? r.runtime_s.toFixed(1) + " s" : ""], ["Python", isObj(r.environment) ? r.environment.python : ""],
      ["Code version", isObj(r.environment) ? r.environment.git_head : "", "mono"],
      ["Uncommitted entries", isObj(r.environment) ? r.environment.dirty_entries : ""], ["Created", r.created], ["Finished", r.finished]]));
    if (lk) { tech.appendChild(el("h4", "sec-h", "Leakage checks")); tech.appendChild(el("pre", "lab-pre mono", JSON.stringify(lk, null, 1))); }
    b.appendChild(tech);
    b.appendChild(el("p", "lab-limits", "What this does not tell you: the lab re-analyses recorded scores. It cannot confirm that the model never saw these samples in training unless the score file records its training data, and a number here is not an approval of any model for use."));
    renderCompare(r);
  }

  /* ---------------- step 5: compare and export ---------------- */
  function compareValue(v) {
    if (isObj(v) && ("estimate" in v)) return f4(v.estimate) + (num(v.low) && num(v.high) ? " " + interval(v) : "");
    if (num(v)) return f4(v);
    if (typeof v === "string") return v;
    return JSON.stringify(v);
  }
  function renderCompare(r) {
    setStep(5);
    var sec = $("lab-compare"), b = body("lab-compare");
    sec.hidden = false; clear(b);
    var ex = el("div", "lab-export"); ex.id = "lab-export";
    ex.appendChild(el("span", "lab-exl", "Export this run:"));
    [["json", "JSON (full record)"], ["md", "Markdown report"]].forEach(function (f) {
      var a = el("a", "btn", f[1]);
      a.href = "/api/evallab/runs/" + encodeURIComponent(str(r.id)) + "/export?format=" + f[0];
      a.setAttribute("download", "");
      a.setAttribute("data-format", f[0]);
      ex.appendChild(a);
    });
    b.appendChild(ex);
    b.appendChild(el("h3", "lab-sub", "Compare with a baseline run"));
    b.appendChild(el("p", "lab-hint", "A comparison is possible only when both runs cover exactly the same samples, labels and groups. The lab then resamples the groups together (a paired bootstrap). Otherwise it says why it cannot compare; it never invents a baseline."));
    var box = el("div", "lab-cmp"); box.id = "lab-cmp-body";
    b.appendChild(box);
    box.appendChild(el("p", "lab-loading", "Loading earlier runs ..."));
    api("GET", "/api/evallab/runs").then(function (x) {
      clear(box);
      var list = x.ok ? (Array.isArray(x.j) ? x.j : (isObj(x.j) && Array.isArray(x.j.runs) ? x.j.runs : [])) : [];
      S.runs = list;
      if (!x.ok) { box.appendChild(el("p", "empty", "The list of earlier runs could not be loaded: " + errMsg(x))); return; }
      var others = list.filter(function (q) { return isObj(q) && q.id !== r.id && (q.state === "done" || q.state === undefined); });
      if (!others.length) {
        box.appendChild(el("p", "empty", "No other finished run is available as a baseline. Run a second evaluation on the same block (for example another iteration) first."));
        return;
      }
      var sel = el("select"); sel.id = "lab-baseline";
      others.forEach(function (q) {
        var c = isObj(q.config) ? q.config : q;
        var o = el("option", null, str(q.id) + (c.label ? " (" + c.label + ")" : "") + (c.dataset ? ": " + c.dataset + " / " + c.split + ", " + c.iteration + " " + str(c.modality) : ""));
        o.value = q.id; sel.appendChild(o);
      });
      var ms = el("select"); ms.id = "lab-cmp-metric";
      Object.keys(isObj(r.metrics) ? r.metrics : {}).forEach(function (k) { var o = el("option", null, (METRICS[k] || [k])[0]); o.value = k; ms.appendChild(o); });
      if (ms.querySelector("option[value=auc]")) ms.value = "auc";
      var row = el("div", "lab-grid");
      row.appendChild(field("Baseline run", sel, "", "lab-baseline"));
      row.appendChild(field("Metric", ms, "", "lab-cmp-metric"));
      box.appendChild(row);
      var go = el("button", "btn primary", "Compare"); go.type = "button"; go.id = "lab-cmp-btn";
      var out = el("div", "lab-cmp-out"); out.id = "lab-cmp-out"; out.setAttribute("aria-live", "polite");
      go.addEventListener("click", function () {
        clear(out); out.appendChild(el("p", "lab-loading", "Comparing ..."));
        api("POST", "/api/evallab/compare", { run_a: r.id, run_b: sel.value, metric: ms.value }).then(function (y) {
          clear(out);
          var j = isObj(y.j) ? y.j : null;
          if (!y.ok && !(j && "available" in j)) { out.appendChild(el("p", "lab-unavail", "Comparison failed: " + errMsg(y))); return; }
          if (!j || j.available !== true) {
            var u = el("div", "lab-unavail"); u.id = "lab-unavailable";
            u.appendChild(el("strong", null, "Comparison unavailable. "));
            u.appendChild(document.createTextNode(str(j && j.reason) || "The server gave no reason."));
            out.appendChild(u);
            return;
          }
          var mname = (METRICS[ms.value] || [ms.value])[0];
          var ok = el("div", "lab-cmp-ok"); ok.id = "lab-cmp-result";
          var rows = [["Run A (this run)", r.id, "mono"], ["Run B (baseline)", sel.value, "mono"], ["Metric", mname]];
          if (isObj(j.a)) rows.push(["A: " + mname, compareValue(j.a)]);
          if (isObj(j.b)) rows.push(["B: " + mname, compareValue(j.b)]);
          if (isObj(j.difference)) rows.push(["Difference A minus B", compareValue(j.difference)]);
          if (isObj(j.block)) rows.push(["Samples compared", str(j.block.n_samples) + " in " + str(j.block.n_groups) + " groups (" + str(j.block.dataset) + " / " + str(j.block.split) + ")"]);
          if (isObj(j.protocol)) rows.push(["Protocol", str(j.protocol.rule) + (num(j.protocol.n_boot) ? "; " + j.protocol.n_boot + " resamples" : "") + (num(j.protocol.level) ? "; interval level " + j.protocol.level : "")]);
          var verdict = "Paired comparison on identical samples.";
          if (isObj(j.difference) && typeof j.difference.contains_zero === "boolean") {
            verdict += j.difference.contains_zero ? " The interval of the difference includes 0, so these data do not separate the two runs."
              : " The interval of the difference excludes 0 on these data; it says nothing about other data.";
          }
          ok.appendChild(el("p", null, verdict));
          ok.appendChild(kv(rows));
          if (Array.isArray(j.notes) && j.notes.length) {
            var nl = el("ul", "lab-notes"); j.notes.forEach(function (q) { nl.appendChild(el("li", null, q)); }); ok.appendChild(nl);
          }
          out.appendChild(ok);
        }).catch(function (e) { clear(out); out.appendChild(el("p", "lab-unavail", "Comparison failed: " + (e && e.message ? e.message : "request failed"))); });
      });
      box.appendChild(go);
      box.appendChild(out);
    }).catch(function (e) { clear(box); box.appendChild(el("p", "empty", "The list of earlier runs could not be loaded: " + (e && e.message ? e.message : "request failed"))); });
  }

  function open(host) {
    root = host || root || $("lab-root");
    if (!built) { build(); loadCatalogue(); }
    else if (!S.cat) loadCatalogue();
    var h = $("lab-h");
    if (h) { h.setAttribute("tabindex", "-1"); }
  }
  window.evallab = { open: open, state: S, metricLabels: METRICS };
})();
