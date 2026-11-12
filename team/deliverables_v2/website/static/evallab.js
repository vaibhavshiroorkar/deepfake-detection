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
  var G = { data: null, ds: "", group: "", confirmFull: false, token: 0, busy: false, run: null };
  var R = { data: null, filter: "all", open: "" };

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
    hero.appendChild(el("p", "lede", "Start with the recommended demonstrations: logged results of the project's detectors, with sample counts and limitations. To evaluate yourself, choose a dataset and one evaluation group of it, run, and read the result with its uncertainty. The lab re-analyses recorded scores; it never runs a model."));
    var how = el("ul", "lab-explain");
    [["What data", "Score files recorded by earlier evaluations: one row per sample with its label, group and score. Nothing is uploaded and no model is run."],
     ["Is it unseen?", "Each block shows its role. Development and validation blocks may have been used to tune a model; only a final held-out block is sealed, and the lab opens it only to re-analyse an evaluation that is already logged."],
     ["What the numbers mean", "Every metric has an interval from a bootstrap that resamples whole groups (a person or source), so near-identical samples do not count twice. A wide interval means the block is too small to say much."],
     ["What it cannot tell you", "A score file may not record what the model was trained on, so the lab checks only what the file records (groups, splits, hashes). A good number here is not an approval of any model."]
    ].forEach(function (x) {
      var li = el("li");
      li.appendChild(el("b", null, x[0] + (/[?.!]$/.test(x[0]) ? " " : ". ")));
      li.appendChild(document.createTextNode(x[1]));
      how.appendChild(li);
    });
    hero.appendChild(how);
    root.appendChild(hero);

    var reco = el("section", "panel lab-section lab-reco");
    reco.id = "lab-reco";
    reco.setAttribute("aria-labelledby", "lab-reco-h");
    var rh = el("h2", null, "Recommended demonstrations");
    rh.id = "lab-reco-h";
    reco.appendChild(rh);
    reco.appendChild(el("div", "lab-body"));
    root.appendChild(reco);

    var guided = el("section", "panel lab-section lab-guided");
    guided.id = "lab-guided";
    guided.setAttribute("aria-labelledby", "lab-guided-h");
    var gh = el("h2", null, "Evaluate a dataset");
    gh.id = "lab-guided-h";
    guided.appendChild(gh);
    guided.appendChild(el("div", "lab-body"));
    root.appendChild(guided);

    var adv = el("details", "panel lab-advanced");
    adv.id = "lab-advanced";
    adv.appendChild(el("summary", null, "Advanced: re-analyse stored score files"));
    var advBody = el("div", "lab-adv-body");
    adv.appendChild(advBody);
    root.appendChild(adv);
    var savedRoot = root;
    root = advBody;
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
    root = savedRoot;
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

  /* ---------------- guided flow: evaluate a dataset (post-freeze evaluation groups) ----------------
     Reads GET /api/evallab/groups and starts runs with POST /api/evallab/group-runs. Nothing is started by opening the lab or by choosing a
     dataset or a group; a run starts only from the "Run evaluation on this group" button, or from the confirmed whole-pool action. */
  var STATUS_FALLBACK = "Experimental evaluation output. Evaluating a model on a dataset does not approve it for deployment.";
  var SPLIT_TEXT = { post_x: "post-freeze evaluation on previously unread data",
    test_x: "sealed cross-dataset test, read once at the freeze (E-014); stored scores re-analysed, nothing scored again",
    test_in: "sealed in-domain test, read once at the freeze (E-014); stored scores re-analysed, nothing scored again" };
  var SEALED_RESULT = "Stored sealed-test scores re-analysed: the frozen checkpoint scored this sealed split once, at the freeze (E-014). Nothing was scored again now; the metrics were recomputed from those logged scores for the chosen group. This is not a new held-out test and not an approval.";
  function thrText(th) {
    if (!isObj(th)) return "";
    if (num(th.value)) return th.value.toFixed(4);
    if (th.source === "val_fit") return "one per seed, fitted on validation (D-024)";
    return "";
  }
  var HEADLINE = [["auc", "AUROC"], ["f1", "F1"], ["precision", "Precision"], ["recall", "Recall (sensitivity)"], ["specificity", "Specificity"], ["accuracy", "Accuracy"]];
  var MORE = [["eer", "EER"], ["fpr", "False positive rate"], ["fnr", "False negative rate"], ["balanced_accuracy", "Balanced accuracy"]];
  var GROUP_MODALITY = { visual: "Face video", audio: "Speech / audio", image: "Image" };

  function gbody() { return body("lab-guided"); }
  function curDataset() { return G.data && G.data.datasets.filter(function (d) { return d.dataset === G.ds; })[0] || null; }
  function pct(x) { return num(x) ? (100 * x).toFixed(1) + "%" : ""; }
  function plural(n, one, many) { return n + " " + (n === 1 ? one : many); }
  function noteBanner(text, id) {
    var b = el("div", "lab-banner", text);
    if (id) b.id = id;
    b.setAttribute("role", "note");
    return b;
  }
  function loadGroups() {
    var b = gbody();
    clear(b);
    b.appendChild(el("p", "lab-loading", "Looking for evaluation datasets ..."));
    return api("GET", "/api/evallab/groups").then(function (x) {
      clear(b);
      if (!x.ok || !isObj(x.j)) {
        var box = el("div", "error-box");
        box.id = "ev-load-error";
        box.setAttribute("role", "alert");
        box.appendChild(el("strong", null, "The evaluation datasets could not be loaded. "));
        box.appendChild(document.createTextNode(errMsg(x)));
        b.appendChild(box);
        var again = el("button", "btn", "Try again");
        again.type = "button";
        again.addEventListener("click", loadGroups);
        b.appendChild(again);
        return;
      }
      G.data = x.j;
      if (!Array.isArray(G.data.datasets)) G.data.datasets = [];
      renderGuided();
    }).catch(function (e) {
      clear(b);
      var box = el("div", "error-box", "The Evaluation Lab API could not be reached: " + (e && e.message ? e.message : "request failed"));
      box.setAttribute("role", "alert");
      b.appendChild(box);
    });
  }
  function renderGuided() {
    var b = gbody();
    clear(b);
    var ds = G.data.datasets;
    if (!ds.length) {
      var em = el("div", "lab-empty");
      em.id = "ev-empty";
      em.appendChild(el("h3", null, "No evaluation groups are installed"));
      em.appendChild(el("p", null, "No post-freeze evaluation groups were found on this computer, so there is nothing to evaluate here and no example data is invented. The advanced section below can still re-analyse stored score files."));
      b.appendChild(em);
      return;
    }
    b.appendChild(el("p", "lab-hint", "Choose a dataset, then one evaluation group of it. Nothing runs until you press the run button."));
    var sel = el("select");
    sel.id = "ev-dataset";
    var ph = el("option", null, "Choose a dataset ...");
    ph.value = "";
    sel.appendChild(ph);
    ds.forEach(function (d) {
      var mod = GROUP_MODALITY[d.modality];
      var o = el("option", null, (mod ? mod + ": " : "") + (str(d.title) || str(d.dataset)));
      o.value = str(d.dataset); sel.appendChild(o);
    });
    sel.value = G.ds && curDataset() ? G.ds : "";
    sel.addEventListener("change", function () {
      G.ds = sel.value; G.group = ""; G.confirmFull = false; G.token++; G.busy = false; G.run = null;
      renderDataset();
    });
    b.appendChild(field("Dataset", sel, "Each dataset comes with evaluation groups that were fixed before any score was read.", "ev-dataset"));
    var area = el("div", "ev-area");
    area.id = "ev-area";
    b.appendChild(area);
    var out = el("div", "ev-out");
    out.id = "ev-out";
    out.setAttribute("aria-live", "polite");
    b.appendChild(out);
    renderDataset();
  }
  function groupCard(d, g) {
    var on = G.group === g.id;
    var lab = el("label", "pick ev-group" + (on ? " on" : ""));
    lab.setAttribute("data-group", str(g.id));
    var r = el("input");
    r.type = "radio"; r.name = "ev-group"; r.value = str(g.id); r.checked = on;
    r.addEventListener("change", function () { G.group = str(g.id); G.confirmFull = false; renderDataset(); });
    var t = el("span", "pick-body");
    var top = el("span", "pick-top");
    top.appendChild(el("b", "pick-title", str(g.id) + ": " + str(g.title)));
    top.appendChild(el("span", "role-badge ev-kind ev-kind-" + (g.kind === "diagnostic" ? "diagnostic" : "general"), g.kind === "diagnostic" ? "Diagnostic" : "General"));
    t.appendChild(top);
    if (g.description) t.appendChild(el("span", "pick-sub ev-desc", g.description));
    t.appendChild(el("span", "pick-sub ev-counts", g.n_samples + " samples: " + g.n_real + " real, " + g.n_fake + " fake"));
    if (num(g.n_groups)) t.appendChild(el("span", "pick-sub ev-units", "Independent groups: " + plural(g.n_groups, str(g.group_unit) || "group", (str(g.group_unit) || "group") + "s")));
    if (g.overlap) t.appendChild(el("span", "pick-sub ev-overlap", "Overlap: " + g.overlap));
    if (num(g.n_missing_scores) && g.n_missing_scores > 0) t.appendChild(el("span", "pick-sub ev-missing", g.n_missing_scores + " declared sample(s) have no stored score and are left out."));
    lab.appendChild(r);
    lab.appendChild(t);
    return lab;
  }
  function renderDataset() {
    var area = $("ev-area");
    clear(area);
    var out = $("ev-out");
    if (out) clear(out);
    var d = curDataset();
    if (!d) return;
    area.appendChild(noteBanner(str(d.status) || STATUS_FALLBACK, "ev-status-banner"));
    var th = isObj(d.threshold) ? d.threshold : {};
    var sum = el("div", "lab-summary");
    sum.id = "ev-summary";
    sum.appendChild(el("h3", null, str(d.title)));
    sum.appendChild(kv([["Experiment", d.experiment], ["Head", d.head], ["Modality", d.modality],
      ["Split", str(d.split) + (SPLIT_TEXT[d.split] ? ": " + SPLIT_TEXT[d.split] : "")], ["Eligibility", d.eligibility],
      [d.sealed_reanalysis ? "Threshold" : "Fixed threshold", thrText(th)], ["Threshold provenance", th.provenance], ["Score kind", d.score_kind]]));
    if (Array.isArray(d.caveats) && d.caveats.length) {
      var cv = el("details", "lab-caveats");
      cv.open = true;
      cv.appendChild(el("summary", null, "Caveats (" + d.caveats.length + ")"));
      var cu = el("ul");
      d.caveats.forEach(function (c) { cu.appendChild(el("li", null, c)); });
      cv.appendChild(cu);
      sum.appendChild(cv);
    }
    area.appendChild(sum);
    area.appendChild(el("h3", "lab-sub", "Evaluation groups"));
    var list = el("div", "lab-blocks ev-groups");
    list.id = "ev-groups";
    list.setAttribute("role", "radiogroup");
    list.setAttribute("aria-label", "Evaluation groups");
    (Array.isArray(d.groups) ? d.groups : []).forEach(function (g) { list.appendChild(groupCard(d, g)); });
    if (!list.children.length) list.appendChild(el("p", "empty", "This dataset lists no evaluation groups."));
    area.appendChild(list);
    var go = el("button", "btn primary", "Run evaluation on this group");
    go.type = "button"; go.id = "ev-run-btn";
    go.disabled = !G.group || G.busy;
    go.addEventListener("click", function () { startGroupRun(G.group, false); });
    area.appendChild(go);
    if (isObj(d.full)) renderFull(area, d);
  }
  function renderFull(area, d) {
    var f = d.full;
    var box = el("div", "ev-full");
    box.id = "ev-full";
    box.appendChild(el("h3", "lab-sub", "Whole eligible pool"));
    box.appendChild(el("p", "lab-hint", "Secondary action. The whole pool is not an evaluation group and is never chosen by default."));
    if (!G.confirmFull) {
      var open = el("button", "btn", "Evaluate the whole eligible pool");
      open.type = "button"; open.id = "ev-full-btn";
      open.disabled = G.busy;
      open.addEventListener("click", function () { G.confirmFull = true; renderDataset(); var c = $("ev-full-confirm-btn"); if (c) c.focus(); });
      box.appendChild(open);
    } else {
      var cf = el("div", "ev-confirm");
      cf.id = "ev-full-confirm";
      cf.setAttribute("role", "group");
      cf.setAttribute("aria-label", "Confirm the whole-pool evaluation");
      cf.appendChild(el("p", null, "This evaluates all " + str(f.n_samples) + " samples (" + str(f.n_real) + " real, " + str(f.n_fake) + " fake)."));
      if (f.warning) cf.appendChild(el("p", "ev-full-warning", f.warning));
      if (f.estimate) cf.appendChild(el("p", "ev-full-estimate", f.estimate));
      var yes = el("button", "btn primary", "Confirm and evaluate the whole pool");
      yes.type = "button"; yes.id = "ev-full-confirm-btn";
      yes.disabled = G.busy;
      yes.addEventListener("click", function () { startGroupRun("FULL", true); });
      var no = el("button", "btn", "Cancel");
      no.type = "button"; no.id = "ev-full-cancel-btn";
      no.addEventListener("click", function () { G.confirmFull = false; renderDataset(); });
      cf.appendChild(yes); cf.appendChild(no);
      box.appendChild(cf);
    }
    area.appendChild(box);
  }
  function evButtons(disabled) {
    var go = $("ev-run-btn"); if (go) go.disabled = disabled || !G.group;
    ["ev-full-btn", "ev-full-confirm-btn"].forEach(function (id) { var b = $(id); if (b) b.disabled = disabled; });
  }
  function evFail(err) {
    G.busy = false;
    var out = $("ev-out"); clear(out);
    var box = el("div", "error-box");
    box.id = "ev-error";
    box.setAttribute("role", "alert");
    box.appendChild(el("strong", null, "The evaluation did not complete. "));
    box.appendChild(document.createTextNode(str(err.code) + ": " + str(err.message)));
    out.appendChild(box);
    evButtons(false);
  }
  function startGroupRun(group, full) {
    if (G.busy || !G.ds || !group) return;
    G.busy = true;
    var token = ++G.token;
    evButtons(true);
    var out = $("ev-out"); clear(out);
    var st = el("div", "lab-status"); st.id = "ev-progress"; st.setAttribute("role", "status");
    st.textContent = "Submitting the evaluation ...";
    out.appendChild(st);
    var payload = { dataset: G.ds, group: group };
    if (full) payload.confirm_full = true;
    api("POST", "/api/evallab/group-runs", payload).then(function (x) {
      if (token !== G.token) return;
      var rec = isObj(x.j) ? x.j : null;
      var id = rec && (rec.id || rec.run_id);
      if (rec && rec.state === "refused") { evFail(isObj(rec.error) ? rec.error : { code: "REFUSED", message: "The lab refused this evaluation." }); return; }
      if (!x.ok || !id) { evFail(isObj(x.j) && isObj(x.j.error) ? x.j.error : { code: "HTTP_" + x.status, message: "The server did not accept the evaluation." }); return; }
      gpoll(str(id), token);
    }).catch(function (e) { if (token === G.token) evFail({ code: "NETWORK", message: e && e.message ? e.message : "request failed" }); });
  }
  function gpoll(id, token) {
    if (token !== G.token) return;
    api("GET", "/api/evallab/runs/" + encodeURIComponent(id)).then(function (x) {
      if (token !== G.token) return;
      if (!x.ok || !isObj(x.j)) { evFail({ code: "HTTP_" + x.status, message: "Could not read the run." }); return; }
      var r = x.j;
      if (r.state === "done") { G.busy = false; G.run = r; renderGroupResults(r); return; }
      if (r.state === "failed" || r.state === "refused") { evFail(isObj(r.error) ? r.error : { code: "FAILED", message: "The evaluation failed without details." }); return; }
      var out = $("ev-out"); clear(out);
      var p = isObj(r.progress) ? r.progress : {};
      var st = el("div", "lab-status"); st.id = "ev-progress"; st.setAttribute("role", "status");
      st.textContent = (r.state === "queued" ? "Queued" : "Running") + (p.stage ? ": " + p.stage : "") + (num(p.total) && p.total > 0 ? " (" + p.done + " of " + p.total + ")" : "");
      out.appendChild(st);
      if (num(p.total) && p.total > 0) {
        var bar = el("div", "lab-bar"); bar.setAttribute("role", "progressbar"); bar.setAttribute("aria-valuemin", "0");
        bar.setAttribute("aria-valuemax", String(p.total)); bar.setAttribute("aria-valuenow", String(p.done || 0));
        bar.setAttribute("aria-label", "Evaluation progress");
        var fill = el("span"); fill.style.width = Math.max(0, Math.min(100, 100 * (p.done || 0) / p.total)) + "%";
        bar.appendChild(fill); out.appendChild(bar);
      }
      setTimeout(function () { gpoll(id, token); }, POLL_MS);
    }).catch(function (e) { if (token === G.token) evFail({ code: "NETWORK", message: e && e.message ? e.message : "request failed" }); });
  }
  function stripIds(node) {
    if (node.removeAttribute) node.removeAttribute("id");
    Array.prototype.forEach.call(node.querySelectorAll ? node.querySelectorAll("[id]") : [], function (n) { n.removeAttribute("id"); });
    return node;
  }
  function headlineTile(r, key, label) {
    var m = isObj(r.metrics) ? r.metrics[key] : null;
    var t = el("div", "ev-tile");
    t.setAttribute("data-metric", key);
    t.appendChild(el("span", "ev-tile-name", label));
    t.appendChild(el("b", "ev-tile-val", isObj(m) ? f4(m.estimate) : "not defined"));
    t.appendChild(el("span", "ev-tile-int", isObj(m) ? "95% interval " + interval(m) : "no interval"));
    return t;
  }
  function renderGroupResults(r) {
    var out = $("ev-out"); clear(out);
    var cfg = isObj(r.config) ? r.config : {};
    var eg = isObj(r.eval_group) ? r.eval_group : {};
    var d = isObj(r.data) ? r.data : {};
    var th = isObj(r.threshold) ? r.threshold : {};
    var ds = curDataset();
    var done = el("p", "lab-status done", "Finished: run " + str(r.id) + ".");
    done.id = "ev-done";
    done.setAttribute("role", "status");
    out.appendChild(done);
    var box = el("div", "ev-results");
    box.id = "ev-results";
    var sealedRun = cfg.purpose === "final_reanalysis";
    box.appendChild(noteBanner(sealedRun ? SEALED_RESULT : STATUS_FALLBACK, "ev-result-banner"));
    box.appendChild(el("h3", null, "Result: " + str(eg.title || (ds && ds.title) || cfg.dataset) + (eg.id ? " (" + str(eg.id) + ")" : "")));
    var tiles = el("div", "ev-tiles");
    tiles.id = "ev-headline";
    HEADLINE.forEach(function (h) {
      var t = headlineTile(r, h[0], h[1]);
      if (h[0] === "accuracy") {
        var share = num(d.class_balance) ? d.class_balance : (d.n_samples ? d.n_fake / d.n_samples : null);
        if (num(share)) t.appendChild(el("span", "ev-tile-note", "class balance: " + pct(share) + " fake"));
      }
      tiles.appendChild(t);
    });
    box.appendChild(tiles);
    box.appendChild(kv([["Samples", d.n_samples], ["Real", d.n_real], ["Fake", d.n_fake],
      ["Independent groups", d.n_groups !== undefined && d.n_groups !== null ? d.n_groups + (eg.group_unit ? " (" + eg.group_unit + ")" : "") : ""],
      ["Threshold", num(th.value) ? th.value.toFixed(4) + " (" + str(th.source) + ")" : (th.source === "val_fit" ? "one per seed, fitted on validation (val_fit)" : "")], ["Threshold provenance", th.provenance],
      ["Experiment", eg.experiment || (ds && ds.experiment)], ["Dataset", cfg.dataset], ["Split", cfg.split]]));
    box.appendChild(el("p", "lab-hint ev-scorekind", (sealedRun
      ? "Mean over the three frozen seeds, each judged at its own threshold fitted on the validation split before the sealed split was read (the rule E-014 used); nothing was tuned on these samples."
      : "These are ranking scores judged at one fixed threshold taken from the checkpoint; the threshold was not tuned on these samples.") + (eg.score_kind ? " Score kind: " + eg.score_kind : "")));

    var more = el("details", "ev-more");
    more.id = "ev-more";
    more.appendChild(el("summary", null, "More metrics"));
    var mt = el("table", "lab-t");
    mt.appendChild(el("caption", null, "Further metrics with 95% bootstrap intervals (whole groups resampled)"));
    var hr = el("tr");
    ["Metric", "Estimate", "Interval"].forEach(function (c) { var h = el("th", null, c); h.scope = "col"; hr.appendChild(h); });
    var mh = el("thead"); mh.appendChild(hr); mt.appendChild(mh);
    var mb = el("tbody");
    MORE.forEach(function (k) {
      var m = isObj(r.metrics) ? r.metrics[k[0]] : null;
      var tr = el("tr"); tr.setAttribute("data-metric", k[0]);
      var h = el("th", null, k[1]); h.scope = "row"; tr.appendChild(h);
      tr.appendChild(el("td", "num", isObj(m) ? f4(m.estimate) : "not defined"));
      tr.appendChild(el("td", "num", isObj(m) ? interval(m) : "no interval"));
      mb.appendChild(tr);
    });
    mt.appendChild(mb);
    var mw = el("div", "lab-scroll"); mw.appendChild(mt);
    more.appendChild(mw);
    box.appendChild(more);

    var cr = el("details", "ev-cr");
    cr.id = "ev-cr";
    cr.appendChild(el("summary", null, "Confusion matrix and ROC"));
    var two = el("div", "lab-two");
    var l = el("div");
    if (isObj(r.confusion)) l.appendChild(stripIds(confusion(r.confusion))); else l.appendChild(el("p", "empty", "No confusion matrix is stored."));
    two.appendChild(l);
    var rr = el("div"); rr.appendChild(stripIds(rocPlot(r))); two.appendChild(rr);
    cr.appendChild(two);
    box.appendChild(cr);

    var wr = Array.isArray(d.warnings) ? d.warnings : [];
    var wd = el("details", "ev-warnings");
    wd.id = "ev-warnings";
    wd.appendChild(el("summary", null, "Warnings (" + wr.length + ")"));
    if (wr.length) { var wu = el("ul"); wr.forEach(function (w) { wu.appendChild(el("li", null, isObj(w) ? (str(w.code) + ": " + str(w.message)) : w)); }); wd.appendChild(wu); }
    else wd.appendChild(el("p", "empty", "No warnings were recorded."));
    box.appendChild(wd);

    var pv = el("details", "ev-prov");
    pv.id = "ev-prov";
    pv.appendChild(el("summary", null, "Provenance and configuration"));
    var pr = isObj(r.protocol) ? r.protocol : {};
    pv.appendChild(kv([["Group", str(eg.id) + (eg.title ? ": " + eg.title : "")], ["Kind", eg.kind], ["Description", eg.description], ["Overlap", eg.overlap],
      ["Group unit", eg.group_unit], ["Declared samples", eg.n_declared], ["Declared samples without a score", d.group_missing_scores],
      ["Groups file", eg.groups_file, "mono"], ["Groups file SHA-256", eg.groups_sha256, "mono"], ["Descriptor", eg.descriptor, "mono"],
      ["Score file", isObj(r.score_file) ? r.score_file.path : "", "mono"], ["Score file SHA-256", isObj(r.score_file) ? r.score_file.sha256 : "", "mono"],
      ["Configuration SHA-256", r.config_sha256, "mono"], ["Purpose", cfg.purpose], ["Bootstrap resamples", num(pr.n_boot) ? pr.n_boot : cfg.n_boot],
      ["Bootstrap seed", num(pr.seed) ? pr.seed : cfg.seed], ["Code version", isObj(r.environment) ? r.environment.git_head : "", "mono"],
      ["Created", r.created], ["Finished", r.finished]]));
    if (Array.isArray(eg.caveats) && eg.caveats.length) {
      pv.appendChild(el("h4", "sec-h", "Caveats"));
      var cu = el("ul"); eg.caveats.forEach(function (c) { cu.appendChild(el("li", null, c)); }); pv.appendChild(cu);
    }
    if (isObj(eg.provenance)) { pv.appendChild(el("h4", "sec-h", "Provenance record")); pv.appendChild(el("pre", "lab-pre mono", JSON.stringify(eg.provenance, null, 1))); }
    box.appendChild(pv);

    var ex = el("div", "lab-export");
    ex.id = "ev-export";
    ex.appendChild(el("span", "lab-exl", "Export this run:"));
    [["json", "JSON (full record)"], ["md", "Markdown report"]].forEach(function (f) {
      var a = el("a", "btn", f[1]);
      a.href = "/api/evallab/runs/" + encodeURIComponent(str(r.id)) + "/export?format=" + f[0];
      a.setAttribute("download", "");
      a.setAttribute("data-format", f[0]);
      ex.appendChild(a);
    });
    box.appendChild(ex);
    out.appendChild(box);
    evButtons(false);
  }

  /* ---------------- recommended demonstrations (logged results, displayed as logged) ----------------
     Reads GET /api/evallab/recommended. Nothing is run or recomputed: the numbers come from the logged result files.
     "Open in Lab evaluation" only preselects an evaluation group in the guided flow; the run still needs its button. */
  var RECO_FILTERS = [["all", "All"], ["face_video", "Face video"], ["speech", "Speech / audio"], ["image", "Image"]];
  var RECO_METRICS = [["auc", "AUROC"], ["balanced_accuracy", "Balanced accuracy"], ["precision", "Precision"], ["recall", "Recall (sensitivity)"],
    ["specificity", "Specificity"], ["fpr", "False positive rate"], ["f1", "F1"], ["accuracy", "Accuracy"], ["eer", "EER"], ["fnr", "False negative rate"]];
  var KIND_BADGE = { stored_sealed: "Stored result (sealed test, E-014)", stored_post_freeze: "Stored result (post-freeze)" };

  function rbody() { return body("lab-reco"); }
  function loadRecommended() {
    var b = rbody();
    clear(b);
    b.appendChild(el("p", "lab-loading", "Reading the logged results ..."));
    return api("GET", "/api/evallab/recommended").then(function (x) {
      clear(b);
      if (!x.ok || !isObj(x.j) || !Array.isArray(x.j.entries)) {
        var box = el("div", "error-box");
        box.id = "reco-load-error";
        box.setAttribute("role", "alert");
        box.appendChild(el("strong", null, "The recommended demonstrations could not be loaded. "));
        box.appendChild(document.createTextNode(errMsg(x)));
        b.appendChild(box);
        return;
      }
      R.data = x.j;
      renderRecommended();
    }).catch(function (e) {
      clear(b);
      var box = el("div", "error-box", "The Evaluation Lab API could not be reached: " + (e && e.message ? e.message : "request failed"));
      box.setAttribute("role", "alert");
      b.appendChild(box);
    });
  }
  function recoLegend() {
    var ul = el("ul", "lab-explain reco-legend");
    ul.id = "reco-legend";
    [["Stored evaluation result", "A logged result, read from its result file and shown as logged. Nothing is recomputed and no sealed split is reopened."],
     ["Lab evaluation", "In the guided section below, a post-freeze group, or a predeclared group of the stored sealed-test scores, can be re-analysed: metrics are recomputed now from the stored scores. The detector model is not run again; live model inference happens only on the Analyse page for an uploaded file."],
     ["Experimental", "Every number here is a research measurement. None approves a model for the product, and none changes a verdict."]
    ].forEach(function (x) { var li = el("li"); li.appendChild(el("b", null, x[0] + ". ")); li.appendChild(document.createTextNode(x[1])); ul.appendChild(li); });
    return ul;
  }
  function recoFilterBar() {
    var bar = el("div", "reco-filter");
    bar.id = "reco-filter";
    bar.setAttribute("role", "group");
    bar.setAttribute("aria-label", "Filter by modality");
    RECO_FILTERS.forEach(function (f) {
      var n = f[0] === "all" ? R.data.entries.length : R.data.entries.filter(function (e) { return e.modality === f[0]; }).length;
      var bt = el("button", "btn reco-chip" + (R.filter === f[0] ? " on" : ""), f[1] + " (" + n + ")");
      bt.type = "button";
      bt.setAttribute("data-filter", f[0]);
      bt.setAttribute("aria-pressed", R.filter === f[0] ? "true" : "false");
      bt.addEventListener("click", function () { R.filter = f[0]; renderRecommended(); var nb = document.querySelector('#reco-filter [data-filter="' + f[0] + '"]'); if (nb) nb.focus(); });
      bar.appendChild(bt);
    });
    return bar;
  }
  function recoMetricLine(e) {
    var m = isObj(e.metrics) ? e.metrics : {};
    return ["balanced_accuracy", "recall", "specificity"].map(function (k) {
      var lbl = { balanced_accuracy: "Balanced accuracy", recall: "Recall", specificity: "Specificity" }[k];
      return lbl + " " + (isObj(m[k]) ? f4(m[k].estimate) : "not logged");
    }).join(" · ");
  }
  function recoDetail(e) {
    var d = el("div", "reco-detail");
    d.id = "reco-detail-" + e.id;
    d.appendChild(el("p", "lab-hint", str(e.kind_text)));
    var t = el("table", "lab-t");
    t.appendChild(el("caption", null, "Logged metrics, " + str(e.interval) + (e.n_seeds > 1 ? "; mean over " + e.n_seeds + " seeds" : "")));
    var hr = el("tr");
    ["Metric", "Estimate", "95% CI"].forEach(function (c) { var h = el("th", null, c); h.scope = "col"; hr.appendChild(h); });
    var th = el("thead"); th.appendChild(hr); t.appendChild(th);
    var tb = el("tbody");
    RECO_METRICS.forEach(function (k) {
      var m = isObj(e.metrics) ? e.metrics[k[0]] : null;
      if (!isObj(m)) return;
      var tr = el("tr"); tr.setAttribute("data-metric", k[0]);
      var notes = [];
      if (m.derived) notes.push("derived (see note)");
      if (k[0] === "accuracy" && num(e.n_samples) && e.n_samples > 0) notes.push("class balance " + pct(e.n_fake / e.n_samples) + " fake");
      var h = el("th", null, k[1]); h.scope = "row";
      if (notes.length) h.appendChild(el("span", "reco-note", notes.join("; ")));
      tr.appendChild(h);
      tr.appendChild(el("td", "num", f4(m.estimate)));
      tr.appendChild(el("td", "num", interval(m)));
      tb.appendChild(tr);
    });
    t.appendChild(tb);
    var w = el("div", "lab-scroll"); w.appendChild(t); d.appendChild(w);
    if (e.derived_note) d.appendChild(el("p", "lab-hint", e.derived_note));
    var thr = isObj(e.threshold) ? e.threshold : {};
    var thrVals = isObj(thr.values) ? Object.keys(thr.values).map(function (k) { return k + " " + f4(thr.values[k]); }).join("; ") : "";
    d.appendChild(kv([["Experiment", e.experiment], ["Block or group", e.block, "mono"], ["Threshold provenance", thr.provenance],
      ["Per-seed thresholds", thrVals, "mono"], ["Lab run id of the logged result", e.lab_run_id, "mono"],
      ["Source file", e.source_file, "mono"], ["Source file SHA-256", e.source_sha256, "mono"]]));
    return d;
  }
  function recoCard(e) {
    var c = el("article", "reco-card reco-" + str(e.section));
    c.setAttribute("data-reco", str(e.id));
    c.setAttribute("data-modality", str(e.modality));
    var top = el("div", "pick-top");
    top.appendChild(el("span", "role-badge reco-mod", str(e.modality_label)));
    top.appendChild(el("span", "role-badge reco-kind", KIND_BADGE[e.kind] || str(e.kind)));
    if (e.task) top.appendChild(el("span", "role-badge reco-task", "Head " + str(e.task)));
    c.appendChild(top);
    c.appendChild(el("h4", "reco-title", str(e.title)));
    c.appendChild(el("p", "reco-setting", str(e.setting)));
    if (!e.available) {
      c.appendChild(el("p", "lab-unavail", "Not available on this computer: " + str(e.reason)));
      return c;
    }
    c.appendChild(el("p", "reco-counts", e.n_real + " real · " + e.n_fake + " fake · " + e.n_samples + " total" + (num(e.n_groups) ? " · " + e.n_groups + " independent groups" : "")));
    var m = isObj(e.metrics) ? e.metrics : {};
    var pm = m[e.primary_metric || "auc"];
    var p = el("div", "reco-primary");
    p.appendChild(el("span", "ev-tile-name", "AUROC (threshold-free ranking)"));
    p.appendChild(el("b", "ev-tile-val", isObj(pm) ? f4(pm.estimate) : "not logged"));
    p.appendChild(el("span", "ev-tile-int", isObj(pm) ? "95% interval " + interval(pm) + (e.n_seeds > 1 ? ", mean of " + e.n_seeds + " seeds" : "") : ""));
    if (num(e.seed0_auc)) p.appendChild(el("span", "ev-tile-int", "Product checkpoint (seed 0): " + f4(e.seed0_auc)));
    c.appendChild(p);
    c.appendChild(el("p", "reco-line", recoMetricLine(e)));
    c.appendChild(el("p", "reco-why", str(e.why)));
    if (Array.isArray(e.caveats) && e.caveats.length) {
      var cv = el("details", "lab-caveats reco-caveats");
      cv.appendChild(el("summary", null, "Limitations and caveats (" + e.caveats.length + ")"));
      var cu = el("ul"); e.caveats.forEach(function (x) { cu.appendChild(el("li", null, x)); }); cv.appendChild(cu);
      c.appendChild(cv);
    }
    var acts = el("div", "reco-actions");
    var on = R.open === e.id;
    var sb = el("button", "btn", on ? "Hide stored result" : "Show stored result");
    sb.type = "button";
    sb.setAttribute("aria-expanded", on ? "true" : "false");
    sb.setAttribute("aria-controls", "reco-detail-" + e.id);
    sb.setAttribute("data-action", "show");
    sb.addEventListener("click", function () { R.open = on ? "" : e.id; renderRecommended(); var nb = document.querySelector('[data-reco="' + e.id + '"] [data-action="show"]'); if (nb) nb.focus(); });
    acts.appendChild(sb);
    if (isObj(e.lab)) {
      var lb = el("button", "btn", "Open in Lab evaluation");
      lb.type = "button";
      lb.setAttribute("data-action", "lab");
      lb.addEventListener("click", function () { openInLab(e.lab); });
      acts.appendChild(lb);
    }
    c.appendChild(acts);
    if (on) c.appendChild(recoDetail(e));
    return c;
  }
  function recoGrid(entries, id, emptyText) {
    var g = el("div", "reco-grid");
    g.id = id;
    entries.forEach(function (e) { g.appendChild(recoCard(e)); });
    if (!entries.length) g.appendChild(el("p", "empty", emptyText));
    return g;
  }
  function renderRecommended() {
    var b = rbody();
    clear(b);
    var all = R.data.entries.filter(function (e) { return R.filter === "all" || e.modality === R.filter; });
    var sec = function (s) { return all.filter(function (e) { return e.section === s; }); };
    b.appendChild(noteBanner(str(R.data.status) || STATUS_FALLBACK, "reco-status-banner"));
    b.appendChild(el("p", "lab-hint", "The best-supported results of the project's detectors, each with its setting, sample counts and limitations. They show where the detectors work; they are not representative of every kind of deepfake. Weak results are listed too."));
    b.appendChild(recoLegend());
    b.appendChild(recoFilterBar());
    b.appendChild(el("h3", "lab-sub", "Recommended"));
    b.appendChild(recoGrid(sec("recommended"), "reco-recommended", "No recommended demonstration for this modality."));
    b.appendChild(el("h3", "lab-sub", "Known limitations (show these too)"));
    b.appendChild(recoGrid(sec("limitation"), "reco-limitations", "No listed limitation for this modality."));
    var more = el("details", "reco-more");
    more.id = "reco-more";
    if (R.open && sec("other").some(function (e) { return e.id === R.open; })) more.open = true;
    more.appendChild(el("summary", null, "All other stored results (" + sec("other").length + ")"));
    more.appendChild(recoGrid(sec("other"), "reco-other", "No other stored result for this modality."));
    var ns = Array.isArray(R.data.not_shown) ? R.data.not_shown : [];
    if (ns.length) {
      more.appendChild(el("h4", "sec-h", "Not used as demonstrations, and why"));
      var nu = el("ul", "reco-notshown");
      nu.id = "reco-notshown";
      ns.forEach(function (x) { var li = el("li"); li.appendChild(el("b", null, str(x.title) + ". ")); li.appendChild(document.createTextNode(str(x.reason))); nu.appendChild(li); });
      more.appendChild(nu);
    }
    b.appendChild(more);
    if (R.data.note) b.appendChild(el("p", "lab-hint", R.data.note));
  }
  function openInLab(target) {
    G.ds = str(target.dataset); G.group = str(target.group); G.confirmFull = false; G.token++; G.busy = false; G.run = null;
    if (G.data) renderGuided();
    var sec = $("lab-guided");
    if (sec && sec.scrollIntoView) sec.scrollIntoView({ block: "start" });
    var s = $("ev-dataset");
    if (s) s.focus();
  }

  function open(host) {
    root = host || root || $("lab-root");
    if (!built) { build(); loadRecommended(); loadGroups(); loadCatalogue(); }
    else { if (!R.data) loadRecommended(); if (!G.data) loadGroups(); if (!S.cat) loadCatalogue(); }
    var h = $("lab-h");
    if (h) { h.setAttribute("tabindex", "-1"); }
  }
  window.evallab = { open: open, state: S, metricLabels: METRICS };
})();
